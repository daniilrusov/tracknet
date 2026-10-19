# Положение относительно трубки: координата вдоль и лево/право

На шаге t модель получает только измеренные хиты префикса до t включительно:
[x0, y0, z0, dr, station]. Классификатор tube_logits_t1 выбирает трубку t+1.
Две дополнительные головы теперь поддерживают представление tube_relative:

| Выход | Смысл | Размер |
| --- | --- | --- |
| along_t0 | Координата частицы вдоль текущей проволоки, мм | (batch, steps, 1) |
| lr_logits_t0 | Логит положительной стороны текущей проволоки | (batch, steps, 1) |
| along_t1 | Координата вдоль проволоки следующего хита, мм | (batch, steps, 1) |
| lr_logits_t1 | Логит положительной стороны следующей проволоки | (batch, steps, 1) |

Одна Linear(hidden, 2) на текущую точку и одна на будущую. Выход along умножается
на along_coordinate_scale (по умолчанию 750 мм). LR не масштабируется:
p_positive = sigmoid(lr_logits), флаг lr=+1 при lr_logits >= 0, иначе lr=-1.
В raw V3 lr по-прежнему является только truth-полем и не входит во вход сети.

## Система координат

Для нормали n к проволоке положительное направление вдоль неё u=(-n_y,n_x):
X: вдоль +y; Y: вдоль -x; U: вдоль (-x+y)/sqrt(2); V: вдоль (x+y)/sqrt(2).
Начало вдоль проволоки — её anchor (x0,y0), ближайшая к началу координат точка.
Знак LR определяется относительно положительной нормали n и совпадает с V3.
Термины лево/право здесь задаются этой ориентацией, а не видом детектора на экране.

s = u · (truth_xy - anchor_xy)
LR = 1, если n · (truth_xy - anchor_xy) >= 0, иначе 0.

Текущая цель вычисляется из truth и геометрии текущего входного измерения.
Будущая — из truth и геометрии истинного target-класса. Последняя используется
только для построения supervision; будущая трубка не подаётся в GRU или головы.
В режиме инференса будущая локальная оценка относится к выбранной трубке;
качество её интерпретации дополнительно проверяется при правильном классе.

## Восстановление xyz

reconstruct_tube_position(along, lr_logits, measurement) принимает измерение
кандидата [x0,y0,z0,dr,station] и вычисляет

xy = anchor_xy + s*u + sign*dr*n; z = z0.

Для текущей точки измерение уже есть во входе. Для будущей необходимо выбрать
реальный хит-кандидат и получить его измеренный dr. Сам dr будущей трубки сеть
не предсказывает. Один лишь класс трубки без измеренного dr недостаточен для
получения полной точки. Истинный будущий dr не используется при forward.
soft=True возвращает среднюю оценку со знаком 2*sigmoid(logit)-1; она может
лежать между двумя физическими LR-гипотезами. По умолчанию выбирается одна ветвь.

## Данные, loss и метрики

Существующий V3-кэш с coords_current/coords_next можно использовать повторно:
локальные цели вычисляются во время обучения. Менять или пересчитывать его
не требуется. Кэш без truth-координат нужно пересоздать. Режим tube_relative
поддерживает фиксированную геометрию V3: размер 750 мм, шаг 10 мм, плоскости
X/Y/U/V/X/Y/U/V на z=0..240 мм и 1456 классов. Legacy-геометрия несовместима.

Одна и та же маска исключает warm-up и padding из CE, along-MSE, LR-BCE
и метрик. При seed_hits=2 первая учитываемая текущая точка — хит 2,
первая будущая — хит 3.

L_local = mean(((s_pred - s_true)/along_coordinate_scale)^2)
          + lr_loss_weight * BCEWithLogits(lr_logits, LR_true)
L = CE + ranking_loss_weight*ranking
    + current_coordinate_loss_weight*L_current
    + next_coordinate_loss_weight*L_next

TensorBoard содержит train/val_current/next_along_rmse_mm, lr_accuracy,
along_loss, lr_loss и суммарный loss соответствующей головы. RMSE и accuracy
накапливаются по валидным хитам всей эпохи. Веса двух голов и LR настраиваются
независимо. Общий val_loss используется scheduler/checkpoint, как прежде.

Evaluator сохраняет evaluation.tube_relative: along-RMSE/MAE, LR-accuracy/BCE,
разбивку по станциям и отдельную оценку next_when_tube_correct. XYZ-метрики
xyz_*_given_measurement получены с геометрией и dr истинного измерения.
Для будущего хита это явно условная реконструкция, а не оценка автономного
предсказания xyz без будущего измерения. Классификационные метрики сохраняются.

## Совместимость

В configs/model/straw_model.yaml включён tube_relative. Для воспроизведения
старой архитектуры задайте model.coordinate_representation=cartesian.
Конструкторы и evaluator считают отсутствующий параметр в старом checkpoint
равным cartesian; старые classifier-only и xyz checkpoints продолжают работать.
Новый режим нужно обучать с нуля: resume xyz-checkpoint как tube_relative
несовместим по структуре выходов. Resume собственного нового checkpoint работает.

## Локальные проверки и полное обучение

Короткий запуск использует уже существующий outputs/coordinates_smoke/cache:

```bash
python -m pytest tests -q
python train.py --config-name=train_v3_tube_relative_smoke model.hidden_features=32 training.accelerator=cpu training.max_epochs=2 training.limit_train_batches=3 training.limit_val_batches=2 hydra.run.dir=outputs/tube_relative_smoke/cpu
python train.py --config-name=train_v3_tube_relative_smoke training.precision=16-mixed hydra.run.dir=outputs/tube_relative_smoke/gpu
python scripts/evaluate_straw_checkpoint.py --checkpoint outputs/tube_relative_smoke/gpu/logs/straw_tracknet_v3_tube_relative_smoke/version_0/checkpoints/last.ckpt --data outputs/coordinates_smoke/test/output.tsv --output-dir outputs/tube_relative_smoke/evaluation --batch-size 128
```

Отдельный конфиг полного эксперимента, запускать позже:

```bash
python train.py --config-name=train_v3_tube_relative dataset.cache_dir=outputs/drift_sim_v3_1m_coordinates_cache
```

Короткие запуски проверяют работоспособность, а не итоговое качество.
