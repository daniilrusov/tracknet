# Две координатные головы StrawTrackNET

На шаге `t` GRU получает только последовательность измеренных трубок до `t`
включительно (`x0, y0, z0, dr, station`). От одного скрытого состояния работают:

- `tube_logits_t1`: класс трубки следующего хита `t+1`;
- `coords_t0`: истинные координаты частицы `x, y, z` в текущем хите `t`,
  последнем хите текущего трека-кандидата;
- `coords_t1`: истинные координаты частицы в следующем хите `t+1`, чья
  трубка классифицируется на этом же шаге.

Координаты частицы не входят во вход модели. Будущая трубка тоже не передаётся
координатной голове. Выходы координатных голов имеют размер `(batch, steps, 3)`
и выражены в миллиметрах. Головы — отдельные Linear-слои после общего GRU.

## Данные и loss

Препроцессор сохраняет `coords_current = truth[:-1]` и `coords_next = truth[1:]`
в отдельные тензоры каждого shard. Для V3 используются именно `x, y, z`,
а не координаты центра проволоки. Legacy-генераторы не имеют `z`; для них
используется глубина плоскости `z0` с сохранением прежней геометрии.

Все три задачи используют одну маску: после двух seed-хитов первый учитываемый
шаг классифицирует хит 3, восстанавливает координаты хита 2 и предсказывает
координаты хита 3. Warm-up и padding исключаются из всех losses и метрик.

`loss = CE + ranking_loss_weight * ranking_loss
             + current_coordinate_loss_weight * MSE_current
             + next_coordinate_loss_weight * MSE_next`

Обе MSE считаются по истинным координатам, с делением ошибки на
`coordinate_scale = [750, 750, 120]` и усреднением по валидным хитам и трём осям.
Начальные веса обеих задач равны `1.0`; классификация и ranking сохраняются.
Нормировка выхода: `xyz = normalized_head * scale + center`,
`center = [0, 0, 120]`. Значения предсказаний не ограничиваются активацией.

TensorBoard содержит отдельные `train/val_coords_current_loss`,
`train/val_coords_next_loss` и `train/val_coords_current/next_rmse_mm`.
RMSE — корень из средней квадратной евклидовой ошибки по валидным хитам,
с накоплением по всей эпохе и сбросом между эпохами. Scheduler и checkpoint
мониторят общий `val_loss`, включая обе координатные задачи.

Evaluator сохраняет в `metrics.json -> evaluation.coordinates` метрики
для `current` и `next`: число валидных хитов, RMS и среднюю евклидову ошибку,
RMSE и MAE отдельно по `x, y, z`. Классификационные отчёты сохраняются.

## Локальная проверка

```bash
python scripts/driftsim_v3.py -n 500 -o outputs/coordinates_smoke/raw --seed 131026
python scripts/preprocess_drift_sim.py --schema-version v3 --input-dir outputs/coordinates_smoke/raw --output-dir outputs/coordinates_smoke/cache --validation-split 0.2 --chunk-size 5000 --shard-size 1000
python -m pytest tests -q
python train.py --config-name=train_v3_coordinates_smoke model.hidden_features=32 training.accelerator=cpu training.max_epochs=1 training.limit_train_batches=3 training.limit_val_batches=2 hydra.run.dir=outputs/coordinates_smoke/cpu
python train.py --config-name=train_v3_coordinates_smoke training.precision=16-mixed hydra.run.dir=outputs/coordinates_smoke/gpu
```

Для независимой проверки после короткого обучения:

```bash
python scripts/driftsim_v3.py -n 100 -o outputs/coordinates_smoke/test --seed 131027
python scripts/evaluate_straw_checkpoint.py --checkpoint outputs/coordinates_smoke/gpu/logs/straw_tracknet_v3_coordinates_smoke/version_0/checkpoints/last.ckpt --data outputs/coordinates_smoke/test/output.tsv --output-dir outputs/coordinates_smoke/evaluation --batch-size 128
```

Короткие запуски проверяют работоспособность, а не итоговое качество модели.

## Полное обучение позже

В `configs/model/straw_model.yaml` координатные головы включены по умолчанию.
Существующий raw V3 dataset можно использовать повторно, но кэш обязательно
пересоздать: старые shards не содержат координатных целей. Новый эксперимент
начать с нуля, без `training.resume_from` от модели без координатных голов:

```bash
python scripts/preprocess_drift_sim.py --schema-version v3 --input-dir outputs/drift_sim_v3_1m --output-dir outputs/drift_sim_v3_1m_cache --validation-split 0.1 --split-seed 42 --chunk-size 1000000 --shard-size 100000
python train.py --config-name=train_v3_1m experiment.name=straw_tracknet_v3_1m_coordinates
```

Resume нового checkpoint поддерживается обычным `training.resume_from`.
Старые checkpoint доступны для инференса: отсутствие `predict_coordinates`
в их гиперпараметрах означает прежнюю модель без координатных голов.
Для классификационного режима можно задать `model.predict_coordinates=false`;
старые кэши в этом режиме продолжают работать.
