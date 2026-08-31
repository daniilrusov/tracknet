# Запуск обучения DriftSim v3 на удалённой машине

## Что подготовлено

- train generator config: 1 000 000 событий, seed `420031`;
- независимый test config: 100 000 событий, seed `20260830`;
- v3 preprocessing без modulo и с проверкой 1456 физических классов;
- отдельный train profile с mixed precision, effective batch `4096`, early stopping и `last.ckpt`;
- перемешивание train-shards и треков меняется между эпохами; validation остаётся детерминированной;
- train/validation TorchMetrics сбрасываются между эпохами;
- evaluator автоматически распознаёт v3 и использует диапазоны `151/151/213/213/151/151/213/213`.

По локальному sample ожидаемый размер raw train TSV — около 7.3 GB, test TSV — около 0.73 GB, preprocessing cache — около 0.9 GB. Рекомендуется иметь не менее 15–20 GB свободного места с запасом под логи и checkpoints.

## 1. Обновление репозитория и окружения

```bash
cd ~/tracknet
git switch dev
git pull --ff-only origin dev

conda env update -f environment.yml --prune
conda activate tracknet
mkdir -p outputs

python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Команда должна показать Python 3.11 и доступную CUDA GPU. Если `torch.cuda.is_available()` возвращает `False`, обучение с профилем `train_v3_1m` не стартует.

## 2. Полный автоматический запуск

Рекомендуется запускать в `tmux`:

```bash
cd ~/tracknet
conda activate tracknet
tmux new -s tracknet-v3
bash scripts/run_v3_1m_pipeline.sh
```

Отключиться от `tmux`, не останавливая процесс: `Ctrl+B`, затем `D`.

Вернуться:

```bash
tmux attach -t tracknet-v3
```

Логи отдельных стадий будут лежать в `outputs/v3_1m_pipeline_logs/`.

## 3. Те же стадии отдельными командами

### Генерация миллиона train-событий

```bash
python -u scripts/driftsim_v3.py \
  --config configs/drift_sim_v3_1m.yaml \
  2>&1 | tee outputs/v3_generate_1m.log
```

Результат: `outputs/drift_sim_v3_1m/output.tsv`, а также эффективный config, metadata, SHA-256 и seed lock.

### Препроцессинг

```bash
python -u scripts/preprocess_drift_sim.py \
  --schema-version v3 \
  --input-dir outputs/drift_sim_v3_1m \
  --output-dir outputs/drift_sim_v3_1m_cache \
  --validation-split 0.1 \
  --split-seed 42 \
  --chunk-size 1000000 \
  --shard-size 100000 \
  2>&1 | tee outputs/v3_preprocess_1m.log
```

Preprocessor проверяет metadata, station/local/class ids и создаёт независимые `train/` и `validation/` shards.
Текущая схема входа сохраняет пять полей: `x0, y0, z0, dr, station`; `lr` в модель
не передаётся. Модель нормализует непрерывные признаки внутри `forward`, а `station`
использует только как индекс общего embedding плоскости X/Y/U/V. После перехода со
старой шестипризнаковой схемы cache необходимо создать заново.

### Обучение

```bash
python -u train.py --config-name=train_v3_1m \
  2>&1 | tee outputs/v3_train_1m.log
```

Профиль использует:

- batch size `8000`;
- `8` data-loader workers and shuffled train shards/tracks;
- gradient accumulation `4` — effective batch `32000`;
- `16-mixed` precision;
- максимум 1000 эпох;
- early stopping после 20 validation-эпох без улучшения;
- три лучших checkpoint и отдельный `last.ckpt`;
- geometry-aware cross-entropy: 10% целевой массы распределяется между соседними
  трубками в пределах той же станции (радиус две трубки).

Checkpoint старой модели с входом `x0,y0,z0,dr,lr,station` нельзя использовать для
resume новой архитектуры: размер входной матрицы GRU изменился. Обучение новой версии
нужно начинать с нуля.

Если GPU не хватает памяти, сохранить effective batch можно так:

```bash
python -u train.py --config-name=train_v3_1m \
  training.batch_size=1000 \
  training.accumulate_grad_batches=32 \
  2>&1 | tee outputs/v3_train_1m.log
```

Для GPU без нормальной поддержки FP16:

```bash
python -u train.py --config-name=train_v3_1m training.precision=32-true
```

## 4. Продолжение прерванного обучения

Найти последний checkpoint:

```bash
find outputs -path '*straw_tracknet_v3_1m*checkpoints/last.ckpt' -print
```

Продолжить:

```bash
python -u train.py --config-name=train_v3_1m \
  training.resume_from=/absolute/path/to/last.ckpt \
  2>&1 | tee -a outputs/v3_train_1m.log
```

## 5. Независимый v3 benchmark и подробные метрики

Legacy `outputs/test_100k` нельзя использовать с v3-моделью: у него 1208 legacy-классов. Для v3 создаётся отдельный benchmark:

```bash
python -u scripts/driftsim_v3.py \
  --config configs/drift_sim_v3_test_100k.yaml \
  2>&1 | tee outputs/v3_generate_test_100k.log
```

После обучения:

```bash
python -u scripts/evaluate_straw_checkpoint.py \
  --checkpoint /absolute/path/to/best.ckpt \
  --data outputs/test_v3_100k/output.tsv \
  --output-dir outputs/test_v3_100k/metrics \
  --batch-size 4096 \
  2>&1 | tee outputs/v3_evaluate_test_100k.log
```

Evaluator автоматически прочитает `metadata.yaml` и запишет:

- `metrics.json`: общие cross-entropy, perplexity, MRR, top-1/3/5/10 recall, station accuracy, exact-track recall и macro recall по трубкам;
- `metrics_by_group.csv`: recall по шагу предсказания, номеру target-хита 2–8, source/target station, переходу между станциями и длине трека;
- `metrics_by_tube.csv`: support и метрики каждого из 1456 tube classes;
- station-masked варианты метрик с корректными диапазонами 151 для X/Y и 213 для U/V.

Первый физический хит трека не предсказывается моделью: он является входным seed. Поэтому `target_hit_number=2` соответствует первому прогнозу после первого известного хита.

## 6. TensorBoard

```bash
tensorboard --logdir outputs --bind_all --port 6006
```

Для доступа с локальной машины безопаснее использовать SSH tunnel:

```bash
ssh -L 6006:localhost:6006 user@remote-host
```
