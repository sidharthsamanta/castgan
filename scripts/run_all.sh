
#!/usr/bin/env bash
set -e
python -m src.train_baseline --config configs/baseline.yaml
python -m src.train_castgan --config configs/castgan.yaml
python -m src.build_intermediate --config configs/castgan.yaml
python -m src.train_reid --config configs/reid.yaml
python -m src.eval_target --config configs/reid.yaml
