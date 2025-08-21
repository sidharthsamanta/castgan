
# CAST-GAN (Skeleton)

This is a minimal, **implementation-ready skeleton** for the CAST-GAN project you described.
It includes code folders, placeholder modules, and configs so you can start filling in details.

## Quickstart
```bash
# create venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (1) Train baseline ReID on source
python -m src.train_baseline --config configs/baseline.yaml

# (2) Train CAST-GAN (generator + style discriminator)
python -m src.train_castgan --config configs/castgan.yaml

# (3) Build intermediate domain (translate source & target)
python -m src.build_intermediate --config configs/castgan.yaml

# (4) Train final ReID on translated source; evaluate on real target
python -m src.train_reid --config configs/reid.yaml
python -m src.eval_target --config configs/reid.yaml
```

> This is a **skeleton**: modules compile and organize things, but you must fill in dataset loading, real training loops, and exact loss wiring per your paper.
