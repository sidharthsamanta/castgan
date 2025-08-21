
import argparse
from .utils.train_utils import load_config, set_seed

def main(cfg_path):
    cfg = load_config(cfg_path)
    set_seed(42)
    # TODO: 
    # 1) compute confidence on source; split good/bad
    # 2) build Generator, Discriminator, StyleEncoder (optional)
    # 3) wire losses: hinge adv, id, feat, entropy, src-cls, style
    # 4) train with TTUR; save checkpoints
    print('[castgan] loaded config:', cfg_path)
    print('This is a skeleton; implement your GAN training here.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    main(args.config)
