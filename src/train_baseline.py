
import argparse
from .utils.train_utils import load_config, set_seed

def main(cfg_path):
    cfg = load_config(cfg_path)
    set_seed(cfg['train']['seed'])
    # TODO: build dataloaders, model, losses, optimizer; train CE + Triplet
    print('[baseline] loaded config:', cfg_path)
    print('This is a skeleton; implement your training loop here.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    main(args.config)
