
import argparse
from .utils.train_utils import load_config, set_seed

def main(cfg_path):
    cfg = load_config(cfg_path)
    set_seed(cfg['train']['seed'])
    # TODO: build dataloaders on translated source; train fresh ReID model; save ckpt
    print('[reid] loaded config:', cfg_path)
    print('This is a skeleton; implement your final ReID training here.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    main(args.config)
