
import argparse
from .utils.train_utils import load_config

def main(cfg_path):
    cfg = load_config(cfg_path)
    # TODO: load trained generator; translate all source and target; save to out.inter_dir
    print('[intermediate] loaded config:', cfg_path)
    print('This is a skeleton; implement the translation export here.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    main(args.config)
