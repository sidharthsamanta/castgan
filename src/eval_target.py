
import argparse
from .utils.train_utils import load_config

def main(cfg_path):
    cfg = load_config(cfg_path)
    # TODO: load final ReID model; evaluate on real target; print Rank-1/5/10, mAP
    print('[eval] loaded config:', cfg_path)
    print('This is a skeleton; implement your evaluation here.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    main(args.config)
