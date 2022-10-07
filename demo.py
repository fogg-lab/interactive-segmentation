"""
Usage:
    With GPU:
        python demo.py --checkpoint=/path/to/checkpoint.pth --gpu=0
    Without GPU:
        python demo.py --checkpoint=/path/to/checkpoint.pth --cpu
"""
import argparse
import tkinter as tk

import torch

from isegm.inference import utils
from interactive_demo.app import InteractiveDemoApp

import yaml

with open('config.yml', 'r') as stream:
    config = yaml.safe_load(stream)

def main():
    args = parse_args()

    torch.backends.cudnn.deterministic = True
    checkpoint_path = args.checkpoint
    model = utils.load_is_model(checkpoint_path, args.device, cpu_dist_maps=True)

    root = tk.Tk()
    root.minsize(960, 960)
    app = InteractiveDemoApp(root, args, model)
    root.deiconify()
    app.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.checkpoint = config['checkpoint-path']
    args.gpu = config['gpu']
    args.cpu = (config['torch-device'] == 'cpu')
    args.debug = config['debug']
    args.timing = config['timing']
    args.limit_longest_size = config['limit-longest-size']

    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')

    return args


if __name__ == '__main__':
    main()
