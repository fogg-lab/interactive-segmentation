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

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Absolute path to the checkpoint.')

    parser.add_argument('--gpu', type=int, default=0,
                        help='Id of GPU to use.')

    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use only CPU for inference.')

    parser.add_argument('--limit-longest-size', type=int, default=960,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')

    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')

    parser.add_argument('--timing', action='store_true', default=False, help='Performance timing.')

    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')

    return args


if __name__ == '__main__':
    main()
