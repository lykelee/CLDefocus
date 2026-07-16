##################################################
# borrowed from https://github.com/nashory/pggan-pytorch
##################################################
import os

import torch
from tensorboardX import SummaryWriter

from . import utils


class TensorBoardX:
    def __init__(self, config_filename, sub_dir="", root="experiments/train/nrknet"):
        if sub_dir:
            sub_dir = '/' + sub_dir
        base = '{}{}'.format(root, sub_dir)
        os.makedirs(base, exist_ok=True)
        for i in range(1000):
            self.path = '{}/{}'.format(base, i)
            if not os.path.exists(self.path):
                print("Saving logs at {}".format(self.path))
                self.writer = {
                    'train': SummaryWriter(self.path + '/train'),
                    'val':   SummaryWriter(self.path + '/val'),
                }
                os.system('cp {} {}/'.format(config_filename, self.path))
                break

    def add_scalar(self, index, val, niter, logtype):
        self.writer[logtype].add_scalar(index, val, niter)

    def add_image_grid(self, index, ngrid, x, niter, logtype):
        grid = utils.make_image_grid(x, ngrid)
        self.writer[logtype].add_image(index, grid, niter)

    def add_image_single(self, index, x, niter, logtype):
        self.writer[logtype].add_image(index, x, niter)

    def export_json(self, out_file, logtype):
        self.writer[logtype].export_scalars_to_json(out_file)
