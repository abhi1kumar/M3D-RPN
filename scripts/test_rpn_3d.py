# -----------------------------------------
# python modules
# -----------------------------------------
from importlib import import_module
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import os

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.imdb_util import *

split = "val1" # val1/val2/test

weights_path = "released/m3d_rpn_depth_aware_" + split
conf_path = "released/m3d_rpn_depth_aware_" + split + "_config.pkl"

# load config
conf = edict(pickle_read(conf_path))
conf.pretrained = None
conf.batch_size = 1

data_path = os.path.join(os.getcwd(), 'data')
results_path = os.path.join('output', split, 'data')

# make directory
mkdir_if_missing(results_path, delete_if_exist=False)

# -----------------------------------------
# torch defaults
# -----------------------------------------

# defaults
init_torch(conf.rng_seed, conf.cuda_seed)

# -----------------------------------------
# setup network
# -----------------------------------------

# net
net = import_module('models.' + conf.model).build(conf)

# load weights
load_weights(net, weights_path, remove_module=True)

# switch modes for evaluation
net.eval()

print(pretty_print('conf', conf))
print("Config file   = {}".format(conf_path))
print("Weights file  = {}".format(weights_path))
print("Output folder = {}".format(results_path))
# -----------------------------------------
# test kitti
# -----------------------------------------

test_kitti_3d(conf.dataset_test, net, conf, results_path, data_path, use_log=False)
