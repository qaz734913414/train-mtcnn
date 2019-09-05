import numpy as np
from easydict import EasyDict as edict

config = edict()
config.root = 'F:/train-mtcnn'

config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = True
config.BBOX_OHEM_RATIO = 0.7
config.LANDMARK_OHEM = False
config.LANDMARK_OHEM_RATIO = 0.7
config.landmark_L1_thresh = 0.001
config.landmark_L1_outlier_thresh = 1

config.EPS = 1e-14
config.enable_gray = True
config.enable_blur = True
config.enable_gaussian_noise = True
config.enable_color_jitter = True
config.use_landmark10 = False
config.landmark106_migu_weighting = False
config.landmark106_migu_random_flip = False
config.landmark106_migu_init_rot = False
config.enable_black_border = False
config.HeatMapSize = 32
config.HeatMapSigma = 3
config.HeatMapStage = 3
config.min_rot_angle = -45
config.max_rot_angle = 45
config.landmark_img_set = 'img_cut_celeba'
