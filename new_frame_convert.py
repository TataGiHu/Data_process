
import json
import numpy as np
# import cv2
from scipy.spatial.transform import Rotation as R
import yaml
import math
import bisect
import os

F50_CAMERA_NAME = "camera_front_mid"
SKIP_LINES = 2

class FrameTransformer:
  def __init__(self, calib_dir):
    with open(calib_dir + F50_CAMERA_NAME +".yaml", "r") as file:
      for i in range(SKIP_LINES):
        _ = file.readline()
      calib = yaml.load(file)
    self.camT = calib["t_s2b"]
    camera_rotation = calib["r_s2b"]
    self.camR = R.from_euler('zyx', camera_rotation).as_matrix()
    # self.camR = np.array(calib['rotation_matrix']).reshape((3,3))
    pass

  def tranform_points(self, points):
    return (self.camR @ points.transpose()).transpose() + self.camT


if __name__ == "__main__":


  calib_dir = "/Users/liyupeng/momenta/ros-dev/script/calib/MKZ_A7WM18/calibration/"

  points3d = np.array([[ 3.85331747e+01,  1.25621820e+01,  1.79999963e+00],
                       [ 3.84836191e+01,  1.45408230e+01,  1.79999963e+00],
                       [ 4.30586095e+01,  1.46554051e+01,  1.79999963e+00],
                       [ 4.31081652e+01,  1.26767640e+01,  1.79999963e+00],
                       [ 3.85331747e+01,  1.25621820e+01, -3.24467976e-07],
                       [ 3.84836191e+01,  1.45408230e+01, -3.24467976e-07],
                       [ 4.30586095e+01,  1.46554051e+01, -3.24467976e-07],
                       [ 4.31081652e+01,  1.26767640e+01, -3.24467976e-07]])
  frame_transformer = FrameTransformer(calib_dir)
  transformed_points = frame_transformer.tranform_points(points3d)
  print(transformed_points)








