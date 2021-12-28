
import json
import numpy as np
import argparse
from helper.io_helper import split_data, read_files, read_dir, write_list
# import cv2
from scipy.spatial.transform import Rotation as R
import yaml
import math
import bisect
import os

F50_CAMERA_NAME = "camera_front_mid"
SKIP_LINES = 2


class FrameTransformer:
  def __init__(self, calib):
    self.camT = calib["t_s2b"]
   # HardCoded, to be updated in the future
    rot_vec = [1.1976413591081210e+00, -1.2270284553620212e+00,1.2270284553620210e+00]
    trans_vec = [3.1807698470150018e-02, 1.4658220000000000e+00, -1.9146595187181716e+00]
    rotation_matrix = R.from_rotvec(np.array(rot_vec)).as_matrix()
    self.camR = np.eye(4)
    self.camR[:3,:3] = rotation_matrix
    self.camR[:3,3] = trans_vec
    self.camR = np.linalg.inv(self.camR)
    return

  def tranform_points(self, points):
    modified_points = np.concatenate((points, np.ones(points.shape[0]).reshape(-1,1)), axis=1)
    return (self.camR @ modified_points.T).T[:,0:3]

  
def get_closest_ld3d_points(old_points, ld3d_points_list):
    shortest_time_diff = np.infty
    closest_index = -1
    old_vision_timestamp = old_points['ts']['vision']
    for i, ld3d_point in enumerate(ld3d_points_list): # Can be optimized, check later
      cur_time_diff = abs(old_vision_timestamp - int(ld3d_point['ts']['vision']))
      if cur_time_diff < shortest_time_diff:
        shortest_time_diff = cur_time_diff
        closest_index = i
    print(closest_index)
    return ld3d_points_list[closest_index]["points"]


def convert_lanes_to_cam2car(lanes, frame_transformer):
    res_lanes = lanes.copy()
    for i, lane in enumerate(lanes):
      res_lanes[i] = frame_transformer.tranform_points(np.array(lane))[:,0:2].tolist()
    return res_lanes
      
def convert():
    old_points_list = read_files(bag_points)
    new_points_list = old_points_list.copy()
    ld3d_points_list = read_files(ld3d_points)
    meta = old_points_list[0].copy()
    frame_transformer = FrameTransformer(meta['calib'])
    old_points_list = old_points_list[1:]
    with open(output_file_name, 'w') as file:
      file.write(json.dumps(meta) + "\n")

      for i, old_points in enumerate(old_points_list): 
        points_before_frame_trans = get_closest_ld3d_points(old_points, ld3d_points_list)
        transformed_points = convert_lanes_to_cam2car(points_before_frame_trans, frame_transformer)
        new_points_list[i+1]['dt'] = dict(lanes=[transformed_points], road_edges=[])
        file.write(json.dumps(new_points_list[i+1]) + "\n")
    return
    

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Process args.")
  parser.add_argument('--bag_points', '-bp', required=True, help='Original bag points file name ')
  parser.add_argument('--ld3d_points', '-lp', required=True, help="new bag points file name")
  parser.add_argument('--output_dir', '-o', required=False, help="Output path", default="/home/ros/Downloads/bags/SOPPILOT/")

  args = parser.parse_args()
  bag_points = args.bag_points
  ld3d_points = args.ld3d_points
  output_dir = args.output_dir
  output_file_name = output_dir + bag_points.split('/')[-1].split('.')[0] +"_new.txt"
  
  convert()








