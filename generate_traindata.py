import sys
reload(sys)
sys.setdefaultencoding('UTF-8')

import os 
import json
import argparse
import numpy as np
import math
from multiprocessing import Pool

from helper.coordination_helper import CoordinateConverter
from helper.io_helper import split_data, read_files, read_dir, write_list


CAMERA_SOURCE_FRONT_MID = 1

def is_ignore_vision_lane(vision_lane):
  return (vision_lane["is_failed_3d"] or vision_lane["is_centerline"] or 
      vision_lane["camera_source"]["value"] !=  CAMERA_SOURCE_FRONT_MID )

def is_ignore_road_edge(road_edge):
  return (road_edge["is_failed_3d"] or 
      road_edge["camera_source"]["value"] != CAMERA_SOURCE_FRONT_MID)

class DataGenerator():
  def __init__(self):
    self.cc_ = CoordinateConverter()
    self.n_frame = 5
    self.feature_size = 3
    self.gt_scope_start = -20
    self.gt_scope_end = 80
    self.data_info = {
      "feature": "ax^2 + bx + c",
      "feature_size": self.feature_size,
      "n_frame": self.n_frame,
      "gt_scope": {"start": self.gt_scope_start, "end": self.gt_scope_end}
    }
  def get_data_info(self):
    return self.data_info

  def get_n_frame(self):
    return self.n_frame 

  def update_egopose(self, egopose):
     self.cc_.update(egopose["position"]["position_local"], egopose["orientation"]["quaternion_local"]) 



  def generate_gt(self, wm_lanes):
    for wm_lane in wm_lanes:
      if wm_lane["relative_id"] == 0:
        reference_line_points = wm_lane["reference_line"]["reference_line_points"]
        the_original_point_index = -1
        the_smallest_distance = 1e9
        for j, ref_point in enumerate(reference_line_points):
          enu_point = ref_point["enu_point"]
          enu_point = [enu_point["x"], enu_point["y"], enu_point["z"]]
          car_point = self.cc_.enu_to_car(enu_point)
          ref_point["car_point"] = car_point

          the_distance = math.sqrt(math.pow(car_point[0],2) + math.pow(car_point[1], 2))
          if the_distance < the_smallest_distance: 
            the_smallest_distance = the_distance
            the_original_point_index = j


        if the_original_point_index != -1:
          j = the_original_point_index
          the_back_distance = 0
          the_front_distance = 0
          all_points_x = []
          all_points_y = []
          while  j >= 1:
            j -= 1
            cur_point = reference_line_points[j]["car_point"]
            last_point = reference_line_points[j+1]["car_point"]
            the_distance = math.sqrt(math.pow(cur_point[0] - last_point[0],2) +
                                        math.pow(cur_point[1] - last_point[1],2))
            the_back_distance += the_distance
            if the_back_distance > abs(self.gt_scope_start):
              break
          the_back_index = j # 

          j = the_original_point_index
          while j + 1< len(reference_line_points):
            cur_point = reference_line_points[j]["car_point"]
            last_point = reference_line_points[j+1]["car_point"]
            the_distance = math.sqrt(math.pow(cur_point[0] - last_point[0],2) + 
                                        math.pow(cur_point[1] - last_point[1],2))
            the_front_distance += the_distance
            if the_back_distance > abs(self.gt_scope_end):
              break
            j+=1  

          the_front_index = j #
    
          for j in range(the_back_index, the_front_index+1):
            car_point = reference_line_points[j]["car_point"]
            all_points_x.append(car_point[0])
            all_points_y.append(car_point[1])

          coeff = np.polyfit(all_points_x, all_points_y, 2) 
          gt_coeff = coeff.tolist()
          gt_coeff.reverse()

          return gt_coeff

    return []

  def generate_vision_enu_point(self, vision_lane) :

    vision_lanes = vision_lane["lane_perception"]["lanes"] 
    for lane in vision_lanes:
       if is_ignore_vision_lane(lane):
         continue

       lane["points_3d_enu"] = []
       size_points_3d_x = len(lane["points_3d_x"])
       size_points_3d_y = len(lane["points_3d_y"])
       if size_points_3d_x != size_points_3d_y:
         print("lane points 3d size is not same: {}, {}".format(size_points_3d_x, size_points_3d_y))
       size_points_3d = min(size_points_3d_x, size_points_3d_y)
       for index in range(size_points_3d):
         x = lane["points_3d_x"][index]
         y = lane["points_3d_y"][index]
         point_3d = [x,y,0]
         point_3d_enu = self.cc_.car_to_enu(point_3d)
         lane["points_3d_enu"].append(point_3d_enu)
    
    road_edges = vision_lane["road_edge_perception"]["road_edges"] 
    for road_edge in road_edges:
       if is_ignore_road_edge(road_edge):
         continue

       road_edge["points_3d_enu"] = []
       size_points_3d_x = len(road_edge["points_3d_x"])
       size_points_3d_y = len(road_edge["points_3d_y"])
       if size_points_3d_x != size_points_3d_y:
         print("lane points 3d size is not same: {}, {}".format(size_points_3d_x, size_points_3d_y))
       size_points_3d = min(size_points_3d_x, size_points_3d_y)
       for index in range(size_points_3d):
         x = road_edge["points_3d_x"][index]
         y = road_edge["points_3d_y"][index]
         point_3d = [x,y,0]
         point_3d_enu = self.cc_.car_to_enu(point_3d)
         road_edge["points_3d_enu"].append(point_3d_enu)


  def generate_dt(self, i, ori_datas):


    dt_coeffs = {
      "lanes": [],
      "road_edges": []
    } 
    for j in range(i-self.n_frame+1, i+1): 
      history_vision_lanes = ori_datas[j]["vision_lane"]["lane_perception"]["lanes"] 
      frame_coeff = []
      for history_vision_lane in  history_vision_lanes:
        if is_ignore_vision_lane(history_vision_lane):
          continue

        dt_coeff = history_vision_lane["coefficient_bv"][:3]
        frame_coeff.append(dt_coeff)
      dt_coeffs["lanes"].append(frame_coeff)

      #history_vision_road_edges = ori_datas[j]["vision_lane"]["road_edge_perception"]["road_edges"] 
      #for history_vision_road_edge in  history_vision_road_edges:
      #  if is_ignore_road_edge(history_vision_road_edge):
      #    continue
      #  dt_coeff = history_vision_road_edge["coefficient_bv"][:3]
      #  dt_coeffs["road_edges"].append(dt_coeff)
    
        #points_3d_enu = history_vision_lane["points_3d_enu"]
        #for point_3d_enu in points_3d_enu:
        #  point_3d_car = cc.enu_to_car(point_3d_enu)
        #  train_data["dt"].append(point_3d_car[0])
        #  train_data["dt"].append(point_3d_car[1])


    return dt_coeffs



def main(i, file_list, save_root):

  data_generator = DataGenerator()

  n_frame = data_generator.get_n_frame()

  for j, file_path in enumerate(file_list):

    train_datas = []

    train_datas.append(json.dumps(data_generator.get_data_info()))

    ori_datas = read_files(file_path)

    for i, ori_data in enumerate(ori_datas):

      egopose = ori_data["egopose"]
      wm = ori_data["worldmodel"]
      vision_lane = ori_data["vision_lane"]
      
      data_generator.update_egopose(egopose)

      data_generator.generate_vision_enu_point(vision_lane) 


      if i+1 >= n_frame: 

        # gt
        wm_lanes = wm["processed_map_data"]["lanes"] 
        gt = data_generator.generate_gt(wm_lanes)
        if gt == []: # TODO 
          continue 

        # dt
        dt = data_generator.generate_dt(i, ori_datas)

        train_data = {
          "ts": egopose["meta"]["timestamp_us"],
          "dt": dt, 
          "gt":gt 
        } 
        train_datas.append(json.dumps(train_data))
      # time gap 
    file_name = os.path.basename(file_path).split('.')[0]
    save_file = os.path.join(save_root, file_name+".txt")
    write_list(save_file, train_datas) 


def mpl_call(bag_dir, save_root) :
  n_task = 8
  p = Pool(n_task)

  files = read_dir(bag_dir, suffix=".txt")

  splited_data = split_data(files, n_task)
  ress = []

  for i in range(n_task):
    data = splited_data[i]
    res = p.apply_async(main, args=(i, data, save_root))
    ress.append(res)

  p.close()
  p.join()

  for i, r in enumerate(ress):
    print("==============")
    print(i, r.get())

  print("parse {} done".format(bag_dir))


def single_call(bag_dir, save_root) :
  ori_datas = read_dir(bag_dir, suffix=".txt")
  main(0, ori_datas, save_root)



if __name__ == "__main__":


  parser = argparse.ArgumentParser(description="Process args.")
  parser.add_argument('--bag_dir', '-b', required=True, help='bag name')
  parser.add_argument('--save_root', '-s', required=True, help="save folder")

  args = parser.parse_args()
  bag_dir= args.bag_dir
  save_root = args.save_root

  mpl_enable = False

  if mpl_enable: 
    mpl_call(bag_dir, save_root)
  else:
    single_call(bag_dir, save_root)

























