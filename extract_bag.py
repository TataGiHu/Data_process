# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('UTF-8')

import os 
import json
import yaml
import rosbag 
import argparse
import numpy as np
from multiprocessing import Pool
from helper import message_converter
from helper.io_helper import save_json, read_files, \
  read_dir, write_list, split_data, make_dirs
from helper.image_helper import *



def process_vision_lane(msg, all_vision_lane):

  res = {}
  res["sensor_timestamp_us"] = msg.meta.sensor_timestamp_us
  res["lane_perception"] = []
  for lane_msg in msg.lane_perception.lanes:
    if lane_msg.is_failed_3d:
      continue
    lane_res = {}
    lane_res["camera_source"] = lane_msg.camera_source.value
    lane_res["score"] = lane_msg.score
    lane_res["index"] = lane_msg.index
    lane_res["points_2d"] = []
    size_points_x = len(lane_msg.points_2d_x)
    size_points_y = len(lane_msg.points_2d_y)
    if size_points_x != size_points_y:
      print("lane points size is not same:{}, {}".format(size_points_x, size_points_y))
    size_points = min(size_points_x, size_points_y)

    for index in range(size_points):

      x = lane_msg.points_2d_x[index]
      y = lane_msg.points_2d_y[index]
      point = [x,y] 
      lane_res["points_2d"].append(point)
    lane_res["lane_type"] = lane_msg.lane_type.value
    lane_res["lane_color"] = lane_msg.lane_color.value
    lane_res["lane_width"] = lane_msg.lane_width
    lane_res["is_centerline"] = lane_msg.is_centerline
    lane_res["points_3d"] = []
    size_points_3d_x = len(lane_msg.points_3d_x)
    size_points_3d_y = len(lane_msg.points_3d_y)
    if size_points_3d_x != size_points_3d_y:
      print("lane points 3d size is not same: {}, {}".format(size_points_3d_x, size_points_3d_y))
    size_points_3d = min(size_points_3d_x, size_points_3d_y)
    for index in range(size_points_3d):
      x = lane_msg.points_3d_x[index]
      y = lane_msg.points_3d_y[index]
      point_3d = [x,y]
      lane_res["points_3d"].append(point_3d)

    lane_res["is_horizontal_line"] = lane_msg.is_horizontal_line
    lane_res["lane_horizontal_type"] = lane_msg.lane_horizontal_type.value
    lane_res["stopline_depth"] = lane_msg.stopline_depth

    res["lane_perception"].append(lane_res)

  res["road_edge_perception"] = [] 

  for road_edge_msg in msg.road_edge_perception.road_edges:
    if road_edge_msg.is_failed_3d == True:
      continue

    road_edge_res = {}
    road_edge_res["camera_source"] = road_edge_msg.camera_source.value
    road_edge_res["score"] = road_edge_msg.score
    road_edge_res["points_2d"] = []

    size_points_x = len(road_edge_msg.points_2d_x)
    size_points_y = len(road_edge_msg.points_2d_y)
    if size_points_x != size_points_y:
      print("road edge points size is not same:{}, {}".format(size_points_x, size_points_y))
    size_points = min(size_points_x, size_points_y)

    for index in range(size_points):
      x = road_edge_msg.points_2d_x[index]
      y = road_edge_msg.points_2d_y[index]
      point = [x,y] 
      road_edge_res["points_2d"].append(point)

    road_edge_res["points_3d"] = []
    size_points_3d_x = len(road_edge_msg.points_3d_x)
    size_points_3d_y = len(road_edge_msg.points_3d_y)
    if size_points_3d_x != size_points_3d_y:
      print("road points 3d size is not same: {}, {}".format(size_points_3d_x, size_points_3d_y))
    size_points_3d = min(size_points_3d_x, size_points_3d_y)
    for index in range(size_points_3d):
      x = road_edge_msg.points_3d_x[index]
      y = road_edge_msg.points_3d_y[index]
      point_3d = [x,y]
      road_edge_res["points_3d"].append(point_3d)
    res["road_edge_perception"].append(road_edge_res)

  all_vision_lane.append(json.dumps(res))
    

def process_egopose(msg, all_egopose):

  pass

def extract_data(bag_file, save_root, extract_image):

  bag = rosbag.Bag(bag_file)
  bag_name = os.path.basename(bag_file).split('.')[0]

  all_vision_lane = [] 
  all_wm_lane = [] 
  all_egopose = [] 

  vision_lane_topic = "/perception/vision/lane"
  worldmodel_topic = "/worldmodel/processed_map"
  egopose_topic = "/mla/egopose"
  image_topic = "/sensor/camera_front_mid/cylinder/image_raw/compressed"

  topic_names = [vision_lane_topic, worldmodel_topic, egopose_topic]
  image_save_root = None
  bridge = None
  if extract_image:
    topic_names.append(image_topic)
    image_dir = "_".join(image_topic.split("/")[1:])
    image_save_root = os.path.join(save_root, bag_name, image_dir)
    make_dirs(image_save_root)
    bridge = CvBridge()
    # 1. extract  the topic msg
  for topic, msg, ts in bag.read_messages(topics=topic_names): 

    data_json = message_converter.convert_ros_message_to_dictionary(msg)

    if topic == vision_lane_topic:
      all_vision_lane.append(data_json) 
    elif topic == worldmodel_topic:
      all_wm_lane.append(data_json)
    elif topic == egopose_topic:
      all_egopose.append(data_json) 
    elif topic == image_topic:
      if bridge and image_save_root:
        sec = msg.header.stamp.secs
        nsec = msg.header.stamp.nsecs
        stamp = str(sec).zfill(10)+ str(nsec).zfill(9)[:-3]
        img_name = os.path.join(image_save_root, stamp+".jpg")
        save_img_compressed(bridge, msg, img_name)


  # 2. align the msg  
  def take_wm_timestamp(elem):
    return elem["meta"]["egopose_timestamp_us"]
  all_wm_lane.sort(key=take_wm_timestamp) 
 
  def take_egopose_timestamp(elem):
    return elem["meta"]["timestamp_us"]
  all_egopose.sort(key=take_egopose_timestamp)
  def take_vision_lane_timestamp(elem):
    return elem["meta"]["sensor_timestamp_us"]
  all_vision_lane.sort(key=take_vision_lane_timestamp)



  last_time_diff = 100e6 # 100s 

  size_wm, size_egopose, size_vision_lane = len(all_wm_lane), len(all_egopose), len(all_vision_lane)
  tolerate_size = 10
  print("size_wm:{}, size_egopose:{},  size_vision_lane:{}".format(size_wm, size_egopose, size_vision_lane))

  if size_wm <= tolerate_size or size_egopose <= tolerate_size or size_vision_lane <= tolerate_size:
    return

  i=j=k=0
  max_time_diff = 50000 # 50ms 
  the_all_aligned_result = []
  i = -1
  while i + 1 < size_wm:
    i += 1  
    the_worldmodel = all_wm_lane[i]
    wm_timestamp_us = the_worldmodel["meta"]["egopose_timestamp_us"]
    last_time_diff_to_egopose = last_time_diff_to_vision_lane = 1000e6 # 1000s 

    while j < size_egopose:
      egopose_timestamp_us = all_egopose[j]["meta"]["timestamp_us"]
      time_diff = abs(wm_timestamp_us - egopose_timestamp_us)
      if time_diff < last_time_diff_to_egopose:
        last_time_diff_to_egopose =  time_diff
        j += 1
        continue
      break

    the_egopose = all_egopose[j-1]
    if abs(the_egopose["meta"]["timestamp_us"] - wm_timestamp_us) > max_time_diff:
      # egopose cannot possible break
      break
    
    add_count = 0
    while k < size_vision_lane:
      vision_lane_timestamp_us = all_vision_lane[k]["meta"]["sensor_timestamp_us"]
      time_diff = abs(wm_timestamp_us - vision_lane_timestamp_us)
      if time_diff < last_time_diff_to_vision_lane:
        last_time_diff_to_vision_lane = time_diff
        k += 1
        add_count += 1
        continue
      break

    the_vision_lane = all_vision_lane[k-1]
    if abs(the_vision_lane["meta"]["sensor_timestamp_us"] - wm_timestamp_us)  > max_time_diff:
      # the vision lane can disappear
      k -= add_count
      continue    

    aligned_result = {
      "egopose": the_egopose,
      "vision_lane": the_vision_lane,
      "worldmodel": the_worldmodel
    }
 
    the_all_aligned_result.append(json.dumps(aligned_result))
    #i += 1
  
    #print(i,j,k," wm: " , the_worldmodel["meta"]["egopose_timestamp_us"], " egopose: ", the_egopose["meta"]["timestamp_us"], 
    #      " vision_lane:", the_vision_lane["meta"]["sensor_timestamp_us"])

    #from IPython import embed
    #embed() 
  print("bag: {} has {} frame.".format(bag_name,len(aligned_result)))
  # tag = "_".join(topic_name.split("/")[1:])
  save_file = os.path.join(save_root, bag_name+".txt")
  write_list(save_file, the_all_aligned_result)


def main(i, bag_list, save_root, extract_image):

  for j, bag_file in enumerate(bag_list):
    if 0:
      try:
        extract_data(bag_file, save_root, extract_image)

      except Exception as e:
        print("error bag: {}".format(bag_file))
        print("exception: {}".format(e))
      if i == 0:
        print("parse {} : {}".format(j, bag_file))
    else:
      extract_data(bag_file, save_root, extract_image)



def mpl_call(bag_dir, save_root, extract_image) :
  n_task = 8
  p = Pool(n_task)

  bags = read_dir(bag_dir)

  splited_data = split_data(bags, n_task)
  ress = []

  for i in range(n_task):
    data = splited_data[i]
    res = p.apply_async(main, args=(i, data, save_root, extract_image))
    ress.append(res)

  p.close()
  p.join()

  for i, r in enumerate(ress):
    print("==============")
    print(i, r.get())

  print("parse {} done".format(bag_dir))


def single_call(bag_dir, save_root, extract_image) :
  bags = read_dir(bag_dir)
  main(0, bags, save_root, extract_image)



if __name__ == "__main__":


  parser = argparse.ArgumentParser(description="Process args.")
  parser.add_argument('--bag_dir', '-b', required=True, help='bag name')
  parser.add_argument('--save_root', '-s', required=True, help="save folder")
  parser.add_argument('--extract_image',  action='store_true', default=False)

  args = parser.parse_args()
  bag_dir= args.bag_dir
  save_root = args.save_root
  extract_image = args.extract_image

  mpl_enable = False

  if mpl_enable: 
    mpl_call(bag_dir, save_root, extract_image)
  else:
    single_call(bag_dir, save_root, extract_image)

