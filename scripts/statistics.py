
# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('UTF-8')

import os 
import json
import yaml
import argparse
import numpy as np
from multiprocessing import Pool
import json
from collections import defaultdict
from pprint import pprint


def process(pid, bags_list):


  negative_lane_num = 0
  positive_lane_num = 0

  frame_num_dict = defaultdict(int)

  for i, bag_file in enumerate(bags_list):
    if pid == 0:
      print("{} / {}".format(i+1, len(bags_list)))
    cls = bag_file.split("/")[-1].split(".")[0].split("_")[-1] 
    frame_num_dict[cls] += 1 

  #print("negative_lane_num: {}".format(negative_lane_num))
  #print("positive_lane_num: {}".format(positive_lane_num))
  #print("frame_num_dict: {}".format(frame_num_dict))

  return {"negative_lane_num": negative_lane_num, 
          "positive_lane_num": positive_lane_num,
          "frame_num_dict": frame_num_dict}

def split_data(file_list, num):

  split_data = [[] for x in xrange(num)]

  for i in xrange(len(file_list)):
    index = i % num
    split_data[index].append(file_list[i])

  return split_data


if __name__ == "__main__":


  train_file = "/data/sida/train_file/0125/turning_100000.txt"

  train_bags = []
  with open(train_file, "r") as f:
    train_bags = [x.strip() for x in f.readlines()]


  n_task = 10
  p = Pool(n_task)
  splited_data = split_data(train_bags, n_task)

  ress = []
  for i in range(n_task):
    data = splited_data[i]
    res = p.apply_async(process, args=(i, data))
    ress.append(res)

  p.close()
  p.join()

  frame_num_dict = defaultdict(int)
  total_num = 0
  for i, r in enumerate(ress):
    res = r.get()
    print("pid {} ; res: {}".format(i, res))
    for k,v in res["frame_num_dict"].items():
        frame_num_dict[k] += v  
        total_num += v
  percentage  = defaultdict(str)
  for k, v in frame_num_dict.items():
    percentage[k] = str(v / (total_num *1.0) * 100 ) + "%"
  
  print(">>>>>>>>>>>>>>> the total <<<<<<<<<<<<<<<<<<")
  #print("negative_lane_num: {}".format(negative_lane_num))
  #print("positive_lane_num: {}".format(positive_lane_num))
  pprint(dict(frame_num_dict))
  pprint(dict(percentage))














