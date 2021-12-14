import sys
reload(sys)
sys.setdefaultencoding('UTF-8')

import os 
import json





def split_data(file_list, num):

  split_data = [[] for x in xrange(num)]

  for i in xrange(len(file_list)):
    index = i % num
    split_data[index].append(file_list[i])

  return split_data



def read_files(file_name):

  res = []
  with open(file_name) as fin:
    res = [json.loads(x.strip()) for x in fin.readlines()]

  return res


def make_dirs_if_not_exist(file_path):
  if not os.path.exists(os.path.dirname(file_path)):
    os.makedirs(os.path.dirname(file_path))

def make_dirs(path):
  if not os.path.exists(path):
    os.makedirs(path)


def save_json(json_path, res):
  make_dirs_if_not_exist(json_path)
  with open(json_path, 'w') as fout:
    fout.write(json.dumps(res))  



def write_list(file_name, ress):
  make_dirs_if_not_exist(file_name)

  with open(file_name, 'w') as fout:
    for res in ress:
      fout.write(res+"\n")
  print("write to {}".format(file_name))

def read_dir(bag_dir, suffix=".bag"):

  res = []
  for x in os.listdir(bag_dir):
    if x.endswith(suffix):
      res.append(os.path.join(bag_dir, x))

  return res
