import os
import argparse
import sys
from tqdm import tqdm
import json
from multiprocessing import Pool,  RLock, freeze_support

def read_files(file_name):
  res = []
  with open(file_name) as fin:
    res = [x.strip() for x in fin.readlines()]
  return res
 
def read_json_files(file_name):

  res = []
  with open(file_name) as fin:
    res = [json.loads(x.strip()) for x in fin.readlines()]

  return res


def make_dirs_if_not_exist(file_path):
  if not os.path.exists(os.path.dirname(file_path)):
    os.makedirs(os.path.dirname(file_path))


def split_data(file_list, num):

  split_data = [[] for x in xrange(num)]

  for i in xrange(len(file_list)):
    index = i % num
    split_data[index].append(file_list[i])

  return split_data





def split_bags(i, root_dir, file_list, output_dir):

    for file_name in tqdm(file_list, ncols=100, desc="Process-{}".format(i), position=i):
        if "_points.txt" not in file_name:
            continue
        original_name = file_name.split("_points.txt")[0]
        file_data = read_json_files(os.path.join(root_dir, file_name))
        file_data.pop(0)

        for line in file_data:
            ts = str(line['ts']['wm'])
            gt = line['gt']
            flag = ''

            for gt_item in gt:
              if len(gt_item) == 0:
                flag += '0'
              else:
                flag += '1'
            if flag == '':
              print("the bag: {} ts: {} has nothing , please check!!".format(file_name, ts))
              continue

            out_file = os.path.join(output_dir, original_name, ts+"_"+flag+".txt")
            make_dirs_if_not_exist(out_file)
            with open(out_file, 'w') as f:
                f.write(json.dumps(line))
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process args.")
    parser.add_argument('--root_dir', '-r', required=True, help='root dir')
    parser.add_argument('--file_list', '-f', required=True, help='file_list')
    parser.add_argument("--output_dir", '-o', required=True, help="output dir" )
    args = parser.parse_args()
    
    root_dir = args.root_dir
    file_list = args.file_list
    output_dir = args.output_dir

    file_list = read_files(file_list) 
    freeze_support()

    n_task = 32
    p = Pool(n_task, initializer=tqdm.set_lock, initargs=(RLock(),))
    splited_data = split_data(file_list, n_task)

    ress = []
    for i in range(n_task):
      data = splited_data[i]
      res = p.apply_async(split_bags, args=(i, root_dir, data, output_dir))
      ress.append(res)

    p.close()
    p.join()


    #split_bags(root_dir, file_list, output_dir)
