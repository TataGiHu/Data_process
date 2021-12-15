import os
from helper.io_helper import split_data, read_files, read_dir, write_list
from helper.coordination_helper import CoordinateConverter
from multiprocessing import Pool
import math
import numpy as np
import argparse
import json
import sys
reload(sys)
sys.setdefaultencoding('UTF-8')


INVALID_INDEX = -100
CAMERA_SOURCE_FRONT_MID = 1


def get_poly_y_value(x, coeff):
    res = 0
    for param in coeff:
        res = res * x + param
    return res


def is_ignore_vision_lane(vision_lane):
    return (vision_lane["is_failed_3d"] or vision_lane["is_centerline"] or
            vision_lane["camera_source"]["value"] != CAMERA_SOURCE_FRONT_MID)


def is_ignore_road_edge(road_edge):
    return (road_edge["is_failed_3d"] or
            road_edge["camera_source"]["value"] != CAMERA_SOURCE_FRONT_MID)


class DataGenerator():
    def __init__(self):
        self.cc_ = CoordinateConverter()
        self.n_frame = 5
        self.feature_size = 3
        self.gt_scope_start = -20
        self.step_width = 5
        self.gt_scope_end = 80
        self.data_info = {
            "type": "points",
            "feature": "descrete points",
            "n_frame": self.n_frame,
            "gt_scope": {"start": self.gt_scope_start, "end": self.gt_scope_end, "step width" : self.step_width}
        }

    def get_data_info(self):
        return self.data_info

    def get_n_frame(self):
        return self.n_frame

    def get_closest_n_points(self, x, lane, num=6):
      """ Get n + 1 closest points to the given x location on the reference line

      Args:
          x ([float]): [Desired x coordinate]
          lane ([Worldmodel Lane]): [Worldmodel Lane where the points are taken from]
          num (int, optional): [Number of points in total ]. Defaults to 6.

      Returns:
          [type]: [description]
      """          
      reference_line_points = lane["reference_line"]["reference_line_points"]
      output_points = []
      ref_line_size = len(reference_line_points)
      shortest_distance = 1E9
      index_of_closest_point = 0
      start_num = 0
      end_num = 0
      for i, reference_line_point in enumerate(reference_line_points):
          enu_point = reference_line_point["enu_point"]
          car_point = self.wm_enu_point_to_car(enu_point)
          distance_of_this_point = abs(x - car_point[0])
          if distance_of_this_point <= shortest_distance:
              shortest_distance = distance_of_this_point
              index_of_closest_point = i
          elif distance_of_this_point > shortest_distance:
              break
      index_of_point_before = index_of_closest_point - num / 2
      index_of_point_after = index_of_closest_point + num / 2

      if (index_of_point_before > 0) and (index_of_point_after < ref_line_size):
          start_num = index_of_closest_point - num / 2
          end_num = index_of_closest_point + num / 2
      elif (index_of_point_before < 0) and (index_of_point_after > ref_line_size):
          start_num = 0
          end_num = ref_line_size - 1
      elif index_of_point_before < 0:
          start_num = 0
          end_num = start_num + num
      elif index_of_point_after > ref_line_size:
          end_num = ref_line_size - 1
          start_num = end_num - num

      for i in range(start_num, end_num + 1):
          enu_point = reference_line_points[i]["enu_point"]
          car_point = self.wm_enu_point_to_car(enu_point)
          output_points.append([car_point[0], car_point[1]])

      return output_points

    def wm_enu_point_to_car(self, wm_enu_point):
        enu_point = [wm_enu_point["x"], wm_enu_point["y"], wm_enu_point["z"]]
        return self.cc_.enu_to_car(enu_point)

    def update_egopose(self, egopose):
        self.cc_.update(egopose["position"]["position_local"],
                        egopose["orientation"]["quaternion_local"])

    def generate_gt_points(self, wm_lanes):
        """[Generate descrete points along reference lines]

        Args:
            wm_lanes ([processedMap Lanes]): [Lanes from processedMap data]
            dt ([type]): [Descrete Lane points from vision perception]
        """
        gt = []
        for lane in wm_lanes:
            current_lane = []
            if not lane["relative_id"] == 0:
                gt.append(current_lane)
                continue
            for x in range(self.gt_scope_start, self.gt_scope_end, self.step_width):
                output_points = np.array(self.get_closest_n_points(x, lane, 6))
                coeff = np.polyfit(output_points[:, 0], output_points[:, 1], 2)
                gt_lane_point = [x, get_poly_y_value(x, coeff)]
                current_lane.append(gt_lane_point)
            gt.append(current_lane)
        return gt

    def generate_vision_enu_point(self, vision_lane):

        vision_lanes = vision_lane["lane_perception"]["lanes"]
        for lane in vision_lanes:
            if is_ignore_vision_lane(lane):
                continue

            lane["points_3d_enu"] = []
            size_points_3d_x = len(lane["points_3d_x"])
            size_points_3d_y = len(lane["points_3d_y"])
            if size_points_3d_x != size_points_3d_y:
                print("lane points 3d size is not same: {}, {}".format(
                    size_points_3d_x, size_points_3d_y))
            size_points_3d = min(size_points_3d_x, size_points_3d_y)
            for index in range(size_points_3d):
                x = lane["points_3d_x"][index]
                y = lane["points_3d_y"][index]
                point_3d = [x, y, 0]
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
                print("lane points 3d size is not same: {}, {}".format(
                    size_points_3d_x, size_points_3d_y))
            size_points_3d = min(size_points_3d_x, size_points_3d_y)
            for index in range(size_points_3d):
                x = road_edge["points_3d_x"][index]
                y = road_edge["points_3d_y"][index]
                point_3d = [x, y, 0]
                point_3d_enu = self.cc_.car_to_enu(point_3d)
                road_edge["points_3d_enu"].append(point_3d_enu)

    def generate_dt(self, i, ori_datas):

        dt_lane_points = {
            "lanes": [],
            "road_edges": []
        }
        for j in range(i-self.n_frame+1, i+1):
            history_vision_lanes = ori_datas[j]["vision_lane"]["lane_perception"]["lanes"]
            frame_lane_points = []
            for history_vision_lane in history_vision_lanes:
                if is_ignore_vision_lane(history_vision_lane):
                    continue
                current_lane_points = []
                for i, enu_point in enumerate(history_vision_lane["points_3d_enu"]):
                    car_point = self.cc_.enu_to_car(enu_point)
                    current_lane_points.append([car_point[0], car_point[1]])
                frame_lane_points.append(current_lane_points)
            dt_lane_points["lanes"].append(frame_lane_points)

        return dt_lane_points


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

                # dt
                dt = data_generator.generate_dt(i, ori_datas)

                gt = data_generator.generate_gt_points(wm_lanes)
                

                train_data = {
                    "ts": {
                        "egopose": egopose["meta"]["timestamp_us"],
                        "vision": vision_lane["meta"]["sensor_timestamp_us"],
                        "wm": wm["meta"]["egopose_timestamp_us"]
                    },
                    "dt": dt,
                    "gt": gt
                }
                train_datas.append(json.dumps(train_data))
            # time gap
        file_name = os.path.basename(file_path).split('.')[0]
        save_file = os.path.join(save_root, file_name+"_points.txt")
        write_list(save_file, train_datas)


def mpl_call(bag_dir, save_root):
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


def single_call(bag_dir, save_root):
    ori_datas = read_dir(bag_dir, suffix=".txt")
    main(0, ori_datas, save_root)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process args.")
    parser.add_argument('--bag_dir', '-b', required=True, help='bag name')
    parser.add_argument('--save_root', '-s', required=True, help="save folder")

    args = parser.parse_args()
    bag_dir = args.bag_dir
    save_root = args.save_root

    mpl_enable = False

    if mpl_enable:
        mpl_call(bag_dir, save_root)
    else:
        single_call(bag_dir, save_root)
