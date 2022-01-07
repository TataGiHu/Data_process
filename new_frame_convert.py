
import json
import numpy as np
import argparse
from helper.coordination_helper import CoordinateConverter
from helper.io_helper import read_files
from helper.pcmera import Pcamera
# import cv2
from scipy.spatial.transform import Rotation as R

F50_CAMERA_NAME = "camera_front_mid"
SKIP_LINES = 2


class FrameTransformer:
    def __init__(self, calib):
        self.new_camera = Pcamera(height=0, origin_camera_dict=calib)
        rot_vec, trans_vec = self.new_camera.getCar2CamTransform()
        rotation_matrix = R.from_rotvec(np.array(rot_vec)).as_matrix()
        self.camR = np.eye(4)
        self.camR[:3, :3] = rotation_matrix
        self.camR[:3, 3] = trans_vec
        self.camR = np.linalg.inv(self.camR)
        return

    def tranform_points(self, points):
        modified_points = np.concatenate(
            (points, np.ones(points.shape[0]).reshape(-1, 1)), axis=1)
        return (self.camR @ modified_points.T).T[:, 0:3]


def get_closest_ld3d_points(old_points, ld3d_points_list):
    shortest_time_diff = np.infty
    closest_index = -1
    old_vision_timestamp = old_points['ts']['vision']
    # Can be optimized, check later
    for i, ld3d_point in enumerate(ld3d_points_list):
        cur_time_diff = abs(old_vision_timestamp -
                            int(ld3d_point['ts']['vision']))
        if cur_time_diff < shortest_time_diff:
            shortest_time_diff = cur_time_diff
            closest_index = i
    # print(closest_index)
    return ld3d_points_list[closest_index]["points"]


def convert_lanes_to_cam2car(lanes, frame_transformer):
    res_lanes = lanes.copy()
    for i, lane in enumerate(lanes):
        res_lanes[i] = frame_transformer.tranform_points(
            np.array(lane)).tolist()
    return res_lanes


def convert():
    old_points_list = read_files(bag_points)
    new_points_list = old_points_list.copy()
    ld3d_points_list = read_files(ld3d_points)
    meta = old_points_list[0].copy()
    n_frames = meta['n_frame']
    # n_frames = 20
    bag_enu_points_list = []
    frame_transformer = FrameTransformer(meta['calib'])
    old_points_list = old_points_list[1:]
    with open(output_file_name, 'w') as file:
        file.write(json.dumps(meta) + "\n")

        for i, old_points in enumerate(old_points_list):
            coordinate_converter.update(
                old_points['egopose']["position"]['position_local'], old_points['egopose']['orientation']["quaternion_local"])
            # print(coordinate_converter.enu2car_transform)
            points_before_frame_trans = get_closest_ld3d_points(
                old_points, ld3d_points_list)
            transformed_points = convert_lanes_to_cam2car(
                points_before_frame_trans, frame_transformer)
            frame_lanes = []
            for lane_points in transformed_points:
                fixed_carpoints = np.concatenate(
                    (np.array(lane_points), np.ones((len(lane_points), 1))), axis=1).T
                enu_lane_points = coordinate_converter.car_to_enu(
                    fixed_carpoints)
                frame_lanes.append(enu_lane_points)
            bag_enu_points_list.append(frame_lanes)
            # if i >= n_frames:
            n_frame_res = []
            for j in range(max(i - n_frames+1, 0), i+1):
                cur_frame_res = []
                for frame_lane_enu in bag_enu_points_list[j]:
                    frame_lane_car = coordinate_converter.enu_to_car(np.concatenate(
                        (frame_lane_enu, np.ones((1, frame_lane_enu.shape[1]))), axis=0))
                    cur_frame_res.append(frame_lane_car[0:2, :].T.tolist())
                n_frame_res.append(cur_frame_res)

            new_points_list[i+1]['dt'] = dict(lanes=n_frame_res, road_edges=[])
            file.write(json.dumps(new_points_list[i+1]) + "\n")
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process args.")
    parser.add_argument('--bag_points', '-bp', required=True,
                        help='Original bag points file name ')
    parser.add_argument('--ld3d_points', '-lp', required=True,
                        help="new bag points file name")
    parser.add_argument('--output_dir', '-o', required=False,
                        help="Output path", default="/home/ros/Downloads/bags/SOPPILOT/")

    # coordinate_converter = CoordinateConverter()

    args = parser.parse_args()
    bag_points = args.bag_points
    ld3d_points = args.ld3d_points
    output_dir = args.output_dir
    coordinate_converter = CoordinateConverter()
    output_file_name = output_dir + \
        bag_points.split('/')[-1].split('.')[0] + "_new.txt"

    convert()
    
    print("Done!!")
