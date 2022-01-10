#########################################################################
# File Name: test_dlinfer.py
# Author: jiahui
# mail: jiahui@momenta.ai
# Created Time: 2021-12-19 13:34:16
#########################################################################
#!/usr/bin/python
import numpy as np
import os
import sys
import cv2
import json
import onnxruntime
import argparse
from datetime import datetime
# import matplotlib.pyplot as plt # For debug purpose only
from multiprocessing import Process
import tqdm


def chunk(xs, n):
    '''Split the list, xs, into n chunks'''
    L = len(xs)
    assert 0 < n <= L
    s = L // n
    return [xs[p:p + s] for p in range(0, L, s)]


def preprocess(img, net_input_h, net_input_w, dtype=np.uint8):
    image = cv2.imread(img)
    image = cv2.resize(image, (net_input_w, net_input_h))
    data = np.array(image, dtype=dtype)
    #data = data - [104, 112, 123]
    #data = np.transpose(data,(2, 0, 1))
    return data.flatten()


def parse_output_blob(blob_data, input_h, input_w):
    s4_h, s4_w = int(input_h/4), int(input_w/4)
    blob_data = blob_data.reshape((1, 14, s4_h, s4_w))[-1][-1]
    depth_map = cv2.merge([blob_data])
    depth_map = cv2.resize(depth_map, (input_w, input_h))
    return depth_map


def parse_output_blob_raw_depth(blob_data, input_h, input_w):
    s4_h, s4_w = int(input_h/4), int(input_w/4)
    blob_data = blob_data.reshape((1, 14, s4_h, s4_w))[0][-1]
    # embed()
    return blob_data


def parse_output_blob_raw_seg(blob_data, input_h, input_w):
    s4_h, s4_w = int(input_h/4), int(input_w/4)
    blob_data = blob_data.reshape((1, 14, s4_h, s4_w))[0][0]
    # embed()
    return blob_data


def parse_ld_2d_rst(json_data):
    new_lanes = []
    lanes = json_data['Lanes']
    for ln in lanes:
        pts = ln['cpoints']
        new_lanes.append(pts)
    return new_lanes  # [[{'x': xxx, 'y': yyy}, {}, ...], [], [], []]


def parse_ld_3d_rst(ld_2d, depth_map, is_raw_points=False):
    ld_3ds = []
    # default params
    black_width = 144
    im_h = 768
    fx, fy, cx, cy = 1.1734175644279260e+03, 1.1734175644279260e+03, 512.0, 384.0

    if is_raw_points:
        for pt2d in ld_2d:
            y, x = pt2d
            zc = depth_map[int(y), int(x)]
            if zc == 0 or (im_h - y) < black_width:
                continue
            th = (x - cx) / fx
            xc = zc * np.sin(th)
            # print(xc)
            yc = 0
            zc = zc * np.cos(th)
            ld_3ds.append([xc, yc, zc])
        return ld_3ds

    for ld in ld_2d:
        lane3d = []
        for pt2d in ld:
            x, y = pt2d['x'], pt2d['y']
            zc = depth_map[int(y), int(x)]
            if zc == 0 or (im_h - y) < black_width:
                continue
            th = (x - cx) / fx
            xc = zc * np.sin(th)
            yc = 0
            zc = zc * np.cos(th)
            lane3d.append([xc, yc, zc])

        if len(lane3d) > 1:
            ld_3ds.append(lane3d)
    return ld_3ds


def draw_depth_map(img, depth_map):
    h, w = depth_map.shape
    colour_begin = np.array([255, 0, 0])
    colour_end = np.array([0, 510, 0])
    diff_range = 10
    diff_array = [25]
    for i in range(5):
        diff_array.append(diff_array[-1]+5)
    for i in range(5):
        diff_array.append(diff_array[-1]+2)

    diff_range = len(diff_array)
    colour_step = (colour_end - colour_begin)/(diff_range-1)
    rate = 1

    for i in range(380, h-244):
        for j in range(w):
            if depth_map[i][j] < 20.0:
                continue
            if depth_map[i][j] > 110.0:
                continue
            val_depth = abs(depth_map[i][j])
            for k in range(diff_range):
                if abs(val_depth - diff_array[k]) < 0.5:
                    break
            if k + 1 >= diff_range:
                continue

            val_colour = colour_begin + k * colour_step
            val_colour.astype(int)
            if val_colour[1] > 255:
                val_colour[2] = val_colour[1] - 255
                val_colour[1] = 255
            b = float((img[i][j][0] + val_colour[0]*rate)/(rate+1))
            g = float((img[i][j][1] + val_colour[1]*rate)/(rate+1))
            r = float((img[i][j][2] + val_colour[2]*rate)/(rate+1))
            img[i][j] = (b, g, r)

    for i in range(diff_range):
        colour = colour_begin + i * colour_step
        if colour[1] > 255:
            colour[2] = colour[1] - 255
            colour[1] = 255
        cv2.putText(img, 'Depth > {}'.format(
            diff_array[i]), (20, h-i*40), cv2.FONT_HERSHEY_COMPLEX, 1, colour, 1)


def draw_ld2d(img, lanes):
    for points in lanes:
        for pt in points:
            x = pt['x']
            y = pt['y']
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 2)


def get_lanepoints_from_resized_segmap(segmap):
    # debug statements
    # heat_points = segmap>75
    # plt.imshow((heat_points*100).astype(int), cmap='hot', interpolation='nearest')
    # plt.savefig("heatpoints.png")

    heat_points_raw = np.where(segmap > 75)
    heat_points = np.concatenate(
        (heat_points_raw[0].reshape(-1, 1), heat_points_raw[1].reshape(-1, 1)), axis=1)
    # twod_pts_np = np.array(heat_points)
    # plt.cla()
    # plt.scatter(twod_pts_np[:,1], twod_pts_np[:,0])
    # plt.savefig("points.png")
    return heat_points.tolist()


def run_forward_show(src_input, dst_dir):
    # init DL-Infer
    runtime = onnxruntime.InferenceSession(model_path)
    input_name = runtime.get_inputs()[0].name
    output_name = runtime.get_outputs()[0].name
    net_input_h, net_input_w, net_input_c = 768, 1024, 3
    net_batch_size = 1  # only support batch 1 now
    dtype = np.float32

    batch_list = chunk(src_input, len(src_input) // net_batch_size)
    for batch in tqdm.tqdm(batch_list):
        data = np.zeros((1, net_input_c, net_input_h, net_input_w))
        # data = np.array([], dtype=dtype)
        for i, image_file in enumerate(batch):
            img_tmp = cv2.imread(image_file)
            np.transpose(img_tmp, (2, 0, 1))
            data[0] = np.transpose(img_tmp, (2, 0, 1)).astype(dtype)

        data = data.astype(dtype)
        name = os.path.basename(image_file)

        input_dict = {input_name: data}
        # inference
        outputs = runtime.run([output_name], input_dict)

        output_blob = outputs[0]
        depth_map = parse_output_blob(output_blob, net_input_h, net_input_w)
        depth_map_raw = parse_output_blob_raw_depth(
            output_blob, net_input_h, net_input_w)
        seg_map_raw = parse_output_blob_raw_seg(
            output_blob, net_input_h, net_input_w)
        # plt.imshow(seg_map_raw, cmap='hot', interpolation='nearest')
        # plt.savefig("heatmap.png")
        resized_segmap = cv2.resize(seg_map_raw, (net_input_w, net_input_h))
        # plt.imshow(resized_segmap, cmap='hot', interpolation='nearest')
        # plt.savefig("heatmap.png")
        twod_pts = get_lanepoints_from_resized_segmap(resized_segmap)

        ld_3d_new = parse_ld_3d_rst(twod_pts, depth_map, True)
        time_string = batch[0].split("/")[-1].split('.')[0]
        ld_3d_dict = dict(points=[ld_3d_new], ts=dict(vision=time_string))

        with open(dst_dir, 'a') as file:
            file.write(json.dumps(ld_3d_dict) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process args.")
    parser.add_argument('--bag_images_dir', '-b', required=True,
                        help='Original bag points file directory ')
    parser.add_argument('--output_dir', '-o', required=False,
                        help="Output path", default="/testout/")

    args = parser.parse_args()

    model_path = '/workspace/1215_ground_vis_debug/ddlane_single_model_bs1.onnx'
    test_input = args.bag_images
    time_now = datetime.today().strftime('%Y-%m-%d-%H_%M_%S')
    process_num = 3
    src_imgs = os.listdir(test_input)
    src_imgs = [os.path.join(test_input, file_name) for file_name in src_imgs]
    print('all img num = ', len(src_imgs))
    imgs_in_process = [[] for n in range(process_num)]
    n = 0
    for img in src_imgs:
        imgs_in_process[n % process_num].append(img)
        n += 1

    process_list = []
    for n in range(process_num):
        test_output = args.output_dir + \
            imgs_in_process[n][0].split('62113_732')[1].split(
                '/')[1] + time_now + ".txt"
        p = Process(target=run_forward_show, args=(imgs_in_process[n],
                                                   test_output))
        process_list.append(p)
        p.start()
    for p in process_list:
        p.join()

    print('='*100)
    print('='*100)

    # trans img to video

    print('='*60)
    print('='*60)
    print('='*60)
    print('trans avi to mp4')

    print('Done!')
