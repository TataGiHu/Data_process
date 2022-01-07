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
from datetime import datetime
import matplotlib.pyplot as plt
from multiprocessing import Process
import tqdm
from vis_ground3d import VisGround3d


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


def draw_ld3d(img, lane3D, color=((0, 0, 255)), is_poly=False):
    vis_handle = VisGround3d(img,
                             x_range=(-10, 10),
                             y_range=(-3, 3),
                             z_range=(0, 120),
                             bv_width=300,
                             bv_height=768)
    vis_map = vis_handle(None,
                         lane3D,
                         img,
                         img_color=color,
                         cam_color=color,
                         is_poly=is_poly)
    return vis_map


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


def run_forward_show(src_input, dst_dir, ld_rst_dir):
    # init DL-Infer
    runtime = onnxruntime.InferenceSession('ddlane_single_model_bs1.onnx')
    input_name = runtime.get_inputs()[0].name
    output_name = runtime.get_outputs()[0].name
    output_shape = runtime.get_outputs()[0].shape
    input_shape = runtime.get_inputs()[0].name
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


        with open(output_point_file_dir,'a') as file:
            file.write(json.dumps(ld_3d_dict) + "\n")



model_name = '/workspace/1215_ground_vis_debug/models/model.bin'
# test_input = '/workspace/1215_ground_vis_debug/bag_data/62113_732/img/'
test_input = "/workspace/1215_ground_vis_debug/bag_data/62113_732/PLAGD0998_event_HNP_wm_sharp_turning_filter_20211222-161634_0/sensor_camera_front_mid_cylinder_image_raw_compressed/"
ld_rst_dir = '/workspace/1215_ground_vis_debug/bag_data/62113_732/output/'
test_output = '/workspace/1215_ground_vis_debug/bag_data/62113_732/yupeng_dlinfer_float_model/'

output_point_file_dir = '/testout/' + datetime.today().strftime('%Y-%m-%d-%H_%M_%S') + ".txt"

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
    p = Process(target=run_forward_show, args=(imgs_in_process[n],
                                               test_output, ld_rst_dir))
    process_list.append(p)
    p.start()
for p in process_list:
    p.join()

print('='*100)
print('='*100)

# trans img to video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
images = sorted(os.listdir(test_output))
video = None
for img_file in tqdm.tqdm(images):
    img = cv2.imread(os.path.join(test_output, img_file))
    cv2.putText(img, img_file.split(
        '/')[-1][:-4], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    if video is None:
        try:
            h, w = img.shape[:2]
            video = cv2.VideoWriter('./temp.avi', fourcc, 20, (w, h))
        except:
            pass
    if video is not None:
        video.write(img)
video.release()

print('='*60)
print('='*60)
print('='*60)
print('trans avi to mp4')

cmd = './ffmpeg -i temp.avi ' + test_output + 'dst.mp4'
os.system(cmd)
os.system('rm temp.avi')

print('Done!')
