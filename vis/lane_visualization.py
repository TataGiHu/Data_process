# _*_ coding: utf-8 _*_
# @Author : Mingquan
# @File: line_visualization.py
# @Information: *Using this file to visualize the lanes line fitting
#               *INPUT: <lane_data.txt>
#               *OUTPUT: <lane_fitting.mp4>

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import sys
import matplotlib.image as mpimg
from matplotlib.animation import FFMpegWriter
matplotlib.use('TkAgg')

###################### Super Parameter Setting #############################
#设定读取车道线数据文件名 
lanes_file_name = 'PLAFB9216_event_HNP_wm_sharp_turning_filter_20211117-203333_0.txt'
#设定读取预测数据文件名
pred_file_name = ''
#设定读取实况图路径
img_source = ''
#设置是否运行时可视化（True：运行时可视化[运行慢]/Fause：仅录制[运行快]）
PAUSE_ON = True
#设置数据读取范围（都为0时默认全部读取）
start_frame_id = 0
end_frame_id = 0
#设置车道线显示范围
left_edge = -20
right_edge = 80
#设定预测显示范围
pred_left = -20
pred_right = 80
#设置视频录制帧率
writer_fps = 10
#设置自车坐标
pos_x = 0
#############################################################################



def line_visualization(this_data, this_pred, pos_x, writer, frame_id):
    
    ##数据图像呈现
    plt.subplot(212)
    plt.cla()
    #中心线定位
    gt = this_data['gt']
    ts_egopose = this_data['ts']['egopose']
    ts_vision = this_data['ts']['vision']
    if this_pred != {}:
        pred_ts = this_pred['ts']
    else:
        pred_ts = 0
    gt_a, gt_b, gt_c = gt[2], gt[1], gt[0]
    x= np.arange(pos_x+left_edge, pos_x+right_edge, 0.1)
    print(gt_a, gt_b, gt_c)
    if pred_ts == ts_egopose:
        x_pred= np.arange(pos_x+pred_left, pos_x+pred_right, 5)
        pred = this_pred['pred']
        for k in range(len(pred)):
            pred_a, pred_b, pred_c = pred[k][2], pred[k][1], pred[k][0]
            plt.plot(x_pred,[pred_a*math.pow(i,2) + pred_b*i + pred_c for i in x_pred], 'o-', color='deepskyblue', label='Prediction')
            plt.text(0,-16,'Pred_func: y = %f x^2 + %f x + %f'%(pred_a,pred_b,pred_c), ha='left', va='bottom', fontsize=10)
    plt.plot(x,[gt_a*math.pow(i,2) + gt_b*i + gt_c for i in x], color='springgreen', label='Center line',linewidth='2')
    plt.text(0,-14,'Center_func: y = %f x^2 + %f x + %f'%(gt_a,gt_b,gt_c), ha='left', va='bottom', fontsize=10)
    plt.text(0,-10,'frame_id: %d'%frame_id, ha='left', va='bottom', fontsize=10)
    plt.text(0,-12,'ts_egopose: %d'%ts_egopose, ha='left', va='bottom', fontsize=10)
    #车道线定位
    dt = this_data['dt']
    lanes = dt['lanes']
    for idx in range(len(lanes)):     
        this_frame = lanes[idx]
        for k in range(len(this_frame)):
            dt_a, dt_b, dt_c = this_frame[k][2], this_frame[k][1], this_frame[k][0]
            if abs(dt_c)<=2.5:
                plt.plot(x,[dt_a*math.pow(i,2) + dt_b*i + dt_c for i in x], '--', color='orangered')
            elif abs(dt_c)>2.5:    
                plt.plot(x,[dt_a*math.pow(i,2) + dt_b*i + dt_c for i in x], '-.k')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylim(-17,15)
    ##实况呈现
    plt.subplot(211)
    plt.cla()
    load_IMG_AND_show(img_source, ts_vision)
    if PAUSE_ON == True:
        plt.pause(0.000001)
    writer.grab_frame()

    
#车道线数据初始化
def lanes_data_init(lanes_file_name):
    if lanes_file_name != '':
        f = open(lanes_file_name,"r",encoding='utf-8')
        data = f.readlines()
        f.close()
        del data[0]
        return data
    else:
        return ''

#预测数据初始化
def pred_data_int(pred_file_name):
    if pred_file_name == '':
        return pred_file_name
    else:
        f = open(pred_file_name,"r",encoding='utf-8')
        data = f.readlines()
        f.close()
        return data

#加载图片和显示图片
def load_IMG_AND_show(img_source, ts_vision):
    if img_source == '':
        plt.axis('off')
        plt.text(0.5,0.5,'No Image Source', ha='center', va='center', fontsize=20)
    else:
        ts_vision_str = str(ts_vision)
        this_img_name = img_source + '/' + ts_vision_str + '.jpg'
        this_img = mpimg.imread(this_img_name)
        plt.imshow(this_img)
        plt.axis('off')
        
        
        
def run(lanes_data, pred_data, start_frame_id, end_frame_id, writer):
    if start_frame_id == 0 and end_frame_id == 0:
        start_frame = start_frame_id
        end_frame = len(lanes_data)
    elif start_frame_id >= 0 and start_frame_id<end_frame_id:
        start_frame = start_frame_id
        end_frame = end_frame_id
    elif start_frame_id <0 or end_frame_id<0 or start_frame_id>end_frame_id:
        print('Error: Reading range setting wrong, please reset the reading range.')
    for i in range(start_frame, end_frame):
        this_data = eval(lanes_data[i])
        if len(pred_data)>0 and i<len(pred_data):
            this_pred = eval(pred_data[i])
        elif len(pred_data) == 0:
            this_pred = {}
        frame_id = i
        line_visualization(this_data, this_pred, pos_x, writer, frame_id)
    
    

def main():
    #车道线数据读取
    lanes_data = lanes_data_init(lanes_file_name)
    if lanes_data == '':
        print('Error: No Lanes Data')
        sys.exit()
    #预测数据读取
    pred_data = pred_data_int(pred_file_name)
    metadata = dict(title='01',artist='Matplotlib',comment='Lane fitting')
    writer = FFMpegWriter(fps = writer_fps, metadata = metadata)
    figure = plt.figure(figsize=(8,8))
    plt.ion()
    record = lanes_file_name.split('.')
    video_name = record[0]+'.mp4'
    with writer.saving(figure, video_name, 100):
        run(lanes_data, pred_data, start_frame_id, end_frame_id, writer)
    
if __name__ == "__main__":
    main()