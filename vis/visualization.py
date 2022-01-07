# _*_ coding: utf-8 _*_
# @Author : Mingquan
# @File: line_visualization.py
# @Information: *Using this file to visualize the lanes line fitting
#               *INPUT: <lane_data.txt>/<pred_data.txt>/<image_source/>
#               *OUTPUT: <lane_fitting.mp4>

from os import write
import sys
import importlib
importlib.reload(sys)

# import time
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import matplotlib.image as mpimg
from matplotlib.animation import FFMpegWriter
matplotlib.use('TkAgg')


class VisualizationTool:
    
    def __init__(self, lane_src, pred_src, img_folder, opt_name, writer, PAUSE_ON):
        self.lane_data = self.dataProcess(lane_src)
        self.pred_data = self.dataProcess(pred_src)
        self.img_folder = img_folder
        self.data_len = len(self.lane_data)
        if self.pred_data !=None:   
            self.pred_len = len(self.pred_data)
        self.lane_type = self.lane_data[0]['type']
        if self.pred_data != None:
            self.pred_type = self.pred_data[0]['type']
        self.writer = writer
        self.PAUSE_ON = PAUSE_ON
        temp = lane_src.split('.')
        if opt_name == None:
            self.output_name = temp[0]
        else:
            self.output_name = opt_name
        
    def dataProcess(self, file_name):
        if file_name != None:
            f = open(file_name, 'r', encoding='utf-8')
            data = []
            for line in f:
                data.append(json.loads(line))
            f.close()
            print(data[0])
            return data
        else:
            return None
    
    def currentLane(self, idx):
        cur_lane = self.lane_data[idx]
        return cur_lane  
    
    def initTS(self, idx):
        this_lane = self.currentLane(idx)
        self.ts_egopose = this_lane['ts']['egopose']
        self.ts_vision = this_lane['ts']['vision']
        self.ts_wm = this_lane['ts']['wm'] 
    
    def showALL(self, idx):
        plt.ion()
        plt.subplot(121)
        plt.cla()
        self.showIMG()   
        plt.subplot(122)
        ax = plt.gca()
        ax.set_facecolor('darkgray')
        plt.cla()
        plt.grid(True)
        plt.ylim(-20,80)
        plt.xlim(-20,20)
        self.showGT(idx)
        self.showDT(idx)
        self.showCarCoordinateSystem()
        if self.pred_data != None:
            self.showPred()
        plt.text(-18,-13,'frame_id: %d'%idx, ha='left', va='bottom', fontsize=8)
        plt.text(-18,-15,'ts_egopose: %d'%self.ts_egopose, ha='left', va='bottom', fontsize=8)
        plt.text(-18,-17,'ts_vision: %d'%self.ts_vision, ha='left', va='bottom', fontsize=8)
        plt.text(-18,-19,'ts_wm: %d'%self.ts_wm, ha='left', va='bottom', fontsize=8)
        self.color_tag()
        plt.legend(loc='lower right', fontsize=8)
        if self.PAUSE_ON == True:
            plt.pause(0.000001)
    
    def showCarCoordinateSystem(self):
        car_X = [1.5,-1.8,-1.8,1.5,2,2,1.5]
        car_Y = [1,1,-1,-1,-0.8,0.8,1]
        light_X = [18,1.8,1.8,18]
        light_Y = [3,0.8,-0.8,-3]
        geer_X = [[-1.5,-1.5,-0.7,-0.7],[-1.5,-1.5,-0.7,-0.7],[1.5,1.5,0.7,0.7],[1.5,1.5,0.7,0.7]]
        geer_Y = [[1,1.3,1.3,1],[-1,-1.3,-1.3,-1],[1,1.3,1.3,1],[1,-1.3,-1.3,-1]]
        plt.plot(car_Y,car_X,'-',linewidth='1',color='k')
        for i in range(4):
            plt.plot(geer_Y[i],geer_X[i],'-',linewidth='1',color='k')
        plt.plot(light_Y,light_X,'-.',linewidth='1',color='yellow')
        plt.plot(0,0,'.',color='k')

           
    def showGT(self, idx):
        this_lane = self.currentLane(idx)
        gt = this_lane['gt']
        if self.lane_type == 'points':
            for line in gt:
                X_gt = []
                Y_gt = []
                for point in line:
                    X_gt.append(point[0])
                    Y_gt.append(point[1])
                Y_gt_new = [i*-1 for i in Y_gt]
                plt.plot(Y_gt_new, X_gt,'-', color='w',linewidth='1.5')
        elif self.lane_type == 'coeff':
            for line in gt:
                gt_a, gt_b, gt_c = line[2], line[1], line[0]
                X_gt = range(-20,80)
                Y_gt = [gt_a*math.pow(x,2)+gt_b*x+gt_c for x in X_gt]
                Y_gt_new = [i*-1 for i in Y_gt]
                plt.plot(Y_gt_new, X_gt, '-', color='w',linewidth='1.5')
                plt.text(-18,60,'gt_func: y = %f x^2 + %f x + %f'%(gt_a,gt_b,gt_c), ha='left', va='bottom', fontsize=8)
    
    def showLanesOrEdges_points(self, items, col):
        for frame in items:
            for line in frame:
                X_dt = []
                Y_dt = []
                for point in line:
                    X_dt.append(point[0])
                    Y_dt.append(point[1])
                Y_dt_new = [i*-1 for i in Y_dt]
                plt.plot(Y_dt_new, X_dt,'--', color=col, linewidth='1')
                # if abs(line[0][1])>2.5:
                #     X_dt = []
                #     Y_dt = []
                #     for point in line:
                #         X_dt.append(point[0])
                #         Y_dt.append(point[1])
                #     Y_dt_new = [i*-1 for i in Y_dt]
                #     plt.plot(Y_dt_new, X_dt,'--', color=col, linewidth='1')
                # elif abs(line[0][1])<=2.5:
                #     X_dt = []
                #     Y_dt = []
                #     for point in line:
                #         X_dt.append(point[0])
                #         Y_dt.append(point[1])
                #     Y_dt_new = [i*-1 for i in Y_dt]
                #     plt.plot(Y_dt_new, X_dt,'--', color=cur_lane_col, linewidth='1')
    
    def showLanesOrEdges_coeff(self, items, col):
        for frame in items:
            for line in frame:
                dt_a, dt_b, dt_c = line[2], line[1], line[0]
                X_dt = range(-20,80)
                Y_dt = [dt_a*math.pow(x,2)+dt_b*x+dt_c for x in X_dt]
                Y_dt_new = [-1*i for i in Y_dt]
                plt.plot(Y_dt_new, X_dt, '--', color=col,linewidth='1')
                # if abs(dt_c)<=2.5:
                #     plt.plot(Y_dt_new, X_dt, '--', color=cur_lane_col,linewidth='1')
                # elif abs(dt_c)>2.5:
                #     plt.plot(Y_dt_new, X_dt, '--', color=col,linewidth='1')
    
    def showDT(self,idx):
        this_lane = self.currentLane(idx)
        dt = this_lane['dt']
        lanes = dt['lanes']
        road_edges = dt['road_edges']
        if self.lane_type == 'points':
            self.showLanesOrEdges_points(lanes,'m')
            self.showLanesOrEdges_points(road_edges,'r')
        elif self.lane_type == 'coeff':
            self.showLanesOrEdges_coeff(lanes,'m')
            self.showLanesOrEdges_coeff(road_edges,'r')
    
    def showScore(self, score, position_x, position_y):
        # for idx in range(len(score)):
        #     plt.text(position_x,position_y,'pred_socre_%d: %f'%(idx+1,score[idx]), ha='left', va='bottom',fontsize=8)
        #     position_y = position_y-2
        plt.text(position_x,position_y,'pred_socre_%s: %f'%('<',score[0]), ha='left', va='bottom',fontsize=8)
        plt.text(position_x,position_y-2,'pred_socre_%s: %f'%('+',score[1]), ha='left', va='bottom',fontsize=8)
        plt.text(position_x,position_y-4,'pred_socre_%s: %f'%('>',score[2]), ha='left', va='bottom',fontsize=8)
                    
    def showPred(self):
        for item in self.pred_data:
            if 'ts'in item.keys() and self.ts_egopose == item['ts']:
                this_pred = item
                pred = this_pred['pred']
                score = this_pred['score']
                if self.pred_type == 'points':
                    for i,line in enumerate(pred):
                        X_pred = []
                        Y_pred = []
                        for point in line:
                            X_pred.append(point[0])
                            Y_pred.append(point[1])
                        Y_pred_new = [-1*i for i in Y_pred]
                        if i == 0:   
                            plt.plot(Y_pred_new, X_pred, '<--', color='dodgerblue',linewidth='0.8',label=None)
                        elif i == 1:
                            plt.plot(Y_pred_new, X_pred, 'P--', color='dodgerblue',linewidth='0.8',label=None)
                        elif i == 2:
                            plt.plot(Y_pred_new, X_pred, '>--', color='dodgerblue',linewidth='0.8',label=None)
                    self.showScore(score,-18,75)
                elif self.pred_type == 'coeff':
                    for line in pred:
                        pred_a, pred_b, pred_c = line[2], line[1], line[0]
                        X_pred = range(-20,80)
                        Y_pred = [pred_a*math.pow(x,2)+pred_b*x+pred_c for x in X_pred]
                        Y_pred_new = [-1*i for i in Y_pred]
                        plt.plot(Y_pred_new, X_pred, '.--', color='dodgerblue',linewidth='1',label='pred')
                        # plt.text(-18,13,'pred_func: y = %f x^2 + %f x + %f'%(pred_a,pred_b,pred_c), ha='left', va='bottom', fontsize=8)
                    self.showScore(score,-18,75)
                break
                
    def showIMG(self):
        if self.img_folder != None:
            ts_vision_str = str(self.ts_vision)
            this_img_name = self.img_folder + '/' + ts_vision_str + '.jpg'
            img = mpimg.imread(this_img_name)
            plt.axis('off')
            plt.imshow(img,extent=(-110,110,-100,100))
        else:
            plt.axis('off')
            plt.text(0.5,0.5,"No Image Source", ha='center', va='center', fontsize=20)
    
    def color_tag(self):
        plt.plot(-500, 500, '-', color='w',linewidth='2',label='gt')
        plt.plot(-500, 500, '--', color='m',linewidth='2',label='lanes')    
        plt.plot(-500, 500, '--', color='r',linewidth='2',label='road_edges')
        plt.plot(-500, 500, '.', color='k',linewidth='0.8',label='car_coordinate_sys')
        if self.pred_data != None:
            plt.plot(-500, 500, '.-', color='dodgerblue',linewidth='0.8',label='pred')
        
          
    def doVisualization(self):
        fig = plt.figure(figsize=(13, 10))
        video_name = self.output_name + '.mp4'
        with self.writer.saving(fig, video_name, 100):
            for idx in range(1,self.data_len):
                self.initTS(idx)
                self.showALL(idx)
                self.writer.grab_frame()
                print(f'----  writing: {idx}/{self.data_len}  ----')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process args.')
    parser.add_argument('--lane_src', '-l', required=True, help='lane source file name')
    parser.add_argument('--pred_src', '-p', required=False, help='prediction source file name')
    parser.add_argument('--img_folder', '-i', required=False, help='image source folder')
    parser.add_argument('--output_name', '-o', required=False, help='output video name')
    args = parser.parse_args()
    lane_src = args.lane_src
    pred_src = args.pred_src
    img_folder = args.img_folder
    opt_name = args.output_name

    
    
    
    metadata = dict(title='lane_data', artist='Matplotlib', comment='Lane_visualization')
    writer = FFMpegWriter(fps=8, metadata=metadata)
    PAUSE_ON = False
    vis = VisualizationTool(lane_src, pred_src, img_folder, opt_name, writer, PAUSE_ON)
    vis.doVisualization()
