
import json
import numpy as np
# import cv2
from scipy.spatial.transform import Rotation as R
import yaml
import math
import bisect
import os


class CameraProjecter():
  def project(self, point_3d):
    pass
  def project_multi(self, corners):
    pass

class PinholeCameraProjecter(CameraProjecter):
  def __init__(self, calib_file):
    calib = None
    skip_lines = 2
    with open(calib_file, 'r') as file:
      for i in range(skip_lines):
        _ = file.readline()
      calib = yaml.load(file)

    self.camT = calib["t_s2b"]
    camera_rotation = calib["r_s2b"]
    self.image_width = int(calib['width'])
    self.image_height = int(calib['height'])
    self.camR = R.from_euler('zyx', camera_rotation).as_matrix()


    self.cx = calib['cx'] 
    self.cy = calib['cy']
    self.fx = calib['fx']
    self.fy = calib['fy']

    self.camK = np.array([[self.fx,0,self.cx],
                      [0, self.fy, self.cy],
                      [0, 0, 1]])
    self.kc2 = calib['kc2']
    self.kc3 = calib['kc3']
    self.kc4 = calib['kc4']
    self.kc5 = calib['kc5']

  def get_image_size(self):
    return (self.image_width, self.image_height)

  def undistort(self, point_2d):
    u = point_2d[0]
    v = point_2d[1]
    x1 = (u-self.cx) / self.fx
    y1 = (v-self.cy) / self.fy

    k1, k2, p1, p2, k3 = self.kc2, self.kc3, self.kc4, self.kc5, 0

    r2 = x1**2 + y1**2
    x2  = x1*(1+k1*r2+k2*math.pow(r2,2)+k3*math.pow(r2,3))+2*p1*x1*y1+p2*(r2+2*x1*x1)
    y2 = y1*(1+k1*r2+k2*math.pow(r2,2)+k3*math.pow(r2,3))+p1*(r2+2*y1*y1)+2*p2*x1*y1

    u_distorted = self.fx*x2 + self.cx
    v_distorted = self.fy*y2 + self.cy
    return [u_distorted, v_distorted]

  def project(self, point_3d):
    cam_point_3d = np.array(point_3d) - self.camT
    point_world = np.transpose(cam_point_3d)
    point_with_cam = np.dot(self.camR, point_world)
    if point_with_cam[2] < 0 or point_with_cam[0] ** 2 + point_with_cam[2]**2 <1e-6:
      return False, [0,0]
    point_projected = np.dot(self.camK, point_with_cam)
    point_img = point_projected / point_with_cam[2]
    point_img = np.transpose(point_img[:2])

    point_img_undistorted = self.undistort(point_img)
    return True, point_img_undistorted

  def project_from_cam_to_car(self, point_3d):
    return self.camR @ point_3d + self.camT

  def project_multi(self, corners):

    points_2d_8 = []
    points_2d_4 = []
    all_valid = True

    for corner in corners:
      valid, point_2d  = self.project(corner)
      if point_2d[0] < 0:
        point_2d[0] = 0
      elif point_2d[0] > self.image_width:
        point_2d[0] = self.image_width

      if point_2d[1] < 0:
        point_2d[1] = 0
      elif point_2d[1] > self.image_height:
        point_2d[1] = self.image_height

      points_2d_8.append(point_2d)
      point_valid = (point_2d[0] > 0 and point_2d[0] < self.image_width)
      
      all_valid &= (valid & point_valid) 

    if all_valid:
      left = 10000
      top = 10000
      right = -1
      bottom = -1
      for point in points_2d_8:
        x = point[0]
        y = point[1]
        left = min(left, x)
        right = max(right, x)
        top = min(top, y)
        bottom = max(bottom, y)
      points_2d_4.append((left,top))
      points_2d_4.append((right,top))
      points_2d_4.append((right,bottom))
      points_2d_4.append((left,bottom))

    return all_valid, points_2d_8, points_2d_4


class CylinderCamera(CameraProjecter):
  def __init__(self, calib_file):
    calib = None
    skip_lines = 2
    with open(calib_file, 'r') as file:
      for i in range(skip_lines):
        _ = file.readline()
      calib = yaml.load(file)

    self.camT = calib["t_s2b"]
    camera_rotation = calib["r_s2b"]
    # self.cam_to_right = calib['cam_to_right']
    # self.cam_to_left = calib['cam_to_left']
    # self.cam_to_front = calib['cam_to_front']
    # self.cam_to_bottom = calib['camera_height']
    self.image_width = int(calib['width'])
    self.image_height = int(calib['height'])
    self.camR = R.from_euler('zyx', camera_rotation).as_matrix()
    # self.camR = np.array(calib['rotation_matrix']).reshape((3,3))

    self.cx = calib['cx']
    self.cy = calib['cy']
    self.fx = calib['fx']
    self.fy = calib['fy']
    self.kc2 = calib['kc2']
    self.kc3 = calib['kc3']
    self.kc4 = calib['kc4']
    self.kc5 = calib['kc5']

    self.fov_w = 180 #width fov, default 180 degree 
    self.fov_u = 30 #height up fov, default 30 degree 
    self.fov_d = 60 #height down fov, default 60 degree

    #self.cx_cylinder = self.cx
    #self.cy_cylinder = self.cy
    #self.fx_cylinder = self.fx
    #self.fy_cylinder = self.fy
    #self.image_height_cylinder = self.image_height
    #self.camK_cylinder = self.camK
    #self.camR_cylinder = self.camR

    self.init()

  def init(self):
    R_l2c_fisheye = self.extrinsic()
    self.camR = self.rotate_upright(R_l2c_fisheye)
    self.camK = self.cylinder_intrinsic()

  def get_image_size(self):
    return (self.image_width, self.image_height)

  def extrinsic(self):
    R_w2c = self.camR
    T_w2c = self.camT
    T_w2c = R_w2c.dot(T_w2c)
    R_l2c = np.eye(4)
    R_l2c[:3, :3] = R_w2c
    R_l2c[:3, 3] = T_w2c
    return R_l2c

  def eulerAnglesToRotationMatrix(self,theta):
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(theta[0]), -math.sin(theta[0])],
        [0, math.sin(theta[0]), math.cos(theta[0])]
    ])
    R_y = np.array([
        [math.cos(theta[1]), 0, math.sin(theta[1])],
        [0, 1, 0],
        [-math.sin(theta[1]), 0, math.cos(theta[1])]
    ])
    R_z = np.array([
        [math.cos(theta[2]), -math.sin(theta[2]), 0],
        [math.sin(theta[2]), math.cos(theta[2]), 0],
        [0, 0, 1]
    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

  def regular_angle(self,angle):
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < -np.pi:
        angle += 2 * np.pi
    return angle

  def rotate_upright(self,R_l2c):
    ext_cam_body = np.linalg.inv(R_l2c)
    camhlu_to_cam_R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    camhlu_to_body_R = ext_cam_body[:3, :3].dot(camhlu_to_cam_R)
    rot = R.from_matrix(camhlu_to_body_R)
    euler = rot.as_euler('zyx')
    if abs(euler[1]) + abs(euler[2]) > 0.8 * np.pi:
        yaw = self.regular_angle(euler[0] + np.pi)
    else:
        yaw = self.regular_angle(euler[0])
    #upright_ext_camhlu_body_R = self.eulerAnglesToRotationMatrix([0, 0, yaw])
    r = R.from_euler('zyx', np.array([yaw,0,0]))
    upright_ext_camhlu_body_R = r.as_matrix()

    upright_ext_cam_body_R = upright_ext_camhlu_body_R.dot(np.linalg.inv(camhlu_to_cam_R))
    R_c2l = np.zeros((4, 4))
    R_c2l[:3, :3] = upright_ext_cam_body_R
    R_c2l[:, 3] = ext_cam_body[:, 3]
    R_c2l[3, 3] = 1
    return np.linalg.inv(R_c2l)


  def cylinder_intrinsic(self):
    cx = self.image_width * 0.5
    fx = self.image_width / (self.fov_w / 180.0 * math.pi)
    fy = fx
    upper_height = math.tan(self.fov_u * math.pi / 180.0) * fy
    lower_height = math.tan(self.fov_d * math.pi / 180.0) * fy
    height = (int)(lower_height + upper_height)
    cy = int(upper_height * 2) * 0.5
    self.fx = fx
    self.fy = fy
    self.cx = cx
    self.cy = cy
    self.image_height = height
    camK_cylinder = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape((3,3))
    return camK_cylinder


  def project(self, point_3d):
    point_world = np.array([point_3d[0],point_3d[1],point_3d[2],1])
    #print("point_world: {}".format(point_world.tolist()))
    point_with_cam = np.dot(self.camR, point_world)[:3]
    #print("point_with_cam: {}".format(point_with_cam.tolist()))

    if point_with_cam[2] < 0 or point_with_cam[0] ** 2 + point_with_cam[2]**2 <1e-6:
      return False, [0,0] 
    
    #point_projected = np.dot(self.camK, point_with_cam[:3])
    #point_img = point_projected[:3] / point_with_cam[2]
    #point_img = np.transpose(point_img[:2])
    theta_x = math.atan2(point_with_cam[0], point_with_cam[2])
    yr = point_with_cam[1] / math.sqrt(point_with_cam[0]**2 + point_with_cam[2]**2)
    point_img = np.array([self.fx * theta_x + self.cx, self.fy * yr + self.cy])

    return True, point_img 

  def project_multi(self, corners):

    points_2d_8 = []
    points_2d_4 = []
    all_valid = True 
    #print("====")
    for corner in corners:
      valid, point_2d  = self.project(corner)

      if point_2d[0] < 0:
        point_2d[0] = 0
      elif point_2d[0] > self.image_width:
        point_2d[0] = self.image_width

      if point_2d[1] < 0:
        point_2d[1] = 0
      elif point_2d[1] > self.image_height:
        point_2d[1] = self.image_height

      points_2d_8.append(point_2d)
      point_valid = (point_2d[0] >= 0 and point_2d[0] <= self.image_width)

      all_valid &= (point_valid & valid)

    if all_valid:
      left = 10000
      top = 10000
      right = -1
      bottom = -1
      for point in points_2d_8:
        x = point[0]
        y = point[1]
        left = min(left, x)
        right = max(right, x)
        top = min(top, y)
        bottom = max(bottom, y)
      points_2d_4.append((left,top))
      points_2d_4.append((right,top))
      points_2d_4.append((right,bottom))
      points_2d_4.append((left,bottom))


    return all_valid, points_2d_8, points_2d_4


class CameraFactory():
  @staticmethod
  def get_camera_helper(is_cylinder, calib_file):
    camera_helper = CameraProjecter()

    if is_cylinder:
      camera_helper = CylinderCamera(calib_file)
    else:
      camera_helper = PinholeCameraProjecter(calib_file)

    return camera_helper

class LidarProjecterHelper():

  def __init__(self, calib_dir):
    
    cylinder_map = {
      "cameras": [ "camera_front_mid"],
      "tags": ["F50"]
    }
    pinhole_map  = {
      "cameras": ["camera_front_mid"],
      "tags": ["F50"]
    }
    self.projecter_helper_map = {}

    # cylinder_cameras = cylinder_map['cameras']
    # cylinder_tags = cylinder_map['tags']

    # is_cylinder = True
    # for i, tag in enumerate(cylinder_tags):
    #   calib_file = os.path.join(calib_dir, cylinder_cameras[i] + ".yaml")
    #   self.projecter_helper_map[tag] = CameraFactory.get_camera_helper(is_cylinder, calib_file)

    pinhole_cameras = pinhole_map['cameras']
    pinhole_tags = pinhole_map['tags']

    is_cylinder = False
    for i, tag in enumerate(pinhole_tags):
      calib_file = os.path.join(calib_dir, pinhole_cameras[i] + ".yaml")
      self.projecter_helper_map[tag] = CameraFactory.get_camera_helper(is_cylinder, calib_file)


  
  def project(self, corners):
    
    points_2d_all = {}
    for tag, project_helper in self.projecter_helper_map.items(): 
      valid, points_2d_8, points_2d_4 = project_helper.project_multi(corners)
      if valid:
        points_2d_all[tag] = points_2d_4
      else:
        points_2d_all[tag] = None
    return points_2d_all


if __name__ == "__main__":


  calib_dir = "/Users/liyupeng/momenta/ros-dev/script/calib/MKZ_A7WM18/calibration"

  points3d = np.array([[ 3.85331747e+01,  1.25621820e+01,  1.79999963e+00],
                       [ 3.84836191e+01,  1.45408230e+01,  1.79999963e+00],
                       [ 4.30586095e+01,  1.46554051e+01,  1.79999963e+00],
                       [ 4.31081652e+01,  1.26767640e+01,  1.79999963e+00],
                       [ 3.85331747e+01,  1.25621820e+01, -3.24467976e-07],
                       [ 3.84836191e+01,  1.45408230e+01, -3.24467976e-07],
                       [ 4.30586095e+01,  1.46554051e+01, -3.24467976e-07],
                       [ 4.31081652e+01,  1.26767640e+01, -3.24467976e-07]])
  helper = LidarProjecterHelper(calib_dir)
  point2d = helper.project(points3d)
  point3d_car = helper.projec
  print(point2d)








