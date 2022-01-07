from math import tan
import numpy as np 
from scipy.spatial.transform import Rotation as R


class Pcamera:
    def __init__(self, width=1024, height=768, hfov_deg=50, vfov_upper=18.1207, vfov_lower=18.1207, origin_camera_dict={}):
        self.origin_camera_dict = origin_camera_dict
        self.PCAMERA_TYPE_CYLINDER = 5
        self.M_PI = 3.14159265358979323846   
        self.intr_parm = [hfov_deg, vfov_upper, vfov_lower]
        self.width = width
        self.height = height
        self.camera_type = self.PCAMERA_TYPE_CYLINDER
        self.__initFromParams()
        pass
    
    def __initFromParams(self):
        hfov_degree, vfov_upper_degree, vfov_lower_degree = self.intr_parm.copy()
        self.cx = self.width * 0.5
        self.fx = self.width / ((hfov_degree / 180.0) * self.M_PI)
        self.fy = self.fx
        self.lower_height = tan((vfov_lower_degree * self.M_PI) / 180.0) * self.fy
        self.upper_height = tan((vfov_upper_degree * self.M_PI) / 180.0) * self.fy     
        self.height = int(self.lower_height + self.upper_height)
        self.cy = int(self.upper_height * 2) * 0.5
        
        
        self.origin_c2b_transform = self.__getOriginC2BTransform()
        self.origin_b2c_transform = np.linalg.inv(self.origin_c2b_transform)
        
        self.__rotateUpRight()
        
    def __getOriginC2BTransform(self):
        rot_mat = np.eye(4)
        rot_mat[:3,:3] = R.from_rotvec(self.origin_camera_dict['r_s2b']).as_matrix()
        rot_mat[:3,3] = self.origin_camera_dict['t_s2b']
        return rot_mat
    
    def __rotateUpRight(self):
        camhlu_to_cam_R = np.array([[0,-1,0],
                                    [0,0,-1],
                                    [1,0,0]])
        camhlu_to_body_R = self.origin_c2b_transform[:3,:3] @ camhlu_to_cam_R
        yaw = R.from_matrix(camhlu_to_body_R).as_euler('xyz')[2]
        upright_ext_camhlu_body_R = R.from_rotvec(yaw * np.array([0,0,1])).as_matrix()
        upright_ext_cam_body_R = upright_ext_camhlu_body_R @ np.linalg.inv(camhlu_to_cam_R)
        
        upright_ext_cam_body = np.eye(4)
        upright_ext_cam_body[:3,:3] = upright_ext_cam_body_R
        upright_ext_cam_body[:3, 3] = self.origin_c2b_transform[:3,3]
        
        self.ext_cam_body = upright_ext_cam_body
        self.ext_body_cam = np.linalg.inv(upright_ext_cam_body)
        
    def getCar2CamTransform(self):
        rotv = R.from_matrix(self.ext_body_cam[:3,:3]).as_rotvec()
        transv = self.ext_body_cam[:3,3]
        return rotv, transv
        
        