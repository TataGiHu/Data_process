
from tf import transformations as ts
import numpy as np




class CoordinateConverter: 
  def __init__(self):
    pass
  
  def update(self, position_local, quaternion_local):

    car2enu_trans = ts.translation_matrix((position_local["x"], position_local["y"], position_local["z"]))
    car2enu_rot = ts.quaternion_matrix([quaternion_local["x"], quaternion_local["y"], quaternion_local["z"], quaternion_local["w"]])
    self.car2enu_transform = ts.concatenate_matrices(car2enu_trans, car2enu_rot)
    self.enu2car_transform = ts.inverse_matrix(self.car2enu_transform)



  def car_to_enu(self, position):

    car_frame_pos = ts.translation_matrix((position[0], position[1], position[2]))
    enu_frame_pos = np.dot(self.car2enu_transform, car_frame_pos)[:3, 3]
    return enu_frame_pos 
  
  def enu_to_car(self, position):

    enu_frame_pos = ts.translation_matrix((position[0], position[1], position[2]))
    car_frame_pos = np.dot(self.enu2car_transform, enu_frame_pos)[:3, 3]
    return car_frame_pos 
 

