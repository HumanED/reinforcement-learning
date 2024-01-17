import os
import pybullet as p
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation
target_dir = r"C:\Users\ethan\Documents\Edinburgh Uni\HumanED\HumanED Github\reinforcement-learning\Shadow-Gym\shadow_gym\resources"
client = p.connect(p.GUI)
urdf_path = os.path.join(target_dir,'cube.urdf')
texture_path = os.path.join(target_dir,'cube_texture.jpg')
start_position = [0.5, 0.5, 0.5]
start_orientation = [0, 0, 0]
cube = p.loadURDF(urdf_path, basePosition=start_position)
texture = p.loadTexture(texture_path)
p.changeVisualShape(objectUniqueId=cube, linkIndex=-1, textureUniqueId=texture)
x_param = p.addUserDebugParameter(paramName="x", rangeMin=-2*math.pi, rangeMax=2*math.pi,startValue=0)
y_param = p.addUserDebugParameter(paramName="y", rangeMin=-2*math.pi, rangeMax=2*math.pi,startValue=0)
z_param = p.addUserDebugParameter(paramName="z", rangeMin=-2*math.pi, rangeMax=2*math.pi,startValue=0)
target_orientation_quaternion = p.getQuaternionFromEuler([0,0,0])

def calculate_angular_difference(orientation1, orientation2):
    """Both orientations must use Quaternions"""
    rot1 = Rotation.from_quat(orientation1)
    rot2 = Rotation.from_quat(orientation2)

    # Calculate angular difference (radians) between two rotations
    # .inv() is transform of the rotation matrix
    angular_difference = rot1.inv() * rot2

    #  axis-angle representation of angular difference
    axis_angle = angular_difference.as_rotvec()

    # Convert angular difference into axis-angle representation
    rotation_magnitude = np.linalg.norm(axis_angle)

    return rotation_magnitude

while True:
    """
    xyz do correspond to rgb
    the eular angles are relative to global orientation
    """
    x_rotation = p.readUserDebugParameter(x_param)
    y_rotation = p.readUserDebugParameter(y_param)
    z_rotation = p.readUserDebugParameter(z_param)
    euler = [x_rotation, y_rotation, z_rotation]
    quaternion = p.getQuaternionFromEuler(euler)
    p.resetBasePositionAndOrientation(cube,posObj=start_position, ornObj=quaternion)
    current_orientation = p.getBasePositionAndOrientation(cube)[1]
    angular_difference = calculate_angular_difference(target_orientation_quaternion, current_orientation )
    print(angular_difference)
    p.stepSimulation(client)
    time.sleep(1/60)
