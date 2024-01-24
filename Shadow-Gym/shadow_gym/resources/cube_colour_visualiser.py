import pybullet as p
import time, os
import numpy as np
client = p.connect(p.GUI)

cubeStartPosition = [0, -1 / 3.5, 1 / 3]
cubeOrientation = [0, 0,0]
class Cube:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'cube.urdf')
        # Default start position.
        self.cube = p.loadURDF(f_name,
                               cubeStartPosition,
                               physicsClientId=client)
        x = p.loadTexture('cube_texture.jpg')
        p.changeVisualShape(self.cube, -1, textureUniqueId=x)

class Hand:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__),'shadow_hand.urdf')

        startPosition = [0, 0, 1/4]
        startOrientation = p.getQuaternionFromEuler([np.pi/2, np.pi, 0])
        self.hand = p.loadURDF(f_name,
                               startPosition,
                               startOrientation,
                               physicsClientId=client)

c = Cube(client)
cube_index = c.cube
h = Hand(client)
cube_x = p.addUserDebugParameter("cube_x",-0.5,0.5,cubeStartPosition[0])
cube_y = p.addUserDebugParameter("cube_y",-0.5,0.5,cubeStartPosition[1])
cube_z = p.addUserDebugParameter("cube_z",-0.5,0.5,cubeStartPosition[2])
while True:
    x = p.readUserDebugParameter(cube_x)
    y = p.readUserDebugParameter(cube_y)
    z = p.readUserDebugParameter(cube_z)
    p.resetBasePositionAndOrientation(cube_index,posObj=[x,y,z], ornObj=p.getQuaternionFromEuler(cubeOrientation))
    p.stepSimulation()
    time.sleep(1/60)
