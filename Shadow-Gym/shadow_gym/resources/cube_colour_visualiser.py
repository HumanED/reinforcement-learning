import pybullet as p
import time, os
client = p.connect(p.GUI)

class Cube:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'cube.urdf')
        # Default start position.
        startPosition = [0, -1 / 3.5, 1 / 3]
        # Start position over the fingers. Same height as default
        # startPosition = [0, -1/3.5 - 0.08, 1/3]
        # startPosition tester
        # startPosition = [0, -1/3.5, 1/3]

        self.cube = p.loadURDF(f_name,
                               startPosition,
                               physicsClientId=client)
        x = p.loadTexture('cube_texture.jpg')
        p.changeVisualShape(self.cube, -1, textureUniqueId=x)


c = Cube(client)
while True:
    p.stepSimulation()
    time.sleep(1000000)
