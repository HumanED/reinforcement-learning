import numpy as np
from time import sleep
import pybullet as p
import pybullet_data

# Connect to the physics client
client = p.connect(p.GUI)
# client = p.connect(p.DIRECT) # Use this to run faster simulations

# Define the gravity for the simulation
p.setGravity(0, 0, -10)

# Load the base plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Load the simulated hand
startPosition = [0, 0, 1 / 4]
startOrientation = p.getQuaternionFromEuler([np.pi / 2, np.pi, np.pi / 8])
hand = p.loadURDF("urdfs/shadow_hand/shadow_hand.urdf", startPosition, startOrientation)


def get_join_info():
    """
    Retrieves each joint in the urdf file and provides some info
    """
    number_of_joints = p.getNumJoints(hand)
    for joint_number in range(number_of_joints):
        info = p.getJointInfo(hand, joint_number)
        print(f"Join ID: {joint_number}")
        print(info)
        print("-------------")


def manipulate_joint(joint_id):
    """
    Used to manipulate a particular joint

    joint number - description
    --------------------------
    1 - wrist motion (horizontal)
    2 - wrist motion (vertical)
    5 - index finger (horizontal)
    6 - index finger base (vertical)
    7 - index finger middle (vertical)
    8 - index finger tip (vertical)
    10 - middle finger (horizontal)
    11 - middle finger base (vertical)
    12 - middle finger middle (vertical)
    13 - middle finger tip (vertical)
    15 - ring finger (horizontal)
    16 - ring finger base (vertical)
    17 - ring finger middle (vertical)
    18 - ring finger tip (vertical)
    20 - little finger grasp
    21 - little finger (horizontal)
    22 - little finger base (vertical)
    23 - little finger middle (vertical)
    24 - little finger tip (vertical)
    26 - thumb rotation
    27 - thumb finger base (vertical)
    28 - thumb finger middle (vertical)
    29 - thumb finger middle (horizontal)
    30 - thumb finger tip (horizontal)
    """
    param = p.addUserDebugParameter('Input Value', -5, 5, 0)
    while True:
        user_input = p.readUserDebugParameter(param)
        p.setJointMotorControl2(hand, joint_id,
                                p.POSITION_CONTROL,
                                targetPosition=user_input)
        p.stepSimulation()


def run_sim():
    """
    Runs a basic simulation with the loaded urdf files
    """
    while True:
        p.stepSimulation()
        sleep(1 / 60)


def take_picture(renderer, view_matrix, width=256, height=256):
    # view_matrix = p.computeViewMatrix(
    #     [0.1, -0.2, 1.75], [0.1, -0.2, 0], [0, -1, 0]
    # )
    proj_matrix = p.computeProjectionMatrixFOV(
        20, 1, 0.05, 2
    )
    w, h, rgba, depth, mask = p.getCameraImage(
        width=width,
        height=height,
        projectionMatrix=proj_matrix,
        viewMatrix=view_matrix,
        renderer=renderer,
    )
    return rgba


def take_picture_above(renderer):
    view_matrix = p.computeViewMatrix(
        [0.1, -0.2, 1.75], [0.1, -0.2, 0], [0, -1, 0]
    )
    return take_picture(renderer, view_matrix)


def take_picture_rhs(renderer):
    view_matrix = p.computeViewMatrix(
        [-1.3, -0.2, 0.25], [0.1, -0.2, 0.25], [0, 0, 1]
    )
    return take_picture(renderer, view_matrix)


def take_picture_lhs(renderer):
    view_matrix = p.computeViewMatrix(
        [1.5, -0.2, 0.25], [0.1, -0.2, 0.25], [0, 0, 1]
    )
    return take_picture(renderer, view_matrix)


def save_obj(filename, verticies, faces):
    with open(filename, "w") as fp:
        for v in verticies:
            fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write("f %d %d %d\n" % (f[0], f[1], f[2]))


renderer = p.ER_TINY_RENDERER
# take_picture(renderer, width=1000, height=1500)
# p.setJointMotorControl2(hand, 11, p.POSITION_CONTROL, targetPosition=1)

take_picture_above(renderer)
take_picture_rhs(renderer)
take_picture_lhs(renderer)

manipulate_joint(30)

