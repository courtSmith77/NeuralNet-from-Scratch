
import numpy as np

class Controls():

    def __init__(self):

        self.file = np.loadtxt('./ds0_Odometry.dat')
        self.time = self.file[:,0]
        self.velocity = self.file[:,1]
        self.angular = self.file[:,2]

class GroundTruth():

    def __init__(self):

        self.file = np.loadtxt('./ds0_Groundtruth.dat')
        self.time = self.file[:,0]
        self.x_pos = self.file[:,1]
        self.y_pos = self.file[:,2]
        self.heading = self.file[:,3]

