# creating data labels for all data

from loadingData import Controls, GroundTruth
import numpy as np

class TrainingData():

    def __init__(self):

        self.position_data = GroundTruth()
        self.controls_data = Controls()

    def labelingData(self):

        allData = []
        for pp in range(len(self.position_data.time)-1):

            p_time = self.position_data.time[pp]
            close_diff = 1

            for cc in range(len(self.controls_data.time)):

                c_time = self.controls_data.time[cc]
                diff_time = c_time - p_time

                if diff_time > 0 and abs(diff_time) < close_diff :
                    close_diff = diff_time
                    closest_time = c_time
                    closest_data = [self.controls_data.velocity[cc], self.controls_data.angular[cc]]
                    
                elif diff_time < 0 :
                    continue

            dt = self.position_data.time[pp+1] - p_time
            
            t_0 = [self.position_data.x_pos[pp], self.position_data.y_pos[pp], self.position_data.heading[pp]]
            t_1 = [self.position_data.x_pos[pp+1], self.position_data.y_pos[pp+1], self.position_data.heading[pp+1]]
            
            timeStampData = [dt, t_0[0], t_0[1], t_0[2], closest_data[0], closest_data[1], t_1[0], t_1[1], t_1[2]]

            allData.append(timeStampData)

            if (pp % 100) == 0 :
                print(pp)

        np.savetxt('learning_dataset1.csv', allData)
        return allData


data = TrainingData()
labeledData = data.labelingData()

            


