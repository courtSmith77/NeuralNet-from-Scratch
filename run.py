# Running Optimal Models
# See sin_optimization.py, aim_optimization.py, and miniBatching.py for the parameter sweeps

import numpy as np
import matplotlib.pyplot as plt
from network import Network
from fullConnectedLayer import FullConnected
from activationLayer import Activation
from functions import tanh, tanh_derivative, sigmoid, sigmoid_derivative, mse, mse_derivative, r_sq_stats

############## Optimal Sine ###############
# making sin data for training
x = np.linspace(0,10,50) % 2*np.pi
y = np.sin(x)
y_noisy = np.sin(x) + (np.random.rand(50)/5 - 0.1) # add noise between 0.1 and -0.1

fcLayer = FullConnected(1,50)
actLayer = Activation(tanh, tanh_derivative)
fcLayer2 = FullConnected(50,50)
actLayer2 = Activation(tanh, tanh_derivative)
fcLayer3 = FullConnected(50,50)
actLayer3 = Activation(tanh, tanh_derivative)
fcLayer4 = FullConnected(50,50)
actLayer4 = Activation(tanh, tanh_derivative)
fcLayer5 = FullConnected(50,1)
actLayer5 = Activation(tanh, tanh_derivative)
layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5]

model_noisy = Network(layers, mse, mse_derivative)
model_noisy.training(x, y_noisy, 100, 0.005)
output = model_noisy.predict(x)
unwrapped_noisy = []
for ii in range(len(output)):
    unwrapped_noisy.append(output[ii][0][0])

fig1 = plt.figure('Noisy Sine Wave - Optimal Model')
ax1 = fig1.add_subplot()
ax1.scatter(x, unwrapped_noisy)
ax1.scatter(x, y_noisy)
ax1.scatter(x,y,c='red', linewidths=0.01)
ax1.legend(['Predicted', 'Actual - noisy', 'Actual'])
ax1.set_title('Optimal Model')
ax1.set_xlabel('Radians')
ax1.set_ylabel('Output')

print('Sine Wave Stats')
r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
print('R_sq')
print(r_sq)
MSE_val = mse(y_noisy, unwrapped_noisy)
print('MSE')
print(MSE_val)

fig2 = plt.figure('R_sq Comparison - Optimal Model')
ax2 = fig2.add_subplot()
ax2.scatter(y_noisy, unwrapped_noisy)
ax2.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
ax2.legend(['Data Comparison', '100% Correlation'])
ax2.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
ax2.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
ax2.set_title('Optimal Model')
ax2.set_xlabel('True Value')
ax2.set_ylabel('Predicted Value')

######### Optimal Learning Aim #############
# loading robot data
all_data = np.loadtxt('learning_dataset.csv')
print(np.shape(all_data))

x = all_data[:,:6]
y = all_data[:,6:]

wrap_x = []
wrap_y = []
for ii in range(len(x)):
    xx = x[ii]
    yy = y[ii]
    wrap_x.append([xx])
    wrap_y.append([yy])

x_data = np.array(wrap_x)
y_data = np.array(wrap_y)

fcLayer = FullConnected(6,25)
actLayer = Activation(tanh, tanh_derivative)
fcLayer2 = FullConnected(25,25)
actLayer2 = Activation(tanh, tanh_derivative)
fcLayer3 = FullConnected(25,25)
actLayer3 = Activation(tanh, tanh_derivative)
fcLayer4 = FullConnected(25,25)
actLayer4 = Activation(tanh, tanh_derivative)
fcLayer5 = FullConnected(25,25)
actLayer5 = Activation(tanh, tanh_derivative)
fcLayer6 = FullConnected(25,3)
layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5, fcLayer6]

model = Network(layers, mse, mse_derivative)
model.training(x_data, y_data, 100, 0.001)
output = model.predict(x_data)

unwrapped = []
for ii in range(len(output)):
    unwrapped.append(output[ii][0][:])
unwrapped = np.array(unwrapped)

fig3 = plt.figure('Robot Data - Optimal Model')
ax3 = fig3.add_subplot()
ax3.plot(unwrapped[:,0], unwrapped[:,1])
ax3.plot(y[:,0], y[:,1])
ax3.legend(['Predicted', 'Actual'])
ax3.set_title('Testing Robot - Optimal Model')
ax3.set_xlabel('X Position')
ax3.set_ylabel('Y Position')
for hh in range(len(unwrapped[:,2])):
    if (hh % 100) == 0:
        dx = 0.1*np.cos(unwrapped[hh, 2])
        dy = 0.1*np.sin(unwrapped[hh, 2])
        ax3.arrow(unwrapped[hh, 0],unwrapped[hh, 1],dx,dy, length_includes_head=True)

for hh in range(len(y[:,2])):
    if (hh % 100) == 0:
        dx = 0.1*np.cos(y[hh, 2])
        dy = 0.1*np.sin(y[hh, 2])
        ax3.arrow(y[hh, 0],y[hh, 1],dx,dy, length_includes_head=True)

print('Learning Aim Stats')
r_sq_x, std_err = r_sq_stats(y[:,0], unwrapped[:,0])
print(f'R_sq X = {r_sq_x}')
r_sq_y, std_err = r_sq_stats(y[:,1], unwrapped[:,1])
print(f'R_sq Y = {r_sq_y}')
r_sq_h, std_err = r_sq_stats(y[:,2], unwrapped[:,2])
print(f'R_sq Heading = {r_sq_h}')

MSE_val_x = mse(y[:,0], unwrapped[:,0])
print(f'MSE_X = {MSE_val_x}')
MSE_val_y = mse(y[:,1], unwrapped[:,1])
print(f'MSE_Y = {MSE_val_y}')
MSE_val_h = mse(y[:,2], unwrapped[:,2])
print(f'MSE_Heading = {MSE_val_h}')

fig4, (ax4, ax5, ax6) = plt.subplots(1, 3)
fig4.suptitle('R_sq Comparison - Optimal Model')
ax4.scatter(y[:,0], unwrapped[:,0], s=1)
ax4.plot([min(y[:,0]), max(y[:,0])], [min(y[:,0]), max(y[:,0])], c='red')
ax4.legend(['Data Comparison', '100% Correlation'])
ax4.text(1.0,4.0,f'R^2 = {round(r_sq_x,3)}')
ax4.text(1.0,3.9,f'MSE = {round(MSE_val_x,3)}')
ax4.set_title('X Position')
ax4.set_xlabel('True Value')
ax4.set_ylabel('Predicted Value')
ax5.scatter(y[:,1], unwrapped[:,1], s=1)
ax5.plot([min(y[:,1]), max(y[:,1])], [min(y[:,1]), max(y[:,1])], c='red')
ax5.legend(['Data Comparison', '100% Correlation'])
ax5.text(-3.0,2.5,f'R^2 = {round(r_sq_y,3)}')
ax5.text(-3.0,2.3,f'MSE = {round(MSE_val_y,3)}')
ax5.set_title('Y Position')
ax5.set_xlabel('True Value')
ax5.set_ylabel('Predicted Value')
ax6.scatter(y[:,2], unwrapped[:,2], s=1)
ax6.plot([min(y[:,2]), max(y[:,2])], [min(y[:,2]), max(y[:,2])], c='red')
ax6.legend(['Data Comparison', '100% Correlation'])
ax6.text(-3.0,2.5,f'R^2 = {round(r_sq_h,3)}')
ax6.text(-3.0,2.35,f'MSE = {round(MSE_val_h,3)}')
ax6.set_title('Heading')
ax6.set_xlabel('True Value')
ax6.set_ylabel('Predicted Value')


######### Optimal Learning Aim - Mini Batching #############
# loading robot data
all_data = np.loadtxt('learning_dataset.csv')
print(np.shape(all_data))

x = all_data[:,:6]
y = all_data[:,6:]

wrap_x = []
wrap_y = []
for ii in range(len(x)):
    xx = x[ii]
    yy = y[ii]
    wrap_x.append([xx])
    wrap_y.append([yy])

x_data = np.array(wrap_x)
y_data = np.array(wrap_y)

fcLayer = FullConnected(6,25)
actLayer = Activation(tanh, tanh_derivative)
fcLayer2 = FullConnected(25,25)
actLayer2 = Activation(tanh, tanh_derivative)
fcLayer3 = FullConnected(25,25)
actLayer3 = Activation(tanh, tanh_derivative)
fcLayer4 = FullConnected(25,25)
actLayer4 = Activation(tanh, tanh_derivative)
fcLayer5 = FullConnected(25,25)
actLayer5 = Activation(tanh, tanh_derivative)
fcLayer6 = FullConnected(25,3)
layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5, fcLayer6]

model = Network(layers, mse, mse_derivative)
model.training_mini(x_data, y_data, 100, 0.001, 25)
output = model.predict(x_data)

unwrapped = []
for ii in range(len(output)):
    unwrapped.append(output[ii][0][:])
unwrapped = np.array(unwrapped)

fig5 = plt.figure('Robot Data - Optimal Model Mini Batching')
ax7 = fig5.add_subplot()
ax7.plot(unwrapped[:,0], unwrapped[:,1])
ax7.plot(y[:,0], y[:,1])
ax7.legend(['Predicted', 'Actual'])
ax7.set_title('Testing Robot - Optimal Model Mini Batching')
ax7.set_xlabel('X Position')
ax7.set_ylabel('Y Position')
for hh in range(len(unwrapped[:,2])):
    if (hh % 100) == 0:
        dx = 0.1*np.cos(unwrapped[hh, 2])
        dy = 0.1*np.sin(unwrapped[hh, 2])
        ax7.arrow(unwrapped[hh, 0],unwrapped[hh, 1],dx,dy, length_includes_head=True)

for hh in range(len(y[:,2])):
    if (hh % 100) == 0:
        dx = 0.1*np.cos(y[hh, 2])
        dy = 0.1*np.sin(y[hh, 2])
        ax7.arrow(y[hh, 0],y[hh, 1],dx,dy, length_includes_head=True)

print('Mini Batching Stats')
r_sq_x, std_err = r_sq_stats(y[:,0], unwrapped[:,0])
print(f'R_sq X = {r_sq_x}')
r_sq_y, std_err = r_sq_stats(y[:,1], unwrapped[:,1])
print(f'R_sq Y = {r_sq_y}')
r_sq_h, std_err = r_sq_stats(y[:,2], unwrapped[:,2])
print(f'R_sq Heading = {r_sq_h}')

MSE_val_x = mse(y[:,0], unwrapped[:,0])
print(f'MSE_X = {MSE_val_x}')
MSE_val_y = mse(y[:,1], unwrapped[:,1])
print(f'MSE_Y = {MSE_val_y}')
MSE_val_h = mse(y[:,2], unwrapped[:,2])
print(f'MSE_Heading = {MSE_val_h}')

fig6, (ax8, ax9, ax10) = plt.subplots(1, 3)
fig6.suptitle('R_sq Comparison - Optimal Model Mini Batching')
ax8.scatter(y[:,0], unwrapped[:,0], s=1)
ax8.plot([min(y[:,0]), max(y[:,0])], [min(y[:,0]), max(y[:,0])], c='red')
ax8.legend(['Data Comparison', '100% Correlation'])
ax8.text(1.0,4.0,f'R^2 = {round(r_sq_x,3)}')
ax8.text(1.0,3.9,f'MSE = {round(MSE_val_x,3)}')
ax8.set_title('X Position')
ax8.set_xlabel('True Value')
ax8.set_ylabel('Predicted Value')
ax9.scatter(y[:,1], unwrapped[:,1], s=1)
ax9.plot([min(y[:,1]), max(y[:,1])], [min(y[:,1]), max(y[:,1])], c='red')
ax9.legend(['Data Comparison', '100% Correlation'])
ax9.text(-3.0,2.5,f'R^2 = {round(r_sq_y,3)}')
ax9.text(-3.0,2.3,f'MSE = {round(MSE_val_y,3)}')
ax9.set_title('Y Position')
ax9.set_xlabel('True Value')
ax9.set_ylabel('Predicted Value')
ax10.scatter(y[:,2], unwrapped[:,2], s=1)
ax10.plot([min(y[:,2]), max(y[:,2])], [min(y[:,2]), max(y[:,2])], c='red')
ax10.legend(['Data Comparison', '100% Correlation'])
ax10.text(-3.0,2.5,f'R^2 = {round(r_sq_h,3)}')
ax10.text(-3.0,2.35,f'MSE = {round(MSE_val_h,3)}')
ax10.set_title('Heading')
ax10.set_xlabel('True Value')
ax10.set_ylabel('Predicted Value')


plt.show()
