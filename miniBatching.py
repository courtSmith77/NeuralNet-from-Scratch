import numpy as np
import matplotlib.pyplot as plt
from network import Network
from fullConnectedLayer import FullConnected
from activationLayer import Activation
from functions import tanh, tanh_derivative, sigmoid, sigmoid_derivative, mse, mse_derivative, r_sq_stats

# # making sin data for training
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


####### Testing Mini Batching ##########
# # # # # 100 Batch Size # # # # #
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
model.training_mini(x_data, y_data, 100, 0.001, 100)
output = model.predict(x_data)
print(f'Output = {np.shape(output)}')

unwrapped = []
for ii in range(len(output)):
    unwrapped.append(output[ii][0][:])
unwrapped = np.array(unwrapped)

print(f'Unwrapped Output = {np.shape(unwrapped)}')
print(f'Testing IO 25')

fig1 = plt.figure('Robot Data - 100 Batch Size')
ax1 = fig1.add_subplot()
ax1.plot(unwrapped[:,0], unwrapped[:,1])
ax1.plot(y[:,0], y[:,1])
ax1.legend(['Predicted', 'Actual'])
ax1.set_title('Testing Robot - 100 Batch Size')
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
for hh in range(len(unwrapped[:,2])):
    if (hh % 100) == 0:
        dx = 0.1*np.cos(unwrapped[hh, 2])
        dy = 0.1*np.sin(unwrapped[hh, 2])
        ax1.arrow(unwrapped[hh, 0],unwrapped[hh, 1],dx,dy, length_includes_head=True)

for hh in range(len(y[:,2])):
    if (hh % 100) == 0:
        dx = 0.1*np.cos(y[hh, 2])
        dy = 0.1*np.sin(y[hh, 2])
        ax1.arrow(y[hh, 0],y[hh, 1],dx,dy, length_includes_head=True)

r_sq_x, std_err = r_sq_stats(y[:,0], unwrapped[:,0])
print('R_sq X')
print(r_sq_x)
r_sq_y, std_err = r_sq_stats(y[:,1], unwrapped[:,1])
print('R_sq Y')
print(r_sq_y)
r_sq_h, std_err = r_sq_stats(y[:,2], unwrapped[:,2])
print('R_sq Heading')
print(r_sq_h)

MSE_val_x = mse(y[:,0], unwrapped[:,0])
print('MSE')
print(MSE_val_x)
MSE_val_y = mse(y[:,1], unwrapped[:,1])
print('MSE')
print(MSE_val_y)
MSE_val_h = mse(y[:,2], unwrapped[:,2])
print('MSE')
print(MSE_val_h)

fig2, (ax2, ax3, ax4) = plt.subplots(1, 3)
fig2.suptitle('R_sq Comparison - 100 Batch Size')
ax2.scatter(y[:,0], unwrapped[:,0], s=1)
ax2.plot([min(y[:,0]), max(y[:,0])], [min(y[:,0]), max(y[:,0])], c='red')
ax2.legend(['Data Comparison', '100% Correlation'])
ax2.text(1.0,4.0,f'R^2 = {round(r_sq_x,3)}')
ax2.text(1.0,3.9,f'MSE = {round(MSE_val_x,3)}')
ax2.set_title('X Position')
ax2.set_xlabel('True Value')
ax2.set_ylabel('Predicted Value')
ax3.scatter(y[:,1], unwrapped[:,1], s=1)
ax3.plot([min(y[:,1]), max(y[:,1])], [min(y[:,1]), max(y[:,1])], c='red')
ax3.legend(['Data Comparison', '100% Correlation'])
ax3.text(-3.0,2.5,f'R^2 = {round(r_sq_y,3)}')
ax3.text(-3.0,2.3,f'MSE = {round(MSE_val_y,3)}')
ax3.set_title('Y Position')
ax3.set_xlabel('True Value')
ax3.set_ylabel('Predicted Value')
ax4.scatter(y[:,2], unwrapped[:,2], s=1)
ax4.plot([min(y[:,2]), max(y[:,2])], [min(y[:,2]), max(y[:,2])], c='red')
ax4.legend(['Data Comparison', '100% Correlation'])
ax4.text(-3.0,2.5,f'R^2 = {round(r_sq_h,3)}')
ax4.text(-3.0,2.35,f'MSE = {round(MSE_val_h,3)}')
ax4.set_title('Heading')
ax4.set_xlabel('True Value')
ax4.set_ylabel('Predicted Value')

# # # # # 75 Batch Size # # # # #
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
model.training_mini(x_data, y_data, 100, 0.001, 75)
output = model.predict(x_data)
print(f'Output = {np.shape(output)}')

unwrapped = []
for ii in range(len(output)):
    unwrapped.append(output[ii][0][:])
unwrapped = np.array(unwrapped)

print(f'Unwrapped Output = {np.shape(unwrapped)}')
print(f'Testing IO 25')

fig3 = plt.figure('Robot Data - 75 Batch Size')
ax5 = fig3.add_subplot()
ax5.plot(unwrapped[:,0], unwrapped[:,1])
ax5.plot(y[:,0], y[:,1])
ax5.legend(['Predicted', 'Actual'])
ax5.set_title('Testing Robot - 75 Batch Size')
ax5.set_xlabel('X Position')
ax5.set_ylabel('Y Position')
for hh in range(len(unwrapped[:,2])):
    if (hh % 100) == 0:
        dx = 0.1*np.cos(unwrapped[hh, 2])
        dy = 0.1*np.sin(unwrapped[hh, 2])
        ax5.arrow(unwrapped[hh, 0],unwrapped[hh, 1],dx,dy, length_includes_head=True)

for hh in range(len(y[:,2])):
    if (hh % 100) == 0:
        dx = 0.1*np.cos(y[hh, 2])
        dy = 0.1*np.sin(y[hh, 2])
        ax5.arrow(y[hh, 0],y[hh, 1],dx,dy, length_includes_head=True)

r_sq_x, std_err = r_sq_stats(y[:,0], unwrapped[:,0])
print('R_sq X')
print(r_sq_x)
r_sq_y, std_err = r_sq_stats(y[:,1], unwrapped[:,1])
print('R_sq Y')
print(r_sq_y)
r_sq_h, std_err = r_sq_stats(y[:,2], unwrapped[:,2])
print('R_sq Heading')
print(r_sq_h)

MSE_val_x = mse(y[:,0], unwrapped[:,0])
print('MSE')
print(MSE_val_x)
MSE_val_y = mse(y[:,1], unwrapped[:,1])
print('MSE')
print(MSE_val_y)
MSE_val_h = mse(y[:,2], unwrapped[:,2])
print('MSE')
print(MSE_val_h)

fig4, (ax6, ax7, ax8) = plt.subplots(1, 3)
fig4.suptitle('R_sq Comparison - 75 Batch Size')
ax6.scatter(y[:,0], unwrapped[:,0], s=1)
ax6.plot([min(y[:,0]), max(y[:,0])], [min(y[:,0]), max(y[:,0])], c='red')
ax6.legend(['Data Comparison', '100% Correlation'])
ax6.text(1.0,4.0,f'R^2 = {round(r_sq_x,3)}')
ax6.text(1.0,3.9,f'MSE = {round(MSE_val_x,3)}')
ax6.set_title('X Position')
ax6.set_xlabel('True Value')
ax6.set_ylabel('Predicted Value')
ax7.scatter(y[:,1], unwrapped[:,1], s=1)
ax7.plot([min(y[:,1]), max(y[:,1])], [min(y[:,1]), max(y[:,1])], c='red')
ax7.legend(['Data Comparison', '100% Correlation'])
ax7.text(-3.0,2.5,f'R^2 = {round(r_sq_y,3)}')
ax7.text(-3.0,2.3,f'MSE = {round(MSE_val_y,3)}')
ax7.set_title('Y Position')
ax7.set_xlabel('True Value')
ax7.set_ylabel('Predicted Value')
ax8.scatter(y[:,2], unwrapped[:,2], s=1)
ax8.plot([min(y[:,2]), max(y[:,2])], [min(y[:,2]), max(y[:,2])], c='red')
ax8.legend(['Data Comparison', '100% Correlation'])
ax8.text(-3.0,2.5,f'R^2 = {round(r_sq_h,3)}')
ax8.text(-3.0,2.35,f'MSE = {round(MSE_val_h,3)}')
ax8.set_title('Heading')
ax8.set_xlabel('True Value')
ax8.set_ylabel('Predicted Value')

# # # # # 50 Batch Size # # # # #
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
model.training_mini(x_data, y_data, 100, 0.001, 50)
output = model.predict(x_data)
print(f'Output = {np.shape(output)}')

unwrapped = []
for ii in range(len(output)):
    unwrapped.append(output[ii][0][:])
unwrapped = np.array(unwrapped)

print(f'Unwrapped Output = {np.shape(unwrapped)}')
print(f'Testing IO 25')

fig5 = plt.figure('Robot Data - 50 Batch Size')
ax9 = fig5.add_subplot()
ax9.plot(unwrapped[:,0], unwrapped[:,1])
ax9.plot(y[:,0], y[:,1])
ax9.legend(['Predicted', 'Actual'])
ax9.set_title('Testing Robot - 50 Batch Size')
ax9.set_xlabel('X Position')
ax9.set_ylabel('Y Position')
for hh in range(len(unwrapped[:,2])):
    if (hh % 100) == 0:
        dx = 0.1*np.cos(unwrapped[hh, 2])
        dy = 0.1*np.sin(unwrapped[hh, 2])
        ax9.arrow(unwrapped[hh, 0],unwrapped[hh, 1],dx,dy, length_includes_head=True)

for hh in range(len(y[:,2])):
    if (hh % 100) == 0:
        dx = 0.1*np.cos(y[hh, 2])
        dy = 0.1*np.sin(y[hh, 2])
        ax9.arrow(y[hh, 0],y[hh, 1],dx,dy, length_includes_head=True)

r_sq_x, std_err = r_sq_stats(y[:,0], unwrapped[:,0])
print('R_sq X')
print(r_sq_x)
r_sq_y, std_err = r_sq_stats(y[:,1], unwrapped[:,1])
print('R_sq Y')
print(r_sq_y)
r_sq_h, std_err = r_sq_stats(y[:,2], unwrapped[:,2])
print('R_sq Heading')
print(r_sq_h)

MSE_val_x = mse(y[:,0], unwrapped[:,0])
print('MSE')
print(MSE_val_x)
MSE_val_y = mse(y[:,1], unwrapped[:,1])
print('MSE')
print(MSE_val_y)
MSE_val_h = mse(y[:,2], unwrapped[:,2])
print('MSE')
print(MSE_val_h)

fig6, (ax10, ax11, ax12) = plt.subplots(1, 3)
fig6.suptitle('R_sq Comparison - 50 Batch Size')
ax10.scatter(y[:,0], unwrapped[:,0], s=1)
ax10.plot([min(y[:,0]), max(y[:,0])], [min(y[:,0]), max(y[:,0])], c='red')
ax10.legend(['Data Comparison', '100% Correlation'])
ax10.text(1.0,4.0,f'R^2 = {round(r_sq_x,3)}')
ax10.text(1.0,3.9,f'MSE = {round(MSE_val_x,3)}')
ax10.set_title('X Position')
ax10.set_xlabel('True Value')
ax10.set_ylabel('Predicted Value')
ax11.scatter(y[:,1], unwrapped[:,1], s=1)
ax11.plot([min(y[:,1]), max(y[:,1])], [min(y[:,1]), max(y[:,1])], c='red')
ax11.legend(['Data Comparison', '100% Correlation'])
ax11.text(-3.0,2.5,f'R^2 = {round(r_sq_y,3)}')
ax11.text(-3.0,2.3,f'MSE = {round(MSE_val_y,3)}')
ax11.set_title('Y Position')
ax11.set_xlabel('True Value')
ax11.set_ylabel('Predicted Value')
ax12.scatter(y[:,2], unwrapped[:,2], s=1)
ax12.plot([min(y[:,2]), max(y[:,2])], [min(y[:,2]), max(y[:,2])], c='red')
ax12.legend(['Data Comparison', '100% Correlation'])
ax12.text(-3.0,2.5,f'R^2 = {round(r_sq_h,3)}')
ax12.text(-3.0,2.35,f'MSE = {round(MSE_val_h,3)}')
ax12.set_title('Heading')
ax12.set_xlabel('True Value')
ax12.set_ylabel('Predicted Value')


# # # # # 25 Batch Size # # # # #
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
print(f'Output = {np.shape(output)}')

unwrapped = []
for ii in range(len(output)):
    unwrapped.append(output[ii][0][:])
unwrapped = np.array(unwrapped)

print(f'Unwrapped Output = {np.shape(unwrapped)}')
print(f'Testing IO 25')

fig7 = plt.figure('Robot Data - 25 Batch Size')
ax13 = fig7.add_subplot()
ax13.plot(unwrapped[:,0], unwrapped[:,1])
ax13.plot(y[:,0], y[:,1])
ax13.legend(['Predicted', 'Actual'])
ax13.set_title('Testing Robot - 25 Batch Size')
ax13.set_xlabel('X Position')
ax13.set_ylabel('Y Position')
for hh in range(len(unwrapped[:,2])):
    if (hh % 100) == 0:
        dx = 0.1*np.cos(unwrapped[hh, 2])
        dy = 0.1*np.sin(unwrapped[hh, 2])
        ax13.arrow(unwrapped[hh, 0],unwrapped[hh, 1],dx,dy, length_includes_head=True)

for hh in range(len(y[:,2])):
    if (hh % 100) == 0:
        dx = 0.1*np.cos(y[hh, 2])
        dy = 0.1*np.sin(y[hh, 2])
        ax13.arrow(y[hh, 0],y[hh, 1],dx,dy, length_includes_head=True)

r_sq_x, std_err = r_sq_stats(y[:,0], unwrapped[:,0])
print('R_sq X')
print(r_sq_x)
r_sq_y, std_err = r_sq_stats(y[:,1], unwrapped[:,1])
print('R_sq Y')
print(r_sq_y)
r_sq_h, std_err = r_sq_stats(y[:,2], unwrapped[:,2])
print('R_sq Heading')
print(r_sq_h)

MSE_val_x = mse(y[:,0], unwrapped[:,0])
print('MSE')
print(MSE_val_x)
MSE_val_y = mse(y[:,1], unwrapped[:,1])
print('MSE')
print(MSE_val_y)
MSE_val_h = mse(y[:,2], unwrapped[:,2])
print('MSE')
print(MSE_val_h)

fig8, (ax14, ax15, ax16) = plt.subplots(1, 3)
fig8.suptitle('R_sq Comparison - 25 Batch Size')
ax14.scatter(y[:,0], unwrapped[:,0], s=1)
ax14.plot([min(y[:,0]), max(y[:,0])], [min(y[:,0]), max(y[:,0])], c='red')
ax14.legend(['Data Comparison', '100% Correlation'])
ax14.text(1.0,4.0,f'R^2 = {round(r_sq_x,3)}')
ax14.text(1.0,3.9,f'MSE = {round(MSE_val_x,3)}')
ax14.set_title('X Position')
ax14.set_xlabel('True Value')
ax14.set_ylabel('Predicted Value')
ax15.scatter(y[:,1], unwrapped[:,1], s=1)
ax15.plot([min(y[:,1]), max(y[:,1])], [min(y[:,1]), max(y[:,1])], c='red')
ax15.legend(['Data Comparison', '100% Correlation'])
ax15.text(-3.0,2.5,f'R^2 = {round(r_sq_y,3)}')
ax15.text(-3.0,2.3,f'MSE = {round(MSE_val_y,3)}')
ax15.set_title('Y Position')
ax15.set_xlabel('True Value')
ax15.set_ylabel('Predicted Value')
ax16.scatter(y[:,2], unwrapped[:,2], s=1)
ax16.plot([min(y[:,2]), max(y[:,2])], [min(y[:,2]), max(y[:,2])], c='red')
ax16.legend(['Data Comparison', '100% Correlation'])
ax16.text(-3.0,2.5,f'R^2 = {round(r_sq_h,3)}')
ax16.text(-3.0,2.35,f'MSE = {round(MSE_val_h,3)}')
ax16.set_title('Heading')
ax16.set_xlabel('True Value')
ax16.set_ylabel('Predicted Value')


plt.show()