import numpy as np
import matplotlib.pyplot as plt
from network import Network
from fullConnectedLayer import FullConnected
from activationLayer import Activation
from functions import tanh, tanh_derivative, sigmoid, sigmoid_derivative, mse, mse_derivative, r_sq_stats

# making sin data for training
x = np.linspace(0,10,50) % 2*np.pi
y = np.sin(x)
y_noisy = np.sin(x) + (np.random.rand(50)/5 - 0.1) # add noise between 0.1 and -0.1

# ############## 1 FC ###############

# fcLayer = FullConnected(1,20)
# actLayer = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer]

# # # # # # Testing on noisy sine wave
# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 100, 0.01)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig1 = plt.figure('Noisy Sine Wave - 1 FC')
# ax1 = fig1.add_subplot()
# ax1.scatter(x, unwrapped_noisy)
# ax1.scatter(x, y_noisy)
# ax1.scatter(x,y,c='red', linewidths=0.01)
# ax1.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax1.set_title('Fully Connected Layers = 1')
# #ax1.set_title('Testing model on a noisy Sine Wave')
# ax1.set_xlabel('Radians')
# ax1.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig2 = plt.figure('R_sq Comparison - 1 FC')
# ax2 = fig2.add_subplot()
# ax2.scatter(y_noisy, unwrapped_noisy)
# ax2.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax2.legend(['Data Comparison', '100% Correlation'])
# ax2.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax2.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax2.set_title('Fully Connected Layers = 1')
# #ax2.set_title('Actual vs Predicted')
# ax2.set_xlabel('True Value')
# ax2.set_ylabel('Predicted Value')

# ############## 2 FC ###############

# fcLayer = FullConnected(1,20)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(20,1)
# actLayer2 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2]

# # # # # # Testing on noisy sine wave
# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 100, 0.01)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig3 = plt.figure('Noisy Sine Wave - 2 FC')
# ax3 = fig3.add_subplot()
# ax3.scatter(x, unwrapped_noisy)
# ax3.scatter(x, y_noisy)
# ax3.scatter(x,y,c='red', linewidths=0.01)
# ax3.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax3.set_title('Fully Connected Layers = 2')
# ax3.set_xlabel('Radians')
# ax3.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig4 = plt.figure('R_sq Comparison - 2 FC')
# ax4 = fig4.add_subplot()
# ax4.scatter(y_noisy, unwrapped_noisy)
# ax4.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax4.legend(['Data Comparison', '100% Correlation'])
# ax4.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax4.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax4.set_title('Fully Connected Layers = 2')
# ax4.set_xlabel('True Value')
# ax4.set_ylabel('Predicted Value')

# ############## 3 FC ###############

# fcLayer = FullConnected(1,20)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(20,20)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(20,1)
# actLayer3 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3]

# # # # # # Testing on noisy sine wave
# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 100, 0.01)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig5 = plt.figure('Noisy Sine Wave - 3 FC')
# ax5 = fig5.add_subplot()
# ax5.scatter(x, unwrapped_noisy)
# ax5.scatter(x, y_noisy)
# ax5.scatter(x,y,c='red', linewidths=0.01)
# ax5.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax5.set_title('Fully Connected Layers = 3')
# ax5.set_xlabel('Radians')
# ax5.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig6 = plt.figure('R_sq Comparison - 3 FC')
# ax6 = fig6.add_subplot()
# ax6.scatter(y_noisy, unwrapped_noisy)
# ax6.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax6.legend(['Data Comparison', '100% Correlation'])
# ax6.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax6.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax6.set_title('Fully Connected Layers = 3')
# ax6.set_xlabel('True Value')
# ax6.set_ylabel('Predicted Value')

# ############## 4 FC ###############

# fcLayer = FullConnected(1,20)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(20,20)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(20,20)
# actLayer3 = Activation(tanh, tanh_derivative)
# fcLayer4 = FullConnected(20,1)
# actLayer4 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4]

# # # # # # Testing on noisy sine wave
# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 100, 0.01)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig7 = plt.figure('Noisy Sine Wave - 4 FC')
# ax7 = fig7.add_subplot()
# ax7.scatter(x, unwrapped_noisy)
# ax7.scatter(x, y_noisy)
# ax7.scatter(x,y,c='red', linewidths=0.01)
# ax7.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax7.set_title('Fully Connected Layers = 4')
# ax7.set_xlabel('Radians')
# ax7.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig8 = plt.figure('R_sq Comparison - 4 FC')
# ax8 = fig8.add_subplot()
# ax8.scatter(y_noisy, unwrapped_noisy)
# ax8.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax8.legend(['Data Comparison', '100% Correlation'])
# ax8.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax8.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax8.set_title('Fully Connected Layers = 4')
# ax8.set_xlabel('True Value')
# ax8.set_ylabel('Predicted Value')

# ############## 5 FC ###############

# fcLayer = FullConnected(1,20)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(20,20)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(20,20)
# actLayer3 = Activation(tanh, tanh_derivative)
# fcLayer4 = FullConnected(20,20)
# actLayer4 = Activation(tanh, tanh_derivative)
# fcLayer5 = FullConnected(20,1)
# actLayer5 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5]

# # # # # # Testing on noisy sine wave
# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 100, 0.01)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig9 = plt.figure('Noisy Sine Wave - 5 FC')
# ax9 = fig9.add_subplot()
# ax9.scatter(x, unwrapped_noisy)
# ax9.scatter(x, y_noisy)
# ax9.scatter(x,y,c='red', linewidths=0.01)
# ax9.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax9.set_title('Fully Connected Layers = 5')
# ax9.set_xlabel('Radians')
# ax9.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig10 = plt.figure('R_sq Comparison - 5 FC')
# ax10 = fig10.add_subplot()
# ax10.scatter(y_noisy, unwrapped_noisy)
# ax10.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax10.legend(['Data Comparison', '100% Correlation'])
# ax10.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax10.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax10.set_title('Fully Connected Layers = 5')
# ax10.set_xlabel('True Value')
# ax10.set_ylabel('Predicted Value')

# ############## 10 io ###############

# fcLayer = FullConnected(1,10)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(10,10)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(10,10)
# actLayer3 = Activation(tanh, tanh_derivative)
# fcLayer4 = FullConnected(10,10)
# actLayer4 = Activation(tanh, tanh_derivative)
# fcLayer5 = FullConnected(10,1)
# actLayer5 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5]

# # # # # # Testing on noisy sine wave
# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 100, 0.01)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig1 = plt.figure('Noisy Sine Wave - 10 IO')
# ax1 = fig1.add_subplot()
# ax1.scatter(x, unwrapped_noisy)
# ax1.scatter(x, y_noisy)
# ax1.scatter(x,y,c='red', linewidths=0.01)
# ax1.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax1.set_title('Input/Output = 10')
# #ax1.set_title('Testing model on a noisy Sine Wave')
# ax1.set_xlabel('Radians')
# ax1.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig2 = plt.figure('R_sq Comparison - 10 IO')
# ax2 = fig2.add_subplot()
# ax2.scatter(y_noisy, unwrapped_noisy)
# ax2.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax2.legend(['Data Comparison', '100% Correlation'])
# ax2.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax2.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax2.set_title('Input/Output = 10')
# #ax2.set_title('Actual vs Predicted')
# ax2.set_xlabel('True Value')
# ax2.set_ylabel('Predicted Value')

# ############## 20 io ###############

# fcLayer = FullConnected(1,20)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(20,20)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(20,20)
# actLayer3 = Activation(tanh, tanh_derivative)
# fcLayer4 = FullConnected(20,20)
# actLayer4 = Activation(tanh, tanh_derivative)
# fcLayer5 = FullConnected(20,1)
# actLayer5 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5]

# # # # # # Testing on noisy sine wave
# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 100, 0.01)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig1 = plt.figure('Noisy Sine Wave - 20 IO')
# ax1 = fig1.add_subplot()
# ax1.scatter(x, unwrapped_noisy)
# ax1.scatter(x, y_noisy)
# ax1.scatter(x,y,c='red', linewidths=0.01)
# ax1.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax1.set_title('Input/Output = 20')
# #ax1.set_title('Testing model on a noisy Sine Wave')
# ax1.set_xlabel('Radians')
# ax1.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig2 = plt.figure('R_sq Comparison - 20 IO')
# ax2 = fig2.add_subplot()
# ax2.scatter(y_noisy, unwrapped_noisy)
# ax2.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax2.legend(['Data Comparison', '100% Correlation'])
# ax2.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax2.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax2.set_title('Input/Output = 20')
# #ax2.set_title('Actual vs Predicted')
# ax2.set_xlabel('True Value')
# ax2.set_ylabel('Predicted Value')

# ############## 30 IO ###############

# fcLayer = FullConnected(1,30)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(30,30)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(30,30)
# actLayer3 = Activation(tanh, tanh_derivative)
# fcLayer4 = FullConnected(30,30)
# actLayer4 = Activation(tanh, tanh_derivative)
# fcLayer5 = FullConnected(30,1)
# actLayer5 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5]

# # # # # # Testing on noisy sine wave
# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 100, 0.01)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig5 = plt.figure('Noisy Sine Wave - 30 IO')
# ax5 = fig5.add_subplot()
# ax5.scatter(x, unwrapped_noisy)
# ax5.scatter(x, y_noisy)
# ax5.scatter(x,y,c='red', linewidths=0.01)
# ax5.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax5.set_title('Input/Output = 30')
# ax5.set_xlabel('Radians')
# ax5.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig6 = plt.figure('R_sq Comparison - 30 IO')
# ax6 = fig6.add_subplot()
# ax6.scatter(y_noisy, unwrapped_noisy)
# ax6.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax6.legend(['Data Comparison', '100% Correlation'])
# ax6.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax6.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax6.set_title('Input/Output = 30')
# ax6.set_xlabel('True Value')
# ax6.set_ylabel('Predicted Value')

# ############## 40 IO ###############

# fcLayer = FullConnected(1,40)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(40,40)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(40,40)
# actLayer3 = Activation(tanh, tanh_derivative)
# fcLayer4 = FullConnected(40,40)
# actLayer4 = Activation(tanh, tanh_derivative)
# fcLayer5 = FullConnected(40,1)
# actLayer5 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5]

# # # # # # Testing on noisy sine wave
# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 100, 0.01)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig7 = plt.figure('Noisy Sine Wave - 40 IO')
# ax7 = fig7.add_subplot()
# ax7.scatter(x, unwrapped_noisy)
# ax7.scatter(x, y_noisy)
# ax7.scatter(x,y,c='red', linewidths=0.01)
# ax7.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax7.set_title('Input/Output = 40')
# ax7.set_xlabel('Radians')
# ax7.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig8 = plt.figure('R_sq Comparison - 40 IO')
# ax8 = fig8.add_subplot()
# ax8.scatter(y_noisy, unwrapped_noisy)
# ax8.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax8.legend(['Data Comparison', '100% Correlation'])
# ax8.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax8.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax8.set_title('Input/Output = 40')
# ax8.set_xlabel('True Value')
# ax8.set_ylabel('Predicted Value')

# ############## 50 IO ###############

# fcLayer = FullConnected(1,50)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(50,50)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(50,50)
# actLayer3 = Activation(tanh, tanh_derivative)
# fcLayer4 = FullConnected(50,50)
# actLayer4 = Activation(tanh, tanh_derivative)
# fcLayer5 = FullConnected(50,1)
# actLayer5 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5]

# # # # # # Testing on noisy sine wave
# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 100, 0.01)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig9 = plt.figure('Noisy Sine Wave - 50 IO')
# ax9 = fig9.add_subplot()
# ax9.scatter(x, unwrapped_noisy)
# ax9.scatter(x, y_noisy)
# ax9.scatter(x,y,c='red', linewidths=0.01)
# ax9.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax9.set_title('Input/Output = 50')
# ax9.set_xlabel('Radians')
# ax9.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig10 = plt.figure('R_sq Comparison - 50 IO')
# ax10 = fig10.add_subplot()
# ax10.scatter(y_noisy, unwrapped_noisy)
# ax10.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax10.legend(['Data Comparison', '100% Correlation'])
# ax10.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax10.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax10.set_title('Input/Output = 50')
# ax10.set_xlabel('True Value')
# ax10.set_ylabel('Predicted Value')

# ############## LR 0.01 ###############

# fcLayer = FullConnected(1,50)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(50,50)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(50,50)
# actLayer3 = Activation(tanh, tanh_derivative)
# fcLayer4 = FullConnected(50,50)
# actLayer4 = Activation(tanh, tanh_derivative)
# fcLayer5 = FullConnected(50,1)
# actLayer5 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5]

# # # # # # Testing on noisy sine wave
# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 100, 0.01)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig9 = plt.figure('Noisy Sine Wave - 0.01 LR')
# ax9 = fig9.add_subplot()
# ax9.scatter(x, unwrapped_noisy)
# ax9.scatter(x, y_noisy)
# ax9.scatter(x,y,c='red', linewidths=0.01)
# ax9.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax9.set_title('Learning Rate = 0.01')
# ax9.set_xlabel('Radians')
# ax9.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig10 = plt.figure('R_sq Comparison - 0.01 LR')
# ax10 = fig10.add_subplot()
# ax10.scatter(y_noisy, unwrapped_noisy)
# ax10.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax10.legend(['Data Comparison', '100% Correlation'])
# ax10.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax10.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax10.set_title('Learning Rate = 0.01')
# ax10.set_xlabel('True Value')
# ax10.set_ylabel('Predicted Value')

# ############## LR 0.005 ###############

# fcLayer = FullConnected(1,50)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(50,50)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(50,50)
# actLayer3 = Activation(tanh, tanh_derivative)
# fcLayer4 = FullConnected(50,50)
# actLayer4 = Activation(tanh, tanh_derivative)
# fcLayer5 = FullConnected(50,1)
# actLayer5 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5]

# # # # # # Testing on noisy sine wave
# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 100, 0.005)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig9 = plt.figure('Noisy Sine Wave - 0.005 LR')
# ax9 = fig9.add_subplot()
# ax9.scatter(x, unwrapped_noisy)
# ax9.scatter(x, y_noisy)
# ax9.scatter(x,y,c='red', linewidths=0.01)
# ax9.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax9.set_title('Learning Rate = 0.005')
# ax9.set_xlabel('Radians')
# ax9.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig10 = plt.figure('R_sq Comparison - 0.005 LR')
# ax10 = fig10.add_subplot()
# ax10.scatter(y_noisy, unwrapped_noisy)
# ax10.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax10.legend(['Data Comparison', '100% Correlation'])
# ax10.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax10.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax10.set_title('Learning Rate = 0.005')
# ax10.set_xlabel('True Value')
# ax10.set_ylabel('Predicted Value')

# ############## LR 0.05 ###############

# fcLayer = FullConnected(1,50)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(50,50)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(50,50)
# actLayer3 = Activation(tanh, tanh_derivative)
# fcLayer4 = FullConnected(50,50)
# actLayer4 = Activation(tanh, tanh_derivative)
# fcLayer5 = FullConnected(50,1)
# actLayer5 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5]

# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 100, 0.05)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig9 = plt.figure('Noisy Sine Wave - 0.05 LR')
# ax9 = fig9.add_subplot()
# ax9.scatter(x, unwrapped_noisy)
# ax9.scatter(x, y_noisy)
# ax9.scatter(x,y,c='red', linewidths=0.01)
# ax9.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax9.set_title('Learning Rate = 0.05')
# ax9.set_xlabel('Radians')
# ax9.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig10 = plt.figure('R_sq Comparison - 0.05 LR')
# ax10 = fig10.add_subplot()
# ax10.scatter(y_noisy, unwrapped_noisy)
# ax10.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax10.legend(['Data Comparison', '100% Correlation'])
# ax10.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax10.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax10.set_title('Learning Rate = 0.05')
# ax10.set_xlabel('True Value')
# ax10.set_ylabel('Predicted Value')

# ############## Epochs 50 ###############

# fcLayer = FullConnected(1,50)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(50,50)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(50,50)
# actLayer3 = Activation(tanh, tanh_derivative)
# fcLayer4 = FullConnected(50,50)
# actLayer4 = Activation(tanh, tanh_derivative)
# fcLayer5 = FullConnected(50,1)
# actLayer5 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5]

# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 50, 0.005)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig9 = plt.figure('Noisy Sine Wave - 50 Epochs')
# ax9 = fig9.add_subplot()
# ax9.scatter(x, unwrapped_noisy)
# ax9.scatter(x, y_noisy)
# ax9.scatter(x,y,c='red', linewidths=0.01)
# ax9.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax9.set_title('Epochs = 50')
# ax9.set_xlabel('Radians')
# ax9.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig10 = plt.figure('R_sq Comparison - 50 Epochs')
# ax10 = fig10.add_subplot()
# ax10.scatter(y_noisy, unwrapped_noisy)
# ax10.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax10.legend(['Data Comparison', '100% Correlation'])
# ax10.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax10.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax10.set_title('Epochs = 50')
# ax10.set_xlabel('True Value')
# ax10.set_ylabel('Predicted Value')

# ############## Epochs 75 ###############

# fcLayer = FullConnected(1,50)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(50,50)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(50,50)
# actLayer3 = Activation(tanh, tanh_derivative)
# fcLayer4 = FullConnected(50,50)
# actLayer4 = Activation(tanh, tanh_derivative)
# fcLayer5 = FullConnected(50,1)
# actLayer5 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5]

# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 75, 0.005)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig9 = plt.figure('Noisy Sine Wave - 75 Epochs')
# ax9 = fig9.add_subplot()
# ax9.scatter(x, unwrapped_noisy)
# ax9.scatter(x, y_noisy)
# ax9.scatter(x,y,c='red', linewidths=0.01)
# ax9.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax9.set_title('Epochs = 75')
# ax9.set_xlabel('Radians')
# ax9.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig10 = plt.figure('R_sq Comparison - 75 Epochs')
# ax10 = fig10.add_subplot()
# ax10.scatter(y_noisy, unwrapped_noisy)
# ax10.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax10.legend(['Data Comparison', '100% Correlation'])
# ax10.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax10.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax10.set_title('Epochs = 75')
# ax10.set_xlabel('True Value')
# ax10.set_ylabel('Predicted Value')

# ############## Epochs 100 ###############

# fcLayer = FullConnected(1,50)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(50,50)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(50,50)
# actLayer3 = Activation(tanh, tanh_derivative)
# fcLayer4 = FullConnected(50,50)
# actLayer4 = Activation(tanh, tanh_derivative)
# fcLayer5 = FullConnected(50,1)
# actLayer5 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5]

# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 100, 0.005)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig9 = plt.figure('Noisy Sine Wave - 100 Epochs')
# ax9 = fig9.add_subplot()
# ax9.scatter(x, unwrapped_noisy)
# ax9.scatter(x, y_noisy)
# ax9.scatter(x,y,c='red', linewidths=0.01)
# ax9.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax9.set_title('Epochs = 100')
# ax9.set_xlabel('Radians')
# ax9.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig10 = plt.figure('R_sq Comparison - 100 Epochs')
# ax10 = fig10.add_subplot()
# ax10.scatter(y_noisy, unwrapped_noisy)
# ax10.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax10.legend(['Data Comparison', '100% Correlation'])
# ax10.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax10.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax10.set_title('Epochs = 100')
# ax10.set_xlabel('True Value')
# ax10.set_ylabel('Predicted Value')

# ############## Epochs 125 ###############

# fcLayer = FullConnected(1,50)
# actLayer = Activation(tanh, tanh_derivative)
# fcLayer2 = FullConnected(50,50)
# actLayer2 = Activation(tanh, tanh_derivative)
# fcLayer3 = FullConnected(50,50)
# actLayer3 = Activation(tanh, tanh_derivative)
# fcLayer4 = FullConnected(50,50)
# actLayer4 = Activation(tanh, tanh_derivative)
# fcLayer5 = FullConnected(50,1)
# actLayer5 = Activation(tanh, tanh_derivative)
# layers = [fcLayer, actLayer, fcLayer2, actLayer2, fcLayer3, actLayer3, fcLayer4, actLayer4, fcLayer5, actLayer5]

# model_noisy = Network(layers, mse, mse_derivative)
# # train model
# model_noisy.training(x, y_noisy, 125, 0.005)
# # test model
# output = model_noisy.predict(x)
# unwrapped_noisy = []
# for ii in range(len(output)):
#     unwrapped_noisy.append(output[ii][0][0])

# fig9 = plt.figure('Noisy Sine Wave - 125 Epochs')
# ax9 = fig9.add_subplot()
# ax9.scatter(x, unwrapped_noisy)
# ax9.scatter(x, y_noisy)
# ax9.scatter(x,y,c='red', linewidths=0.01)
# ax9.legend(['Predicted', 'Actual - noisy', 'Actual'])
# ax9.set_title('Epochs = 125')
# ax9.set_xlabel('Radians')
# ax9.set_ylabel('Output')

# r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
# print('R_sq')
# print(r_sq)
# MSE_val = mse(y_noisy, unwrapped_noisy)
# print('MSE')
# print(MSE_val)

# fig10 = plt.figure('R_sq Comparison - 125 Epochs')
# ax10 = fig10.add_subplot()
# ax10.scatter(y_noisy, unwrapped_noisy)
# ax10.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
# ax10.legend(['Data Comparison', '100% Correlation'])
# ax10.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
# ax10.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
# ax10.set_title('Epochs = 125')
# ax10.set_xlabel('True Value')
# ax10.set_ylabel('Predicted Value')

############## Optimal Sine ###############

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
# train model
model_noisy.training(x, y_noisy, 100, 0.005)
# test model
output = model_noisy.predict(x)
unwrapped_noisy = []
for ii in range(len(output)):
    unwrapped_noisy.append(output[ii][0][0])

fig9 = plt.figure('Noisy Sine Wave - Optimal Model')
ax9 = fig9.add_subplot()
ax9.scatter(x, unwrapped_noisy)
ax9.scatter(x, y_noisy)
ax9.scatter(x,y,c='red', linewidths=0.01)
ax9.legend(['Predicted', 'Actual - noisy', 'Actual'])
ax9.set_title('Optimal Model')
ax9.set_xlabel('Radians')
ax9.set_ylabel('Output')

r_sq, std_err = r_sq_stats(y_noisy, unwrapped_noisy)
print('R_sq')
print(r_sq)
MSE_val = mse(y_noisy, unwrapped_noisy)
print('MSE')
print(MSE_val)

fig10 = plt.figure('R_sq Comparison - Optimal Model')
ax10 = fig10.add_subplot()
ax10.scatter(y_noisy, unwrapped_noisy)
ax10.plot([min(y_noisy), max(y_noisy)], [min(y_noisy), max(y_noisy)], c='red')
ax10.legend(['Data Comparison', '100% Correlation'])
ax10.text(-1.0,0.65,f'R^2 = {round(r_sq,3)}')
ax10.text(-1.0,0.55,f'MSE = {round(MSE_val,3)}')
ax10.set_title('Optimal Model')
ax10.set_xlabel('True Value')
ax10.set_ylabel('Predicted Value')



plt.show()








