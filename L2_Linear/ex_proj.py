import torch
import matplotlib.pyplot as plt
import numpy as np

#data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w_list = []
mse_list = []

# random guess
w = 1

# model for forward pass
def forward(x):
    return x*w

# loss funtion
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)*(y_pred - y)

# Mean square error function
def mse(loss_sum, count):
    return loss_sum / count

# Print and Calc MSE
for w in np.arange(0.0, 4.1, 0.1):                      # Guess range
    l_sum = 0
    print('w= ', w)
    for x_val, y_val in zip(x_data, y_data):            # zip value x and y from 2 different x and y data set
        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        l_sum += l                                      # find the sum of loss from all x, y pair
        print("\t", x_val, y_val, y_pred_val, l)
    mse_value = mse(l_sum, len(x_data))
    print("MSE = ", mse_value)
    w_list.append(w)                                        
    mse_list.append(mse_value)
    
# Plot graph 
plt.plot(w_list, mse_list)
plt.xlabel("loss")
plt.ylabel("w")
plt.show()