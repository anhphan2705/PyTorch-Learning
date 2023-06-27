import torch

#data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# random guess
w = 1

# set alpha learning rate
a = 0.01

# model for forward pass
def forward(x):
    return x*w

# loss funtion
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)*(y_pred - y)

# compute gradient d_loss/d_w = d( ((x*w)-y)^2 )
def gradient(x, y):
    return 2*x*(x*w - y)

# Before training
print('Predict (before training)', 4, forward(4))

# Training loop
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        print("grad: ", x_val, y_val, grad)
        w = w - a * grad
        l = loss(x_val, y_val)
    print("Progress {}: w= {}, loss= {}".format(epoch, w, l))
    
# After training
print('Predict (after training)', 4, forward(4))
        
        