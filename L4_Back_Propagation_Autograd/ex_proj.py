import torch
from torch.autograd import Variable

# data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# random guess
w = Variable(torch.Tensor([1.0]), requires_grad=True)

# set alpha learning rate
a = 0.01

# model for forward pass
def forward(x):
    return x*w

# loss funtion
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)*(y_pred - y)

# Before training
print('Predict (before training)', 4, forward(4).data[0])

# Training loop
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)                                  # Forward pass
        l.backward()                                            # Backward pass
        print("\tgrad: ", x_val, y_val, w.grad.data[0])         # .data is to store the data of the variable
        w.data = w.data - a * w.grad.data                       # d_loss/d_w stored in w.grad
        # Manually zero the gradients after updating weights
        w.grad.data.zero_()
    print("Progress: ", epoch, l.data[0])
    
# After training
print('Predict (after training)', 4, forward(4).data[0])
        
        