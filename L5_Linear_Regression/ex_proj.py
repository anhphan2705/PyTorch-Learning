import torch 
from torch.autograd import Variable

# Set up data
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

# Initializing Model class
class Model(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we want to instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1,1)                       # One in and one out
    
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return a Variable as output data.
        We can use Modules defined in the constructor as well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred
    
# Get a model
model = Model()

# Construct loss and optimizer
# The call to model.parameters() in the SGD constructor will contain the learnable parameters of the 2 nn.Linear modules, which are members of the model
criterion = torch.nn.MSELoss(size_average=False)                # MSE = mean square error
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)        # SDG = stochastic gradient descent 

# Training loop
for epoch in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)
    
    # Compute and print loss
    loss = criterion(y_pred, y_data)                            # Forward pass
    print(epoch, loss.data)
    
    # Zero gradient, perform a backward pass, and update the weights
    optimizer.zero_grad()                                       # Zero the gradients
    loss.backward()                                             # Backward pass
    optimizer.step()                                            # Update the weight and move to the next one
    
# After training
hour_var = Variable(torch.Tensor([[4.0]]))
print("Predict after training: ", 4, model.forward(hour_var).data[0][0])