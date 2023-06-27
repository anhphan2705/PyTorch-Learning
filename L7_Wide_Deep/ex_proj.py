from torch import nn, optim, from_numpy
import numpy as np

# Set up data
xy = np.loadtxt('Pytorch_Learning\L7_Wide_Deep\diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = from_numpy(xy[:, 0:-1])
y_data = from_numpy(xy[:, [-1]])
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')


# Initializing Model class
class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we want to instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)                        # (8x1)
        self.l3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return a Variable as output data.
        We can use Modules defined in the constructor as well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))                      # Using sigmoid 
        return y_pred
    
# Get a model
model = Model()

# Construct loss and optimizer
# The call to model.parameters() in the SGD constructor will contain the learnable parameters of the 2 nn.Linear modules, which are members of the model
criterion = nn.BCELoss(reduction='mean')                # BCE = Binary Cross Entropy
optimizer = optim.SGD(model.parameters(), lr=1e-1)        # SDG = stochastic gradient descent 

# Training loop
for epoch in range(100):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()