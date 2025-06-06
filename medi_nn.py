import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(49, 32)  # input size 49, output size 32
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)  # output size 10

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
# Instantiate model
model = SimpleNN()

# Loss function (for classification, e.g. cross entropy)
criterion = nn.CrossEntropyLoss()

# Optimizer (e.g. Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy dataset: 100 samples, 49 features each
inputs = torch.randn(100, 49)

# Dummy labels: integers from 0 to 9 for 10 classes
labels = torch.randint(0, 10, (100,))

# Training loop
num_epochs = 20
batch_size = 10

for epoch in range(num_epochs):
    for i in range(0, len(inputs), batch_size):
        # Get batch data
        x_batch = inputs[i:i+batch_size]
        y_batch = labels[i:i+batch_size]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x_batch)
        
        # Compute loss
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")