import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Define the neural network model
class YourTrainDatasetClass(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(YourTrainDatasetClass, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

# Create an instance of your train dataset class
train_dataset = YourTrainDatasetClass()

# Create a DataLoader object to load the data in batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Now you can use the `train_loader` to iterate over the data in batches
for batch_data in train_loader:
    # Process the batch_data, perform training steps, etc.
    pass