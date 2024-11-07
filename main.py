import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

class ParkinsonsGaitDataset(Dataset):
    def __init__(self, data_dir, demographics, features, transform=None):
        self.data_dir = data_dir
        self.demographics = demographics
        self.features = features
        self.transform = transform
        
        self.data = []  # List to hold the data and labels
        self.labels = []  # List to hold the labels
        
        # Load and preprocess data
        self.load_data()

    def load_data(self):
        for name in os.listdir(self.data_dir):
            sub_id = name.split('_')[0]
            sub_class = self.demographics[self.demographics['ID'] == sub_id]['Group'].values[0] - 1
            
            # Read the CSV file
            dataframe = pd.read_csv(os.path.join(self.data_dir, name))
            full_size = 400
            skip = 50
            
            for j in range(0, dataframe.shape[0], skip):
                if dataframe.shape[0] >= full_size + j:
                    temp = []
                    for feature in self.features:
                        temp.append(dataframe.iloc[j:j + full_size][feature].to_numpy())
                    
                    # Add the processed data and labels to the lists
                    self.data.append(np.array(temp))  # Append features
                    self.labels.append(sub_class)  # Append class label

        # Convert to numpy arrays
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features_data = self.data[idx]
        label = self.labels[idx]
        
        # Reshape features_data to (1, 400, 19) for Conv2d
        features_data = features_data.reshape(1, 400, 19)  # [channels, height, width]
        features_data = torch.tensor(features_data, dtype=torch.float32)

        # One-hot encode the label
        label_tensor = torch.zeros(2)  # Assuming binary classification
        label_tensor[label] = 1  # One-hot encoding
        label_tensor = label_tensor.float()  # Ensure label is float

        if self.transform:
            features_data = self.transform(features_data)

        return features_data, label_tensor

# class DiseaseModel(nn.Module):
#     def __init__(self, num_features):
#         super(DiseaseModel, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
#         self.batch_norm1 = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.batch_norm2 = nn.BatchNorm2d(64)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.flatten = nn.Flatten()

#         self.lstm_input_size = 64 * 5 * 100
#         self.sequence_length = 19

#         self.lstm1 = nn.LSTM(input_size=self.lstm_input_size, hidden_size=50, batch_first=True, dropout=0.2)
#         self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True, dropout=0.2)

#         self.fc1 = nn.Linear(50, 50)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(50, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.batch_norm1(x)
#         x = torch.relu(x)
#         x = self.pool1(x)

#         x = self.conv2(x)
#         x = self.batch_norm2(x)
#         x = torch.relu(x)
#         x = self.pool2(x)

#         x = self.flatten(x)

#         x = x.view(x.size(0), self.sequence_length, self.lstm_input_size)

#         x, _ = self.lstm1(x)
#         x, _ = self.lstm2(x)

#         x = self.fc1(x[:, -1, :])
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)

#         return x

class DiseaseModel(nn.Module):
    def __init__(self):
        super(DiseaseModel, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # Input channels = 1 for grayscale
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.5)

        # Second convolutional block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.5)

        # Third convolutional block
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 50, 128)  # Adjust based on output size after convolutions and pooling
        self.fc2 = nn.Linear(128, 2)  # Assuming binary classification

        # Activation functions
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Convolutional block 1
        x = self.conv1(x)         # (None, 1, 19, 400) -> (None, 32, 19, 400)
        x = torch.relu(x)
        x = self.conv2(x)       # (None, 32, 19, 400) -> (None, 32, 19, 400)
        x = torch.relu(x)
        x = self.pool1(x)      # (None, 32, 19, 400) -> (None, 32, 9, 200)
        x = self.dropout1(x)

        # Convolutional block 2
        x = self.conv3(x)         # (None, 32, 9, 200) -> (None, 64, 9, 200)
        x = torch.relu(x)
        x = self.conv4(x)         # (None, 64, 9, 200) -> (None, 64, 9, 200)
        x = torch.relu(x)
        x = self.pool2(x)         # (None, 64, 9, 200) -> (None, 64, 4, 100)
        x = self.dropout2(x)

        # Convolutional block 3
        x = self.conv5(x)         # (None, 64, 4, 100) -> (None, 64, 4, 100)
        x = torch.relu(x)
        x = self.conv6(x)         # (None, 64, 4, 100) -> (None, 64, 4, 100)
        x = torch.relu(x)
        x = self.pool3(x)         # (None, 64, 4, 100) -> (None, 64, 2, 50)

        # Flatten and fully connected layers
        x = self.flatten(x)        # (None, 64 * 2 * 50)
        x = self.fc1(x)           # (None, 128)
        x = self.elu(x)
        x = self.fc2(x)           # (None, 2)
        x = self.softmax(x)

        return x

input_dir = 'CSV'
demographics = pd.read_csv('./gait-parkinson/demographics.txt', delim_whitespace=True)
features = ['Time', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'Total_Force_Left', 'Total_Force_Right']

dataset = ParkinsonsGaitDataset(data_dir=input_dir, demographics=demographics, features=features)
train_size = int(0.7 * len(dataset))
valid_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(valid_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DiseaseModel().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
train_losses = []
valid_losses = []
valid_accuracies = []
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU
        
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        
        loss = criterion(outputs.squeeze(), labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    running_valid_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU
            
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            running_valid_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    valid_loss = running_valid_loss / len(valid_loader)
    valid_losses.append(valid_loss)
    
    # Calculate accuracy
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    predicted_classes = np.argmax(all_preds, axis=1)
    true_classes = np.argmax(all_labels, axis=1)
    valid_accuracy = np.mean(predicted_classes == true_classes)
    valid_accuracies.append(valid_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f} - Valid Accuracy: {valid_accuracy:.4f}")

torch.save(model,'model.pth')

# Plot loss and accuracy graphs
plt.figure(figsize=(12, 5))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs + 1), valid_losses, label='Validation Loss', marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Plot validation accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), valid_accuracies, label='Validation Accuracy', marker='o', color='orange')
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Save loss and accuracy plots
plt.savefig('loss_accuracy.png')
plt.close()

# Confusion Matrix
# Confusion matrix
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(np.argmax(outputs.cpu().numpy(), axis=1))

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

if len(y_true.shape) > 1:
    y_true = y_true[:, 0]  # Take the first column if needed
if len(y_pred.shape) > 1:
    y_pred = y_pred[:, 0]  # Take the first column if needed

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=['Control', 'PD'])
cmd.plot()
plt.title("Confusion Matrix")

# Save confusion matrix as PNG
plt.savefig('confusion_matrix.png')
plt.close()

# Save the model if needed
