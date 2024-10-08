import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Data preprocessing (Normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define the DNN model
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Define the ConvNet model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*14*14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*14*14)  # Flatten the output
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the VGG model
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*3*3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256*3*3)
        x = self.classifier(x)
        return x

# Define the ResNet model
from torchvision.models import resnet18

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

# Function to select the model
def get_model(model_name):
    if model_name == 'DNN':
        return DNN()
    elif model_name == 'ConvNet':
        return ConvNet()
    elif model_name == 'VGG':
        return VGG()
    elif model_name == 'ResNet':
        return ResNet()
    else:
        raise ValueError("Unknown model name")
    
 # Function to save the model weights
def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f'Model weights saved to {filename}')

# Training function with micro epochs
def train(model, device, train_loader, optimizer, micro_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # Track metrics after each micro epoch
        if (batch_idx + 1) % micro_epochs == 0:
            epoch_loss = running_loss / micro_epochs
            epoch_accuracy = 100. * correct / total
            yield epoch_loss, epoch_accuracy

            # Reset metrics for next micro epoch
            running_loss = 0.0
            correct = 0
            total = 0

# Testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())
            all_probabilities.extend(torch.softmax(output, dim=1).cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    # Calculate F1 Score
    f1 = f1_score(all_targets, all_predictions, average='weighted')

    # Calculate AUC Score (micro average)
    auc = roc_auc_score(all_targets, np.array(all_probabilities), multi_class='ovr')

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    print(f'F1 Score: {f1:.4f}, AUC Score: {auc:.4f}')

    return test_loss, accuracy, all_targets, all_predictions, all_probabilities

# Plotting function for loss and accuracy
def plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies, model_name):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Micro Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, test_accuracies, label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Micro Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Plotting AUC-ROC and Precision-Recall curves
def plot_curves(all_targets, all_probabilities):
    num_classes = np.unique(all_targets).size
    plt.figure(figsize=(12, 5))

    # AUC-ROC Curve
    plt.subplot(1, 2, 1)
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_targets, np.array(all_probabilities)[:, i], pos_label=i)
        roc_auc = roc_auc_score(all_targets, np.array(all_probabilities), multi_class='ovr')
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(all_targets, np.array(all_probabilities)[:, i], pos_label=i)
        plt.plot(recall, precision, label=f'Class {i}')

    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Main function to run experiments
def run_experiment(model_name, macro_epochs=1, micro_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for macro_epoch in range(macro_epochs):
        print(f"Starting Macro Epoch {macro_epoch + 1}")
        for train_loss, train_accuracy in train(model, device, train_loader, optimizer, micro_epochs):
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Test after each micro epoch
            test_loss, test_accuracy, all_targets, all_predictions, all_probabilities = test(model, device, test_loader)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
        
        save_model(model, f'HW2_weights_{model_name}.pth')

    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)
    plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies, model_name)
    plot_curves(all_targets, all_probabilities)


def test_model(file_path, model_name, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(file_path))
    model.eval()

    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())
            all_probabilities.extend(torch.softmax(output, dim=1).cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    auc = roc_auc_score(all_targets, np.array(all_probabilities), multi_class='ovr')

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    print(f'F1 Score: {f1:.4f}, AUC Score: {auc:.4f}')


# Run the experiment
if __name__ == '__main__':
    # example for training the models
    # model_names = ['DNN', 'ConvNet', 'VGG', 'ResNet']
    # for model_name in model_names:
    #     print(f"\nTesting model: {model_name}")
    #     run_experiment(model_name)
    
    # example for running a saved model
    test_model('/Users/danielvennemeyer/Workspace/Deep Learning/HW2/HW2_weights_VGG.pth', 'VGG', test_loader)


