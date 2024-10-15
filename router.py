import torch
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import wandb
import ast
from rus_mapping import RUS_MAPPING, RUS_TO_LABEL

# Initialize W&B
wandb.init(project="custom_router")  # Set your project name

# Load datasets and create a combined dataset
combined_dataset = []
labels_mapping = {}

for dataset_name, label in RUS_MAPPING.items():
    dataset = load_dataset(dataset_name)
    for item in dataset['train']:
        # Convert the messages list to a string if it's not already
        messages = item['messages']
        if isinstance(messages, list):
            messages = ' '.join(map(str, messages))
        elif isinstance(messages, str):
            # If it's a string representation of a list, convert it to an actual list and then join
            try:
                messages = ' '.join(map(str, ast.literal_eval(messages)))
            except:
                pass  # Keep it as is if it's not a valid list representation
        
        combined_dataset.append({
            'message': messages,
            'label': RUS_TO_LABEL[label]
        })
    
    # Update labels_mapping
    if label not in labels_mapping:
        labels_mapping[label] = len(labels_mapping)

# Shuffle the combined dataset
import random
random.shuffle(combined_dataset)

# Split the data into training and validation sets
train_data, val_data = train_test_split(combined_dataset, test_size=0.2, random_state=42)

# Create a custom PyTorch dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['message'], torch.tensor(item['label'], dtype=torch.long)

# Create DataLoaders
train_dataset = CustomDataset(train_data)
val_dataset = CustomDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define the classifier model
class Classifier(nn.Module):
    def __init__(self, transformer_model_name, num_classes):
        super(Classifier, self).__init__()
        self.transformer = SentenceTransformer(transformer_model_name)
        self.fc1 = nn.Linear(self.transformer.get_sentence_embedding_dimension(), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, sentences):
        embeddings = self.transformer.encode(sentences, convert_to_tensor=True)
        x = self.relu(self.fc1(embeddings))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

# Initialize the classifier
num_classes = len(RUS_TO_LABEL)
model = Classifier(transformer_model_name='sentence-transformers/all-distilroberta-v1', num_classes=num_classes)

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
n_epochs = 10

# Directory to save the best model
runs_dir = "runs"
os.makedirs(runs_dir, exist_ok=True)

# Initialize best validation loss with infinity
best_valid_loss = float('inf')

# Log hyperparameters to W&B
wandb.config = {
    "learning_rate": 0.001,
    "epochs": n_epochs,
    "batch_size": 32,
}

def validate(model, val_loader, criterion, device):
    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    with torch.no_grad():
        for messages, labels in val_loader:
            messages = list(messages)
            labels = labels.to(device)
            
            outputs = model(messages)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            valid_correct += (predicted == labels).sum().item()

    valid_loss /= len(val_loader)
    valid_accuracy = valid_correct / len(val_loader.dataset)
    
    return valid_loss, valid_accuracy

# Training loop
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    for messages, labels in train_loader:
        messages = list(messages)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(messages)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = train_correct / len(train_loader.dataset)

    valid_loss, valid_accuracy = validate(model, val_loader, criterion, device)

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "valid_loss": valid_loss,
        "valid_accuracy": valid_accuracy,
    })

    print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(runs_dir, 'best_model.pt'))

print('Training complete.')
wandb.finish()