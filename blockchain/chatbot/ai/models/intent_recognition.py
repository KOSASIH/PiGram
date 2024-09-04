import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertModel

class IntentRecognitionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class IntentRecognitionModel(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(IntentRecognitionModel, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return outputs

def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

def evaluate_model(model, device, test_loader):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.scores, 1)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy:.4f}')

def main():
    # Load the dataset
    data = pd.read_csv('intent_recognition_data.csv')

    # Create a tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create a dataset and data loader
    dataset = IntentRecognitionDataset(data, tokenizer, max_len=512)
    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create a model
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    num_classes = len(set(data['label']))
    model = IntentRecognitionModel(bert_model, num_classes)

    # Set the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        train_model(model, device, data_loader, optimizer, epoch)

    # Evaluate the model
    test_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluate_model(model, device, test_data_loader)

if __name__ == "__main__":
    main()
