import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Load dataset
with open('/content/content/dataset2.json') as f:
    intents = json.load(f)["intents"]

# Tokenization (Simplified)
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.idx = 2  # Reserving 0 and 1 for padding and unknown tokens

    def fit(self, texts):
        for text in texts:
            for word in text.split():
                if word.lower() not in self.vocab:
                    self.vocab[word.lower()] = self.idx
                    self.idx += 1

    def encode(self, text, max_length=20):
        tokens = [self.vocab.get(word.lower(), 1) for word in text.split()]  # 1 for unknown tokens
        return tokens[:max_length] + [0] * (max_length - len(tokens))

tokenizer = SimpleTokenizer()
patterns = [p for intent in intents for p in intent['patterns']]
tokenizer.fit(patterns)

# Dataset class
class MentalHealthDataset(Dataset):
    def __init__(self, intents, tokenizer):
        self.data = [(tokenizer.encode(p), intent["tag"]) for intent in intents for p in intent['patterns']]
        self.labels = LabelEncoder().fit([intent["tag"] for intent in intents])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, label = self.data[idx]
        return torch.tensor(tokens), self.labels.transform([label])[0]

# Model
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

# Hyperparameters and Training Setup
vocab_size = tokenizer.idx
embed_dim = 64
hidden_dim = 128
output_dim = len(set([intent["tag"] for intent in intents]))

model = SimpleLSTM(vocab_size, embed_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# DataLoader
dataset = MentalHealthDataset(intents, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training Loop
model.train()
for epoch in range(200):
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')

# Inference (Chat)
def chat1():
    model.eval()
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        encoded = torch.tensor(tokenizer.encode(user_input)).unsqueeze(0)
        output = model(encoded)
        predicted_label = torch.argmax(output, dim=1)
        tag = dataset.labels.inverse_transform(predicted_label.cpu().numpy())[0]
        for intent in intents:
            if intent["tag"] == tag:
                print(f"Mental Health Bot: {random.choice(intent['responses'])}")
