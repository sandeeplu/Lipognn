import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import from_smiles
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Data Loading and Preprocessing
df = pd.read_csv('../../logP_final.csv')
df['label'] = (df['exp'] > 0.281620).astype(int)

def smiles_to_graph(smiles, label):
    data = from_smiles(smiles)
    data.y = torch.tensor([label], dtype=torch.float)
    return data

graph_list = []
for i, row in df.iterrows():
    try:
        graph = smiles_to_graph(row['smiles'], row['label'])
        if graph.x.shape[1] == 9:  # Ensure feature dimension consistency
            graph_list.append(graph)
    except Exception as e:
        print(f"Error at index {i}: {e}")

# 2. Dataset and DataLoader
class MyOwnDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.')
        self.data, self.slices = self.collate(data_list)

dataset = MyOwnDataset(graph_list)
num_graphs = len(dataset)
indices = np.arange(num_graphs)
np.random.shuffle(indices)
split = int(0.8 * num_graphs)
train_idx, test_idx = indices[:split], indices[split:]
train_dataset = dataset[train_idx]
test_dataset = dataset[test_idx]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 3. Model Definition with Your Parameters
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=4, dropout=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = global_mean_pool(x, batch)
        out = self.lin(pooled)
        return out, pooled  # Return both output and pooled embeddings

# 4. Training and Evaluation Functions
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.view(-1), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.batch)
            prob = torch.sigmoid(out.view(-1))
            preds.extend((prob > 0.5).cpu().numpy())
            targets.extend(data.y.cpu().numpy())
    acc = accuracy_score(targets, preds)
    return acc

def evaluate(model, loader):
    model.eval()
    all_probs, all_labels, all_embeddings = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, emb = model(data.x, data.edge_index, data.batch)
            prob = torch.sigmoid(out.view(-1)).cpu().numpy()
            all_probs.extend(prob)
            all_labels.extend(data.y.cpu().numpy())
            all_embeddings.append(emb.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return np.array(all_probs), np.array(all_labels), all_embeddings

# 5. Training Loop with Early Stopping and Your Hyperparameters
in_channels = 9
hidden_channels = 192
out_channels = 1
num_layers = 4
dropout = 0.2
lr = 0.0004407086844982349
weight_decay = 0.000004856076517711844
patience = 10

model = GraphSAGEModel(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    num_layers=num_layers,
    dropout=dropout
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)
criterion = torch.nn.BCEWithLogitsLoss()

best_loss = float('inf')
patience_counter = 0
train_accs, test_accs = [], []

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

for epoch in range(1, 101):
    loss = train(model, train_loader, optimizer, criterion)
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    # Early stopping
    if loss < best_loss:
        best_loss = loss
        patience_counter = 0
        torch.save(model.state_dict(), 'models/best_model_cls.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# 6. Final Evaluation and Visualization
model.load_state_dict(torch.load('models/best_model_cls.pth'))

probs, labels, embeddings = evaluate(model, test_loader)
preds = (probs > 0.5).astype(int)
acc = accuracy_score(labels, preds)
prec = precision_score(labels, preds)
rec = recall_score(labels, preds)
f1 = f1_score(labels, preds)
print(f"\nFinal Test Metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

# t-SNE plot
tsne = TSNE(n_components=2, random_state=seed)
emb_2d = tsne.fit_transform(embeddings)
plt.figure(figsize=(8,6))
sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=labels, palette='Set1', alpha=0.7)
plt.title('t-SNE of Graph Embeddings')
plt.savefig('plots/tsne.png')
plt.close()

# KDE plot of predicted probabilities
plt.figure(figsize=(8,6))
sns.kdeplot(probs[labels==0], label='Class 0', fill=True)
sns.kdeplot(probs[labels==1], label='Class 1', fill=True)
plt.xlabel('Predicted Probability')
plt.title('KDE of Predicted Probabilities')
plt.legend()
plt.savefig('plots/kde_probs.png')
plt.close()

# Accuracy curves
plt.figure(figsize=(8,6))
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()
plt.savefig('plots/accuracy_curves.png')
plt.close()

# Confusion matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('plots/confusion_matrix.png')
plt.close()
