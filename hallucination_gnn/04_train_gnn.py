import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

LABEL_NAMES = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

class GATClassifier(torch.nn.Module):
    def __init__(self, in_dim=384, hidden_dim=128, num_classes=3):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=4, concat=True)  # more heads = better learning
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=2, concat=True)
        self.gat3 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=True)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)

        x = F.elu(self.gat2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)

        x = F.elu(self.gat3(x, edge_index))

        x = global_mean_pool(x, batch)
        out = self.fc(x)
        return out

def train_one_epoch(model, loader, optimizer, device, class_weights):
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)

        # ✅ Weighted loss to balance classes
        loss = F.cross_entropy(out, data.y, weight=class_weights)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)

        y_true.extend(data.y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

    return np.array(y_true), np.array(y_pred)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading graphs...")
    train_graphs = torch.load("graphs/train_graphs.pt", weights_only=False)
    val_graphs = torch.load("graphs/val_graphs.pt", weights_only=False)

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)

    model = GATClassifier().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # ✅ Class weights (increase NEI weight slightly)
    class_weights = torch.tensor([1.15, 1.15, 1.25]).to(device)

    best_acc = 0.0
    best_epoch = 0

    EPOCHS = 20

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, class_weights)
        y_true, y_pred = evaluate(model, val_loader, device)
        acc = accuracy_score(y_true, y_pred)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {loss:.4f} | Val Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), "models/best_gnn_model.pth")

    print("\nBest Validation Accuracy:", best_acc)
    print("Best Epoch:", best_epoch)

    # Load best model for final report
    model.load_state_dict(torch.load("models/best_gnn_model.pth", map_location=device))
    y_true, y_pred = evaluate(model, val_loader, device)

    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    with open("outputs/results.txt", "w") as f:
        f.write("Best Validation Accuracy: " + str(best_acc) + "\n")
        f.write("Best Epoch: " + str(best_epoch) + "\n\n")
        f.write("Classification Report:\n" + report + "\n\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n")

    print("\nSaved outputs/results.txt")
    print("Saved models/best_gnn_model.pth")

if __name__ == "__main__":
    main()
