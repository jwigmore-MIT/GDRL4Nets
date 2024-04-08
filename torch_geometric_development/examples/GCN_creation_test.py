import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GCN


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out


model = GCN(1, 2)
print(model)