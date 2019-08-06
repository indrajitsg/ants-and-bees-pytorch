import torch
import torch.nn as nn
from model.vgg import vgg16
from data_loader.data_loader import get_data_loader
from model.utils import LayerActivations
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model1 = vgg16(pretrained=False, num_classes=2)

model1 = nn.Sequential(*list(model1.children())[:-2])

model1 = model1.to(device)

model1.train(False)
model1.eval()

for p in model1.parameters():
    p.requires_grad = False

# Create data loader
data_loader = get_data_loader()

# Extract features for VGG16
trn_labels = []
trn_vgg_features = []
for d, la in data_loader['train']:
    d, _ = d.to(device), la.to(device)
    out = model1(d)
    out = out.view(out.size(0), -1)
    trn_labels.extend(la)
    trn_vgg_features.extend(out.cpu().data().numpy().tolist())
    # final_features.extend(o.cpu().data.numpy().tolist())

print(trn_labels)
print(trn_vgg_features)
