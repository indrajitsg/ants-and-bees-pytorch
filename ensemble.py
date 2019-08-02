
import torch.nn as nn
from model.resnet import resnet18
from model.inception import inception_v3
from model.vgg import vgg16
from data_loader.data_loader import get_data_loader

model1 = vgg16(pretrained=False, num_classes=2)
model2 = resnet18(pretrained=False, num_classes=2)
model3 = inception_v3(pretrained=False, num_classes=2)

model1 = nn.Sequential(*list(model1.children())[:-2])

for p in model1.parameters():
    p.requires_grad = False

# print(model1)

model2 = nn.Sequential(*list(model2.children())[:-2])
for p in model2.parameters():
    p.requires_grad = False

# print(model2)

model3.aux_logits = False
for p in model3.parameters():
    p.requires_grad = False

# print(model3)

# Create data loader
data_loader = get_data_loader()

# print(data_loader)

# Extract features for VGG16
trn_labels = []
trn_vgg_features = []
for d, la in data_loader['train']:
    out = model1(d)
    out = out.view(out.size(0), -1)
    trn_labels.extend(la)
    trn_vgg_features.extend((out.data()))




