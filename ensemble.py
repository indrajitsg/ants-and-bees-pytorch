import torch.nn as nn
from model.resnet import resnet18
from model.inception import inception_v3
from model.vgg import vgg16
from model.alexnet import alexnet
from data_loader.data_loader import get_data_loader
from model.utils import LayerActivations
from torch.utils.data import DataLoader

model1 = vgg16(pretrained=False, num_classes=2)
model2 = resnet18(pretrained=False, num_classes=2)
model3 = alexnet(pretrained=False, num_classes=2)

model1 = nn.Sequential(*list(model1.children())[:-2])

for p in model1.parameters():
    p.requires_grad = False

# print(model1)

model2 = nn.Sequential(*list(model2.children())[:-2])
for p in model2.parameters():
    p.requires_grad = False

# print(model2)

for p in model3.parameters():
    p.requires_grad = False

print(model3)

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
    trn_vgg_features.extend(out.data())

val_labels = []
val_vgg_features = []
for d, la in data_loader['val']:
    out = model1(d)
    out = out.view(out.size(0), -1)
    val_labels.extend(la)
    val_vgg_features.extend(out.data())

# Extract features for ResNet
trn_resnet_features = []
for d, la in data_loader['train']:
    out = model2(d)
    out = out.view(out.size(0), -1)
    trn_resnet_features.extend(out.data())

val_resnet_features = []
for d, la in data_loader['val']:
    out = model2(d)
    out = out.view(out.size(0), -1)
    val_resnet_features.extend(out.data())

# Extract features for Inception
trn_alex_features = []
for d, la in data_loader['train']:
    out = model3(d)
    out = out.view(out.size(0), -1)
    trn_alex_features.extend(out.data())

val_alex_features = []
for d, la in data_loader['val']:
    out = model3(d)
    out = out.view(out.size(0), -1)
    val_alex_features.extend(out.data())


class FeaturesDataset(Dataset):
    def __init__(self,featlst1,featlst2,featlst3,labellst):
        self.featlst1 = featlst1
        self.featlst2 = featlst2
        self.featlst3 = featlst3
        self.labellst = labellst

    def __getitem__(self,index):
        return (self.featlst1[index],self.featlst2[index],self.featlst3[index],self.labellst[index])

    def __len__(self):
        return len(self.labellst)


trn_feat_dset = FeaturesDataset(trn_vgg_features, trn_resnet_features, trn_alex_features, trn_labels)
val_feat_dset = FeaturesDataset(val_vgg_features, val_resnet_features, val_alex_features, val_labels)

trn_feat_loader = DataLoader(trn_feat_dset,batch_size=64,shuffle=True)
val_feat_loader = DataLoader(val_feat_dset,batch_size=64)

print(trn_feat_loader)
