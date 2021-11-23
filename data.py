from torch import cat, load, long, save, zeros
from torch.nn import Identity
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import Bottleneck, ResNet
from tqdm import tqdm
device = 0
batch = 100
model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=48)
model.load_state_dict(load('ig_resnext101_32x48-3e41cc8a.pth'))
model.fc = Identity()
model.eval().requires_grad_(False).to(device)
trainFeatures = zeros(0, 2048, device=device)
trainLabels = zeros(0, dtype=long, device=device)
for i in range(1, 5):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(i & 1),
        transforms.RandomVerticalFlip(i & 2),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    trainLoader = DataLoader(ImageFolder('Train_qtc', preprocess),
                             batch, num_workers=4)
    for image, label in tqdm(trainLoader, total=len(trainLoader)):
        trainFeatures = cat((trainFeatures, model(image.to(device))))
        trainLabels = cat((trainLabels, label.to(device)))
valFeatures = zeros(0, 2048, device=device)
valLabels = zeros(0, dtype=long, device=device)
valLoader = DataLoader(ImageFolder('val', preprocess), batch, num_workers=4)
for image, label in tqdm(valLoader, total=len(valLoader)):
    valFeatures = cat((valFeatures, model(image.to(device))))
    valLabels = cat((valLabels, label.to(device)))
save((trainFeatures, trainLabels, valFeatures, valLabels), 'data.pt')
