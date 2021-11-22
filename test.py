from torch import load
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import Bottleneck, ResNet
from tqdm import tqdm
device = 0
batch = 100
model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=48)
model.load_state_dict(load('ig_resnext101_32x48-3e41cc8a.pth'))
model.fc.load_state_dict(load('classifier.pt'))
model.eval().requires_grad_(False).to(device)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])
dataset = ImageFolder('test_new', preprocess)
loader = DataLoader(dataset, batch, num_workers=4)
classes = ImageFolder('val').classes
file = open('submit.csv', 'w')
index = 0
for image, _ in tqdm(loader, total=len(loader)):
    _, labels = model(image.to(device)).topk(5)
    for label in labels:
        print(dataset.imgs[index][0][14:],
              classes[label[0]], classes[label[1]], classes[label[2]],
              classes[label[3]], classes[label[4]], sep=',', file=file)
        index += 1
