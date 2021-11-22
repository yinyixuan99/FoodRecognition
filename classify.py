from torch import load, no_grad, save
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam
device = 0
batch = 100
classifier = Linear(2048, 1000, device=device)
criterion = CrossEntropyLoss().to(device)
optimizer = Adam(classifier.parameters())
trainFeatures, trainLabels, valFeatures, valLabels = load('data.pt')
trainFeatures = trainFeatures.to(device)
trainLabels = trainLabels.to(device)
valFeatures = valFeatures.to(device)
valLabels = valLabels.to(device)
for epoch in range(500):
    print('epoch', f'{epoch:3}', end='\t')
    optimizer.zero_grad(True)
    loss = criterion(classifier(trainFeatures), trainLabels)
    print('loss', f'{loss.item():.3}', end='\t')
    loss.backward()
    optimizer.step()
    with no_grad():
        _, label = classifier(valFeatures).topk(5)
        print('accuracy', f'{sum(label[:, 0] == valLabels).item() / 26731:.2%}',
              f'{sum((label == valLabels.view(-1, 1)).any(1)).item() / 26731:.2%}')
save(classifier.requires_grad_(False).state_dict(), f'classifier.pt')
