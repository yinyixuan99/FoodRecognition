```sh
mkdir test_new/0 && mv test_new/*.jpg test_new/0  # 只是为了偷懒套用 torchvision.datasets.ImageFolder
wget https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth  # 参见 https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
python data.py
python classify.py
python test.py
```