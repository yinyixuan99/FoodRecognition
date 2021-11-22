```sh
mkdir test_new/0 && mv test_new/*.jpg test_new/0  # 只是为了偷懒套用 torchvision.datasets.ImageFolder
wget https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth  # 参见 https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
python data.py  # 用预训练resnext101提取特征
python classify.py  # 训练全连接分类层
python test.py  # 跑测试集
```
