import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms.functional as TF

from dataset import FabricDefects


def to_gpu(data):
    for i in range(len(data)):
        if type(data[i]) is dict:
            for key in data[i]:
                data[i][key] = data[i][key].cuda()
        else:
            data[i] = data[i].cuda()
    return data


dataset = FabricDefects('data/guangdong1_round1_train1_20190809/', scale=1)
train_loader = DataLoader(
    Subset(dataset, range(0, int(len(dataset)*0.9))),
    batch_size=3,
    shuffle=True,
    num_workers=6,
    collate_fn=lambda x: ([x[i][0] for i in range(len(x))], [x[i][1] for i in range(len(x))]),
    pin_memory=True,
    drop_last=True,
)
valid_loader = DataLoader(
    Subset(dataset, range(int(len(dataset)*0.9), len(dataset))),
    batch_size=3,
    shuffle=True,
    num_workers=6,
    collate_fn=lambda x: ([x[i][0] for i in range(len(x))], [x[i][1] for i in range(len(x))]),
    pin_memory=True,
    drop_last=False
)


model = fasterrcnn_resnet50_fpn(num_classes=29).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(1000):
    for idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = to_gpu(inputs), to_gpu(targets)
        if idx % 100 != 0:
            loss_dict = model(inputs, targets)
            loss = 0.
            for key in loss_dict:
                loss += loss_dict[key]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, idx, loss.item(), end='\r')

        if idx % 100 == 0:
            print('')
            model.eval()
            with torch.no_grad():
                output = model(inputs)
                for i in range(len(output)):
                    print('%d:'%i)
                    print('output labels:', output[i]['labels'])
                    print('targets labels:', targets[i]['labels'])
                    print('output boxes:', output[i]['boxes'])
                    print('targets boxes:', targets[i]['boxes'])
                    print('output scores:', output[i]['scores'])
            model.train()
            torch.save(model.state_dict(), 'model.pth')

    n = 0
    for idx, (images, targets) in enumerate(valid_loader):
        inputs = to_gpu(images.copy())
        model.eval()
        with torch.no_grad():
            output = model(inputs)
        for i in range(len(inputs)):
            # img = TF.to_pil_image(images[i].clamp(0, 1))
            # img = np.array(img)
            # cv2.imwrite('test.jpg', img)
            # input()
            img = dataset.visualize(images[i], targets[i]['boxes'], targets[i]['labels'])
            if len(output[i]['scores']) > 0:
                img = dataset.visualize(img, output[i]['boxes'], output[i]['labels'], (0,0,255))
            cv2.imwrite('visual/%d.jpg'%n, img)
            n += 1
        model.train()
