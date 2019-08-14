import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from dataset import FabricDefects


def to_gpu(data):
    for i in range(len(data)):
        if type(data[i]) is dict:
            for key in data[i]:
                data[i][key] = data[i][key].cuda()
        else:
            data[i] = data[i].cuda()
    return data


dataset = FabricDefects('data/guangdong1_round1_train1_20190809/', scale=4)
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=6,
    collate_fn=lambda x: ([x[i][0] for i in range(len(x))], [x[i][1] for i in range(len(x))]),
    pin_memory=True,
    drop_last=True,
)


model = fasterrcnn_resnet50_fpn(num_classes=29).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(100):
    for idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = to_gpu(inputs), to_gpu(targets)
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
                print(output)
            model.train()
