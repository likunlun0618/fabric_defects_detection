import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
import torchvision.transforms.functional as TF

from dataset import FabricDefects
from utils.utils import *
import os
import json
from tqdm import tqdm
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.models.detection.rpn import AnchorGenerator
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def to_gpu(data):
    for i in range(len(data)):
        if type(data[i]) is dict:
            for key in data[i]:
                data[i][key] = data[i][key].cuda()
        else:
            data[i] = data[i].cuda()
    return data


dataset = FabricDefects('data/guangdong1_round1_train1_20190818/guangdong1_round1_train1_20190818/', scale=1)
train_loader = DataLoader(
    Subset(dataset, range(0, int(len(dataset)*1.0))),
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
'''
test_path='./data/guangdong1_round1_testA_20190818/'
test_dataset=ImageFolder(test_path)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=6, pin_memory=True,drop_last=False)
'''


model = fasterrcnn_resnet50_fpn(num_classes=20).cuda()
'''
anchor_generator = AnchorGenerator(sizes=((16,32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),))

backbone=torchvision.models.resnet101(pretrained=True).layer3#.features
backbone.out_channels = 256#2048
min_size=1000
max_size=2448
model = FasterRCNN(backbone,#min_size=min_size,max_size=max_size, 
    num_classes=20,
    rpn_anchor_generator=anchor_generator).cuda()
'''
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(1000):#
    if 1:
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
    
    if 0:
        model.load_state_dict(torch.load('./model.pth'))
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
                #img = dataset.visualize(images[i], targets[i]['boxes'], targets[i]['labels'])
                img = DrawBBox(images[i],targets[i]['boxes'],targets[i]['labels'],False)
                if len(output[i]['scores']) > 0:
                    #img = dataset.visualize(img, output[i]['boxes'], output[i]['labels'], (0,0,255))
                    threshhold=0.9
                    img = DrawBBox(img,output[i]['boxes'][output[i]['scores']>threshhold],output[i]['labels'][output[i]['scores']>threshhold])
                cv2.imwrite('visual/%d.jpg'%n, img)
                n += 1
            model.train()
            #break
    #break

if 0:
    model.load_state_dict(torch.load('./model.pth'))
    test_path='./data/guangdong1_round1_testA_20190818/'
    filenames = os.listdir(test_path)
    model.eval()
    result=[]
    min_label=30
    max_label=-1
    for file_name in tqdm(filenames):
        images=cv2.imread(test_path+file_name)
        img=np.array(images/255.0,dtype=np.float32).transpose(2,0,1)
        img=torch.from_numpy(img)

        inputs=[]
        inputs.append(img)
        inputs = to_gpu(inputs.copy())
        with torch.no_grad():
            output = model(inputs)
        

        bbox=output[0]['boxes'].cpu().numpy()
        labels=output[0]['labels'].cpu().numpy()
        scores=output[0]['scores'].cpu().numpy()
        bbox=np.array(bbox,dtype=np.float32)
        bbox=np.round(bbox,decimals=2)
        bbox=bbox.tolist()



        #choose score > 0.5
        threshhold=0.4
        bbox_output=[]
        labels_output=[]
        scores_output=[]
        for ii in range(len(scores)):
            if scores[ii] >threshhold:
                bbox_temp=[]
                for jj in range(len(bbox[ii])):
                    bbox_temp.append(round(bbox[ii][jj],2))
                bbox_output.append(bbox_temp)
                labels_output.append(labels[ii])
                scores_output.append(scores[ii])

        #NMS
        bbox_output, scores_output, labels_output=dataset.nms(bbox_output, scores_output, labels_output,0.8)
        if len(bbox_output)!=0:
            bbox_output=np.array(bbox_output)
            bbox_output, scores_output, labels_output=dataset.bounding_box_delete(bbox_output, scores_output, labels_output)

        # write json
            labels_output=labels_output.tolist()
            bbox_output=bbox_output.tolist()
            scores_output=scores_output.tolist()
            for ii in range(len(scores_output)):
                result.append({'name': file_name,'category': int(labels_output[ii]),'bbox':bbox_output[ii],'score': float(scores_output[ii])})  

        # draw img
        if 0:
            img = dataset.visualize(images,np.array(bbox_output,dtype=np.int32), labels_output, (0,0,255))
            cv2.imwrite('visual/'+file_name, img)

    with open('result.json', 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))
        

