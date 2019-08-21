import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class FabricDefects(Dataset):

    '''
    1. 为了处理没有瑕疵的图片，给所有图片加了一个和图片大小相同的box，类别为0
    2. 瑕疵的种类和label的对应关系(顺序根据列表的sort方法得到)见defect_label
    '''

    '''   
    defect_label = {
        # 'normal':0,
        '三丝':1, '修痕':2, '双纬':3, '吊经':4, '断氨纶':5, '断经':6,
        '星跳':7, '松经':8, '死皱':9, '毛粒':10, '水渍':11, '污渍':12,
        '油渍':13, '浆斑':14, '浪纹档':15, '烧毛痕':16, '百脚':17, '破洞':18,
        '磨痕':19, '稀密档':20, '筘路':21, '粗经':22, '粗维':23, '纬缩':24,
        '结头':25, '花板跳':26, '跳花':27, '轧痕':28
    }'''

    def __init__(self, root, scale=1):
        '''
        args:
            root : 数据集的路径，如"./guangdong1_round1_train1_20190809/"
            scale: 控制图片缩小的倍数，4表示长宽均为原来的1/4
        '''
        self.defect_label = {
            '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
            '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
            '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
            '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
        }
        self.root = root

        with open(os.path.join(root, 'Annotations', 'anno_train.json'), 'r') as f:
            gt_result = json.load(f)
        self.defect_data = self.merge(gt_result)
        self.defect_size = len(self.defect_data)
        '''
        self.normal_data = os.listdir(os.path.join(root, 'normal_Images'))
        self.normal_data.sort()
        self.normal_size = len(self.normal_data)
        '''
        self.scale = scale

    def __getitem__(self, idx):
        if idx < self.defect_size:
            # 读取图片
            name = self.defect_data[idx]['name']
            img = cv2.imread(os.path.join(self.root, 'defect_Images', name))
            h, w = img.shape[:2]
            img = cv2.resize(img, (img.shape[1]//self.scale, img.shape[0]//self.scale))
            img = TF.to_tensor(img)

            # 读取bounding box
            bbox = self.defect_data[idx]['bbox']
            # bbox.append([0, 0, w-1, h-1])
            bbox = torch.tensor(bbox) / self.scale

            # 读取瑕疵的类别
            label = []
            for defect_name in self.defect_data[idx]['defect_name']:
                label.append(self.defect_label[defect_name]-1)
            # label.append(0)
            label = torch.tensor(label)
            return img, {'boxes':bbox, 'labels':label}
        '''
        else:
            # 图片
            name = self.normal_data[idx - self.defect_size]
            img = cv2.imread(os.path.join(self.root, 'normal_Images', name))
            img = cv2.resize(img, (img.shape[1]//self.scale, img.shape[0]//self.scale))
            img = TF.to_tensor(img)

            # 用和图片大小相同的bounding box表示没有瑕疵
            bbox = torch.tensor([0, 0, img.size(2)-1, img.size(1)-1]).float().unsqueeze(0)

            # 用0表示没有瑕疵
            label = torch.tensor([0])

            return img, {'boxes':bbox, 'labels':label}'''


    def __len__(self):
        # return self.defect_size + self.normal_size
        return self.defect_size

    def merge(self, gt_result):
        '''
        把gt_result中有相同name的项组合起来
        '''
        name_dict = {}
        for item in gt_result:
            name = item['name']
            if name in name_dict:
                name_dict[name]['bbox'].append(item['bbox'])
                name_dict[name]['defect_name'].append(item['defect_name'])
            else:
                name_dict[name] = {
                    'bbox': [item['bbox']],
                    'defect_name': [item['defect_name']]
                }

        result = []
        for name in name_dict:
            result.append({
                'name': name,
                'bbox': name_dict[name]['bbox'],
                'defect_name': name_dict[name]['defect_name']
            })

        return result

    def visualize(self, img, boxes, labels, color=(0,255,0)):
        if type(img) == torch.Tensor:
            img = TF.to_pil_image(img.clamp(0, 1))
            img = np.array(img)
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 1)
        return img
    '''
    def GetLabel(self,label_path="./defect_name.txt"):
        defect_label={}
        count=0
        for line in open(label_path,"r"): 
            defect_label[line.split(',')[0]]=count
            count=count+1
        return defect_label'''


if __name__ == '__main__':

    dataset = FabricDefects('data/guangdong1_round1_train1_20190818/guangdong1_round1_train1_20190818/', 1)
    for i in range(dataset.__len__()):
        img, target = dataset[i]
        img = dataset.visualize(img, target['boxes'], target['labels'])
        cv2.imshow('test',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.imwrite('test.jpg', img)
