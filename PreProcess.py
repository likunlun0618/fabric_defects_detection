import numpy as np
from tqdm import tqdm
import json
import cv2
from utils.utils import *
import os
with open("./data/guangdong1_round1_train1_20190818/guangdong1_round1_train1_20190818/Annotations/anno_train.json",'r') as load_f:
    load_dict = json.load(load_f)
    print(load_dict[:2])
'''
defect_name=[]
for _iter in load_dict:
    if _iter['defect_name'] not in defect_name:
        defect_name.append(_iter['defect_name'])
print(defect_name)
print(len(defect_name))'''
defect_label = {
            '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
            '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
            '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
            '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
        }
for i in range(1,21):
    if os.path.exists('./defect_visual/'+str(i)):
        continue
    else:
        os.mkdir('./defect_visual/'+str(i));

merge_dict=merge(load_dict)
count=np.zeros((20,1))
for _iter in tqdm(merge_dict):
    for key in defect_label:
        if key in _iter['defect_name']:
            '''
            idx=[i for i,x in enumerate(_iter['defect_name']) if x==key ]
            
            img=cv2.imread("./data/guangdong1_round1_train1_20190818/guangdong1_round1_train1_20190818/defect_Images/"+_iter['name'])
            color=(0,255,0)
            for i in range(len(idx)):

                x1, y1, x2, y2 = _iter['bbox'][idx[i]]
                x1, y1, x2, y2=int(round(x1)), int(round(y1)),int(round(x2)),int(round(y2))
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 1)
            if not os.path.exists('./defect_visual/'+str(defect_label[key])+'/'+key):
                os.mkdir('./defect_visual/'+str(defect_label[key])+'/'+key)    
            cv2.imwrite('./defect_visual/'+str(defect_label[key])+'/'+key+'/'+_iter['name'],img)'''
            count[defect_label[key]-1]=count[defect_label[key]-1]+1
            #cv2.imshow(_iter['name'],img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

print(count)    
'''
for _iter in load_dict:
    for i in range(len(defect_label)):
        if _iter['defect_name'] ==defect_label[i]:
            print(_iter['defect_name'])
            #defect_name.append(_iter['defect_name'])
    break'''
'''
defect_name_round1 = []
for line in open("defect_name.txt","r"): #read defect name
    defect_name_round1.append(line[:-1].split(',')[0])

#count=np.zeros((28,1))
count1=0
for line in tqdm(open("defect_box.txt","r")): #read defect bbox
    count1=count1+1
    if count1<1300:
        continue   
    img=cv2.imread('./defect_Images/'+line.split(' ')[0])
    bbox_list=line.split(' ')[1:]
    bbox=[]
    for i in range(len(bbox_list)):
        #count[int(bbox_list[i].split(',')[4]),0]=count[int(bbox_list[i].split(',')[4]),0]+1
        bbox.append(bbox_list[i].split(','))
    bbox=np.array(bbox,dtype=np.float32)
    ShowBBox(img,bbox)'''
    #break










    
'''
print(count)
print(np.sum(count))
fp = open('defect_name.txt','w')
for i in range(len(defect_name_round1)):
    fp.write(str(defect_name_round1[i])+','+str(count[i][0])+','+str(np.round(count[i][0]/np.sum(count),6)))
    fp.write('\n')'''