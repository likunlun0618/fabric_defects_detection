import numpy as np
from tqdm import tqdm
import json
import cv2
from utils.utils import *
with open("./Annotations/gt_result.json",'r') as load_f:
    load_dict = json.load(load_f)
    #print(load_dict)

defect_name_round1 = []
for line in open("defect_name.txt","r"): #read defect name
    defect_name_round1.append(line[:-1].split(',')[0])

count=np.zeros((28,1))
for line in tqdm(open("defect_box.txt","r")): #read defect bbox
    img=cv2.imread('./defect_Images/'+line.split(' ')[0])
    bbox_list=line.split(' ')[1:]
    bbox=[]
    for i in range(len(bbox_list)):
        count[int(bbox_list[i].split(',')[4]),0]=count[int(bbox_list[i].split(',')[4]),0]+1
        bbox.append(bbox_list[i].split(','))
    bbox=np.array(bbox,dtype=np.float32)
    ShowBBox(img,bbox)
    #break










    
'''
print(count)
print(np.sum(count))
fp = open('defect_name.txt','w')
for i in range(len(defect_name_round1)):
    fp.write(str(defect_name_round1[i])+','+str(count[i][0])+','+str(np.round(count[i][0]/np.sum(count),6)))
    fp.write('\n')'''