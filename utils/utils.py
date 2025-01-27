import numpy as np
from tqdm import tqdm
import json
import cv2
from PIL import Image, ImageDraw, ImageFont 
import colorsys
def ShowBBox(img,bbox):

    #bbox n*5
    colors=GetColor()
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
    for i in range(bbox.shape[0]):
        font = ImageFont.truetype(font='./font/FiraMono-Medium.otf',
                        size=np.floor(3e-2 * img_PIL.size[1] + 0.5).astype('int32'))
        label = '{} {:.2f}'.format(int(bbox[i][4]), 1.0)
        draw = ImageDraw.Draw(img_PIL)  
        draw.text((int(bbox[i][2]),int(bbox[i][1])), label, font=font, fill=(255,0,0)) 
     
    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR) 
    for i in range(bbox.shape[0]):
        cv2.rectangle(img_OpenCV, (int(bbox[i][0]),int(bbox[i][1])), (int(bbox[i][2]),int(bbox[i][3])), colors[int(bbox[i][4])],2)

    #cv2.namedWindow("enhanced",0);
    #cv2.resizeWindow("enhanced", 1200,600);
    cv2.imshow('enhanced',img_OpenCV)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def GetColor(class_num=28):
    hsv_tuples = [(x / class_num, 1., 1.)
                      for x in range(class_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    return colors
