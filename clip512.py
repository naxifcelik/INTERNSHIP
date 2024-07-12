from PIL import Image
import numpy as np
import cv2
import os

path = "your_data_path"
mask_path = "your_targetdata_path
threshold = 0.75


imgList = os.listdir(mask_path)

for image in imgList:
    source = path + image
    mask = mask_path + image
    im = Image.open(mask)
    girdi = np.asarray(im)
    values = girdi.tolist()
    toplam = 0
    back = 0

    for i in values:
        toplam = toplam + i.count(1)
 

    oran = toplam / (512 * 512) #resolution

    print('Sinif 1:', oran)  

    if oran > threshold:
        os.remove(source)
        os.remove(mask)

