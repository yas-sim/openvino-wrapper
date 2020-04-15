import iewrap

import cv2
import numpy as np

label = open('voc_labels.txt').readlines()
img = cv2.imread('car_1.bmp')

ie = iewrap.ieWrapper('public/mobilenet-ssd/FP16/mobilenet-ssd.xml', 'CPU')

output = ie.blockInfer(img)[0]     # Inference

# Draw bounding boxes and labels onto the image
output = output.reshape((100,7))
img_h, img_w, _ = img.shape
for obj in output:
    imgid, clsid, confidence, x1, y1, x2, y2 = obj
    if confidence>0.8:              # Draw a bounding box and label when confidence>0.8
        x1 = int(x1 * img_w)
        y1 = int(y1 * img_h)
        x2 = int(x2 * img_w)
        y2 = int(y2 * img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), thickness=4 )
        cv2.putText(img, label[int(clsid)][:-1], (x1, y1), cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=(0,255,255), thickness=4)

# Displaying the result image
import matplotlib.pyplot as plt
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()