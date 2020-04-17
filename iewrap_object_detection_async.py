import iewrap

import time

import cv2
import numpy as np

imgBuf = {}
label  = []

def callback(infId, output):
    global imgBuf, label

    # Draw bounding boxes and labels onto the image
    output = output.reshape((100,7))
    img = imgBuf.pop(infId)
    img_h, img_w, _ = img.shape
    for obj in output:
        imgid, clsid, confidence, x1, y1, x2, y2 = obj
        if confidence>0.6:              # Draw a bounding box and label when confidence>0.8
            x1 = int(x1 * img_w)
            y1 = int(y1 * img_h)
            x2 = int(x2 * img_w)
            y2 = int(y2 * img_h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), thickness=4 )
            cv2.putText(img, label[int(clsid)][:-1], (x1, y1), cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=(0,255,255), thickness=4)
    cv2.imshow('result', img)
    cv2.waitKey(1)

def main():
    global imgBuf, label
    label = open('voc_labels.txt').readlines()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH , 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    #cap = cv2.VideoCapture('../sample-videos/people-detection.mp4')

    ie = iewrap.ieWrapper('public/mobilenet-ssd/FP16/mobilenet-ssd.xml', 'CPU', 10)
    ie.setCallback(callback)

    while True:
        ret, img = cap.read()
        if ret==False:
            break
        refId = ie.asyncInfer(img)     # Inference
        imgBuf[refId]=img
        #time.sleep(1/30)

if __name__ == '__main__':
    main()