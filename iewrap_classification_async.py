import iewrap

import cv2
import numpy as np

label = open('synset_words.txt').readlines()
img = cv2.imread('car.png')

ie = iewrap.ieWrapper('public/googlenet-v1/FP16/googlenet-v1.xml', 'CPU', 4)

# Callback function - Displaying the inference result
def callback(infer_id, output):
    output=output[0]
    idx = np.argsort(output)[::-1]
    print('InferResult ', infer_id)
    for i in range(1):
        print(idx[i]+1, output[idx[i]], label[idx[i]][:-1])
    print('')

# Set callback function to be called
ie.setCallback(callback)

# Infer the same image for 5 times
for i in range(10):
    infID = ie.asyncInfer(img)         # Start an asynchronous inference
    print('InferRequest ', infID)

# Wait for all asynchronous inference to complete 
ie.waitForAllCompletion()

# Displaying the input image
import matplotlib.pyplot as plt
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()