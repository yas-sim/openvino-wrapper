import iewrap

import cv2
import numpy as np

label = open('synset_words.txt').readlines()
img = cv2.imread('car.png')

ie = iewrap.ieWrapper('public/googlenet-v1/FP16/googlenet-v1.xml', 'CPU', 4)

output = ie.blockInfer(img).reshape((1000,))  # Inferencing

# Sort class probabilities and display top 5 classes
idx = np.argsort(output)[::-1]
for i in range(5):
    print(idx[i]+1, output[idx[i]], label[idx[i]][:-1])

# Displaying the input image
import matplotlib.pyplot as plt
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()