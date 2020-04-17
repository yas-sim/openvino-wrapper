import iewrap

import numpy as np
import cv2

#----------------------------------------------------------------------------------------
# Heatmap class

class heatmap:
    colorTable = ( (  0, (  0,  0,  0)),
                   (  1, (128,  0,  0)),
                   ( 32, (255,  0,  0)),
                   ( 64, (255,  0,255)),
                   ( 96, (  0,255,  0)),
                   (128, (  0,255,  0)),
                   (160, (  0,255,255)),
                   (192, (  0,255,255)),
                   (224, (  0,  0,255)),
                   (256, (  0,  0,255)) )

    def __init__(self, nx, ny, nn):
        self.num_x = nx
        self.num_y = ny
        self.num_n = nn
        self.current = 0        # current level
        self.heatmap = np.zeros((nx, ny, nn), dtype=np.uint8)    # heatmap (has multiple levels)
        self.frame   = np.zeros((ny, nx,  3), dtype=np.uint8)    # frame image genarated from the heatmap
        self.colorLUT= [ self.colorInterpolate(v) for v in range(256) ]
      
    def colorInterpolate(self, col):
        if col<  0: col=  0
        if col>255: col=255
        prevCtbl=(0,(0,0,0))
        for i, ctbl in enumerate(self.colorTable):
            if col<ctbl[0]:
                v1, col1 = prevCtbl
                v2, col2 = ctbl
                p    = 0 if col==v1 else (col-v1)/(v2-v1)
                b    = int((col2[0]-col1[0])*p+col1[0])
                r    = int((col2[1]-col1[1])*p+col1[1])
                g    = int((col2[2]-col1[2])*p+col1[2])
                prevCtbl = ctbl
                return (b,r,g)
        return (0,0,0)

    def clearHeatmapLevel(self, level):
        if level<0 or level>=self.num_n:
            return
        for x in range(self.num_x):
            for y in range(self.num_y):
                self.heatmap[x, y, level] = 0

    def incrementTime(self):
        self.current = (self.current+1) % self.num_n
        self.clearHeatmapLevel(self.current)               # clear the new level

    def generateFrame(self):
        valMap = np.sum(self.heatmap, axis=2)
        for y in range(self.num_y):
            for x in range(self.num_x): 
                val = valMap[x, y]
                val = 0 if val< 0 else 255 if val>255 else val
                col = self.colorLUT[val]
                self.frame[y, x, 0] = col[0]
                self.frame[y, x, 1] = col[1]
                self.frame[y, x, 2] = col[2]

    def addValue(self, x, y, val):
        self.heatmap[x, y, self.current] += val
        if self.heatmap[x, y, self.current] > 255:
            self.heatmap[x, y, self.current] = 255

#----------------------------------------------------------------------------------------

# Person detection & re-identification
model_det  = 'intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.xml'

# Face detection & re-identification
#model_det  = 'intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml'


def main():
    ie_detect = iewrap.ieWrapper(model_det,  'CPU')

    # Open USB webcams
    cam = cv2.VideoCapture(0)
    #cam = cv2.VideoCapture(input_movie)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH , 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hm = heatmap(10, 10, 30)

    n = 0
    while(cv2.waitKey(1)!=27):
        ret, img = cam.read()
        if ret==False:
            break

        img_out = img.copy()
        det = ie_detect.blockInfer(img).reshape((200,7))      # Detect objects

        img_out = img_out>>1
        for obj in det:         # obj = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
            if obj[2] > 0.75:   # Confidence > 75% 
                xmin = abs(int(obj[3] * img_out.shape[1]))
                ymin = abs(int(obj[4] * img_out.shape[0]))
                xmax = abs(int(obj[5] * img_out.shape[1]))
                ymax = abs(int(obj[6] * img_out.shape[0]))
                cv2.rectangle(img_out, (xmin, ymin), (xmax, ymax), (  0,255,255), 2)
                x = int((obj[3]+obj[5])/2*hm.num_x)
                y = int((obj[4]+obj[6])/2*hm.num_y)
                hm.addValue(x, y, 4)

        hm.generateFrame()
        n = (n+1) % 30
        if n==0:
            hm.incrementTime()
        frame = cv2.resize(hm.frame, dsize=(img_out.shape[1], img_out.shape[0]), 
                        interpolation = cv2.INTER_CUBIC)   # INTER_AREA, INTER_LINEAR, INTER_CUBIC
        frame = frame | img_out
        cv2.imshow('output', frame)

if __name__ == '__main__':
    main()
