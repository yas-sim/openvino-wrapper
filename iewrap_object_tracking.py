import iewrap

import sys
import math
import time

import numpy as np
from numpy import linalg as LA

import cv2
from scipy.spatial import distance
from munkres import Munkres               # Hungarian algorithm for ID assignment

# ---------------------------------------------
# Functions for checking boundary line crossing

def line(p1, p2):
  A = (p1[1] - p2[1])
  B = (p2[0] - p1[0])
  C = (p1[0]*p2[1] - p2[0]*p1[1])
  return A, B, -C

# Calcuate the coordination of intersect point of line segments - 線分同士が交差する座標を計算
def calcIntersectPoint(line1p1, line1p2, line2p1, line2p2):
  L1 = line(line1p1, line1p2)
  L2 = line(line2p1, line2p2)
  D  = L1[0] * L2[1] - L1[1] * L2[0]
  Dx = L1[2] * L2[1] - L1[1] * L2[2]
  Dy = L1[0] * L2[2] - L1[2] * L2[0]
  x = Dx / D
  y = Dy / D
  return x,y

# Check if line segments intersect - 線分同士が交差するかどうかチェック
def checkIntersect(p1, p2, p3, p4):
  tc1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
  tc2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
  td1 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
  td2 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
  return tc1*tc2<0 and td1*td2<0

# line(point1)-(point2)
# convert a line to a vector
def line_vectorize(point1, point2):
  a = point2[0]-point1[0]
  b = point2[1]-point1[1]
  return [a,b]

# point = (x,y)
# line1(point1)-(point2), line2(point3)-(point4)
# Calculate the angle made by two line segments - 線分同士が交差する角度を計算
def calc_vector_angle( point1, point2, point3, point4 ):
  u = np.array(line_vectorize(point1, point2))
  v = np.array(line_vectorize(point3, point4))
  i = np.inner(u, v)
  n = LA.norm(u) * LA.norm(v)
  c = i / n
  a = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
  if u[0]*v[1]-u[1]*v[0]<0:
    return a
  else:
    return 360-a


#------------------------------------


# Person detection & re-identification
model_det  = 'intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.xml'
model_reid = 'intel/person-reidentification-retail-0031/FP16/person-reidentification-retail-0031.xml'

# Face detection & re-identification
#model_det  = 'intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml'
#model_reid = 'intel/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml'

boundaryLine = [ (300, 40), (20, 400) ]  # boundary line
crossCount  = [ 0, 0 ]

def main():
    ie_detect = iewrap.ieWrapper(model_det,  'CPU')
    ie_reid   = iewrap.ieWrapper(model_reid, 'CPU')

    # Open USB webcams
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("../sample-videos/people-detection.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH , 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   

    objid = 0
    time_out = 3                        # how long time to retain feature vector (second()
    feature_db = []

    while cv2.waitKey(1)!=27:           # 27 == ESC
        ret, image = cap.read()
        if ret==False:
            break
        #image = cv2.flip(image, 1)

        detObj = ie_detect.blockInfer(image).reshape((200,7))     # [1,1,200,7]
        curr_feature = []
        for obj in detObj:                # obj = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
            if obj[2] > 0.75:             # Confidence > 75% 
                xmin = abs(int(obj[3] * image.shape[1]))
                ymin = abs(int(obj[4] * image.shape[0]))
                xmax = abs(int(obj[5] * image.shape[1]))
                ymax = abs(int(obj[6] * image.shape[0]))
                class_id = int(obj[1])

                obj_img=image[ymin:ymax,xmin:xmax].copy()             # Crop the found object

                # Obtain feature vector of the detected object using re-identification model
                featVec = ie_reid.blockInfer(obj_img).reshape((256)) # Run re-identification model to generate feature vectors (256 elem)
                curr_feature.append({'pos': [xmin,ymin, xmax,ymax], 'feature': featVec, 'id': -1 })

        # discard time out objects
        now = time.monotonic()
        for feature in feature_db:
            if feature['time'] + time_out < now:
                feature_db.remove(feature)     # discard feature vector from DB
                #print("Discarded  : id {}".format(feature['id']))

        # Draw boundary line
        outimg = image.copy()
        cv2.line(outimg, boundaryLine[0], boundaryLine[1], (0,255,255), 8)
        cv2.putText(outimg, str(crossCount[0]), boundaryLine[0], cv2.FONT_HERSHEY_PLAIN, 4, (0,255,255), 2)
        cv2.putText(outimg, str(crossCount[1]), boundaryLine[1], cv2.FONT_HERSHEY_PLAIN, 4, (0,255,255), 2)

        # if no object found, skip the rest of processing
        if len(curr_feature) == 0:             # total 0 faces found
            cv2.imshow('image', outimg)
            continue

        # If any object is registred in the db, assign registerd ID to the most similar object in the current image
        if len(feature_db)>0:
            # Create a matix of cosine distance
            cos_sim_matrix=[ [ distance.cosine(curr_feature[j]["feature"], feature_db[i]["feature"]) 
                            for j in range(len(curr_feature))] for i in range(len(feature_db)) ]
            # solve feature matching problem by Hungarian assignment algorithm
            hangarian = Munkres()
            combination = hangarian.compute(cos_sim_matrix)

            # assign ID to the object pairs based on assignment matrix
            for dbIdx, currIdx in combination:
                curr_feature[currIdx]['id'] = feature_db[dbIdx]['id']            # assign an ID
                feature_db[dbIdx]['feature'] = curr_feature[currIdx]['feature']  # update the feature vector in DB with the latest vector
                feature_db[dbIdx]['time'] = now                                  # update last found time
                xmin, ymin, xmax, ymax = curr_feature[currIdx]['pos']
                feature_db[dbIdx]['history'].append([(xmin+xmax)//2, (ymin+ymax)//2])   # position history for trajectory line
                curr_feature[currIdx]['history'] = feature_db[dbIdx]['history']

        # Register the new objects which has no ID yet
        for feature in curr_feature:
            if feature['id']==-1:           # no similar objects is registred in feature_db
                feature['id'] = objid
                feature_db.append(feature)  # register a new feature to the db
                feature_db[-1]['time']    = now
                xmin, ymin, xmax, ymax = feature['pos']
                feature_db[-1]['history'] = [[(xmin+xmax)//2, (ymin+ymax)//2]]  # position history for trajectory line
                feature['history'] = feature_db[-1]['history']
                objid+=1

        # Draw bounding boxes and IDs
        for obj in curr_feature:
            id = obj['id']
            color = ( (((~id)<<6) & 0x100)-1, (((~id)<<7) & 0x0100)-1, (((~id)<<8) & 0x0100)-1 )
            xmin, ymin, xmax, ymax = obj['pos']
            cv2.rectangle(outimg, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(outimg, 'ID='+str(id), (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 1)
            hist = obj['history']
            if len(hist)>1:
                cv2.polylines(outimg, np.array([hist], np.int32), False, color,4 )

            # Detect boundary line crossing
            if len(hist)>1:
                traj_t0 = (hist[-2][0], hist[-2][1])    # Trajectory of an object
                traj_t1 = (hist[-1][0], hist[-1][1])
                bLine_0 = boundaryLine[0]               # Boundary line
                bLine_1 = boundaryLine[1]
                intersect = checkIntersect(traj_t0, traj_t1, bLine_0, bLine_1)      # Check if intersect or not
                if intersect == True:
                    angle = calc_vector_angle(traj_t0, traj_t1, bLine_0, bLine_1)   # Calculate angle between trajectory and boundary line
                    if angle<180:
                        crossCount[0] += 1
                    else:
                        crossCount[1] += 1
                    cx, cy = calcIntersectPoint(traj_t0, traj_t1, bLine_0, bLine_1) # Calculate the intersect coordination
                    cv2.circle(outimg, (int(cx), int(cy)), 20, (255,0, 255 if angle<180 else 0), -1)

        cv2.imshow('image', outimg)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)
