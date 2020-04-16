import math

import iewrap
import numpy as np
import cv2

# Open a USB webcam
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH , 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ie_faceDet  = iewrap.ieWrapper('./intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml',             'CPU')
ie_headPose = iewrap.ieWrapper('./intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml', 'CPU')
ie_faceLM   = iewrap.ieWrapper('./intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml',   'CPU')
ie_gaze     = iewrap.ieWrapper('./intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml',           'CPU')

print('hit ESC to exit.')
while cv2.waitKey(1) != 27:         # 27 == ESC
    ret,img = cam.read()
    img = cv2.flip(img, 1)          # flip image

    # Detect faces in the image
    det = ie_faceDet.blockInfer(img).reshape((200,7))   # [1,1,200,7]
    for obj in det:                                     # obj = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
        if obj[2] > 0.75:                               # Confidence > 75% 
            xmin = abs(int(obj[3] * img.shape[1]))
            ymin = abs(int(obj[4] * img.shape[0]))
            xmax = abs(int(obj[5] * img.shape[1]))
            ymax = abs(int(obj[6] * img.shape[0]))
            face = img[ymin:ymax,xmin:xmax].copy()      # Crop the found face
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,255,0), 2)

            # Find facial landmarks (to find eyes)
            landmark = ie_faceLM.blockInfer(face).reshape((70,)) # [1,70]
            lm=landmark[:4*2].reshape(4,2)  #  [[left0x, left0y], [left1x, left1y], [right0x, right0y], [right1x, right1y] ]

            # Estimate head rotation angles (yaw, pitch, role)
            hp = ie_headPose.blockInfer(face)
            yaw   = hp['angle_y_fc'][0][0]  # { 'angle_y_fc': array([[9.3833]], dtype=float32), }
            pitch = hp['angle_p_fc'][0][0]
            roll  = hp['angle_r_fc'][0][0]

            _X=0
            _Y=1
            # eye size in the cropped face image
            eye_size   = [ abs(int((lm[0][_X]-lm[1][_X]) * face.shape[1])), abs(int((lm[3][_X]-lm[2][_X]) * face.shape[1])) ]
            # eye center coordinate in the cropped face image
            eye_center = [ [ int(((lm[0][_X]+lm[1][_X])/2 * face.shape[1])), int(((lm[0][_Y]+lm[1][_Y])/2 * face.shape[0])) ], 
                            [ int(((lm[3][_X]+lm[2][_X])/2 * face.shape[1])), int(((lm[3][_Y]+lm[2][_Y])/2 * face.shape[0])) ] ]
            if eye_size[0]<4 or eye_size[1]<4: continue    # Skip if eyes are too small

            ratio = 0.7              # Crop eye size ratio to the calculated eye size
            eyes = []
            for i in range(2):       # 2 = left and right eyes
                # Crop eye images
                x1 = int(eye_center[i][_X]-eye_size[i]*ratio)
                x2 = int(eye_center[i][_X]+eye_size[i]*ratio)
                y1 = int(eye_center[i][_Y]-eye_size[i]*ratio)
                y2 = int(eye_center[i][_Y]+eye_size[i]*ratio)
                eyes.append(face[y1:y2, x1:x2].copy())    # Crop an eye

                # Draw eye boundary boxes
                x=eye_center[i][_X]
                y=eye_center[i][_Y]
                s=int(eye_size[i]*ratio)
                cv2.rectangle(img, (x-s+xmin, y-s+ymin), (x+s+xmin, y+s+ymin), (0,255,0), 2)

                # rotate eyes around Z axis to keep them level
                if roll != 0.:
                    rotMat = cv2.getRotationMatrix2D((eyes[i].shape[0]//2, eyes[i].shape[1]//2), roll, 1.0)
                    eyes[i] = cv2.warpAffine(eyes[i], rotMat, (eyes[i].shape[0], eyes[i].shape[1]), flags=cv2.INTER_LINEAR)

            # Estimate gaze - Gaze estimation model has multiple inputs, so get input blob information with `getInputs()` and set input data
            gaze_in = ie_gaze.getInputs()
            gaze_in['head_pose_angles']['data']=[yaw, pitch, 0] # head_pose_angles (non-image)
            gaze_in['head_pose_angles']['type']='vec'           # head pose angles are not an image
            gaze_in['left_eye_image'  ]['data']=eyes[0]         # left_eye_image
            gaze_in['right_eye_image' ]['data']=eyes[1]         # right_eye_image
            gaze_vec = ie_gaze.blockInfer(gaze_in)[0]
            gaze_vec_norm = gaze_vec / np.linalg.norm(gaze_vec)  # normalize vector

            # Rotate gaze vector by face roll angle
            vcos = math.cos(math.radians(roll))
            vsin = math.sin(math.radians(roll))
            tmpx = gaze_vec_norm[0]*vcos + gaze_vec_norm[1]*vsin
            tmpy = gaze_vec_norm[0]*vsin + gaze_vec_norm[1]*vcos
            gaze_vec_norm = [tmpx, tmpy]

            # draw gaze line
            line_length = 1000
            for i in range(2):
                coord1 = (eye_center[i][_X]+xmin, eye_center[i][_Y]+ymin)
                coord2 = (eye_center[i][_X]+xmin+int((gaze_vec_norm[0]+0.)*line_length), 
                          eye_center[i][_Y]+ymin-int((gaze_vec_norm[1]+0.)*line_length))
                cv2.line(img, coord1, coord2, (0, 0, 255),2)
            cv2.imshow("gaze", img)
cv2.destroyAllWindows()