import cv2
from PIL import Image
import numpy as np
from scipy import ndimage
import os
import skimage.transform as transform

class FACE_ALIGMENT(object):

    def __init__(self):
        with open('eye_xy.txt') as f:
            self.content = f.readlines()
  
    def Geometric_Normalization(self,img,index):

        eye_x = []
        eye_y = []
        xy = self.content[index]
        eye_x.append(int(xy.split(' ')[0]))
        eye_x.append(int(xy.split(' ')[2]))
        eye_y.append(int(xy.split(' ')[1]))
        eye_y.append(int(xy.split(' ')[3]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate straight line distance and tilt angle
        p1 = np.array([eye_x[0],eye_y[0]])# Left eye coordinates
        p2 = np.array([eye_x[1],eye_y[1]])# Right eye coordinates
        dist = np.sqrt(np.sum(p1-p2)**2) # The distance between the two eyes
        dp = p1 - p2
        angle = np.arctan(dp[1] / dp[0])
        # Rotate images
        rot_img = ndimage.rotate(gray, angle=+angle*180/np.pi) # Translate to angle
        # Determine if rotation is performed
        if abs(angle) > 0.15:
            rot_image_center = np.array((np.array(rot_img.shape[:2]) - 1) / 2, dtype=np.int) # Find the center of the rotated image
            # Find the coordinates of the human eye after rotation
            org_eye_center = np.array((p1 + p2) / 2, dtype=np.int)
            # Midpoint of the original image
            org_image_center = np.array((np.array(img.shape[:2]) - 1) / 2, dtype=np.int)
            # Rotate the image by its center and find the midpoint of the eye in the rotated image
            R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            rot_eye_center = np.dot(R, org_eye_center[::-1]
                                    -org_image_center[::-1])[::-1] + rot_image_center
            rot_eye_center = np.array(rot_eye_center, dtype=int)
            # Find the width and height of the box containing the face based on the eye coordinates
            mid_y, mid_x = rot_eye_center
            
            y_top = int(max(mid_y - 2*dist, 0))
            y_bot = int(min(mid_y + 2 * dist, rot_img.shape[0]))
            x_left = int(max(mid_x - 2*dist, 0))
            x_right = int(min(mid_x + 2*dist, rot_img.shape[1]))
            cropped_img = rot_img[y_top:y_bot+1, x_left:x_right+1]
        else:
            cropped_img = gray
            angle = 0


        cropped_img = transform.resize(cropped_img, [100, 100], mode='constant')
        # cv2.destroyAllWindows()
        # cv2.imshow('aaa',cropped_img)
        # cv2.waitKey(1)
        
        return angle,cropped_img

    # 直接利用opencv找到眼睛的位置
    # eye_cascade =cv2.CascadeClassifier(r"haarcascade_eye.xml")
    # eyes=eye_cascade.detectMultiScale(gray)#返回的是多个对象的坐标列表
    # i = 0
    # eye_x = []
    # eye_y = []
    # for (x_eye,y_eye,w_eye,h_eye) in eyes:
    #     if i == 0:
    #         i += 1
    #         continue
    #     cv2.rectangle(img, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (128, 128, 0), 2)
    #     eye_x.append(x_eye + 1/2 * w_eye)
    #     eye_y.append(y_eye + 1/2 * h_eye)


    # cv2.imshow('aaa',img)
    # cv2.waitKey(1)
    #图片校正
    
    # if len(eye_x)!=2:
    #     continue
    
    
