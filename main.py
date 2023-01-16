from face_detection.gen_haar_feature import *
from face_align import *
from face_recognition.PCA import *

from tqdm import tqdm
from PIL import Image
from pylab import  *
import cv2
import xlrd
import pylab
import matplotlib.pyplot as plt
import pandas as pd
import os 

global dist
global integimg
global varNormFactor

def Get_Classifier():

    print('Get 12 Strong Classifier...')
    data = pd.read_excel('Strong Classifier.xlsx',engine='openpyxl')
    table = data
    Split_Value = []
    Alpha = []
    Type = []
    for i in range(12):
        Split_Value.append(table.loc[4 * i + 0])
        Alpha.append(table.loc[4 * i + 1])
        Type.append(table.loc[4 * i + 2])

    return Split_Value,Alpha,Type

class Face_Detect_Processor(object):

    def __init__(self,Split_Value,Alpha,Type,useful_feature_index):

        self.Split_Value = Split_Value
        self.Alpha = Alpha
        self.Type = Type
        self.Useful_Feature_Index = useful_feature_index
    
    def Detect_Face(self,feature,split_value,alpha,type):
        # Judgment on each weak classifier
        result = 0
        for i in range(len(split_value)):
            if type[i] == 'up':
                if feature >= split_value[i]:
                    result += 1*alpha[i]
                else:
                    result -= 1*alpha[i]
            elif type[i] == 'down':
                if feature >= split_value[i]:
                    result -= 1*alpha[i]
                else:
                    result += 1*alpha[i]

        return result

    def draw_rectangle(self,image, rect):
        # Drawing rectangular boxes 
        (x, y, xw, yh) = rect
        cv2.rectangle(image, (int(x), int(y)), (int(xw), int(yh)), (128, 128, 0), 2)

    def Cluster(self,X,Y):
        center_x = []
        center_y = []
        for k in range(int(len(X)/2)):
            # Get the center coordinates of each map
            center_x.append(int(X[2*k] + X[2*k+1])/2)
            center_y.append(int(Y[2*k] + Y[2*k+1])/2)
        distance = np.ones((len(center_x),len(center_x)))
        # A distance of 1 means it can be merged
        for i in range(len(center_x)):
            for j in range(len(center_x)):
                distance[i][j] = sqrt((center_x[i]- center_x[j])**2 + (center_y[i]- center_y[j])**2 )
                if distance[i][j] < 200:
                    distance[i][j] = 1
                else:
                    distance[i][j] = 0
        
        cluster_number = np.sum(distance, axis = 1)
        cluster_index_center = np.argsort(-cluster_number)[0] # Find the center of the cluster, the window with the most neighboring distances
        cluster_x_min = []
        cluster_x_max = []
        cluster_y_min = []
        cluster_y_max = []
        for m in range(len(center_x)):
            if distance[cluster_index_center][m] == 1:
                cluster_x_min.append(X[2*m])
                cluster_x_max.append(X[2*m + 1])
                cluster_y_min.append(Y[2*m])
                cluster_y_max.append(Y[2*m + 1])

        x_min = 0.2*min(cluster_x_min)+0.8*max(cluster_x_min)
        x_max = 0.5*max(cluster_x_max)+0.5*min(cluster_x_max)
        y_min = 0.2*min(cluster_y_min)+0.8*max(cluster_y_min)
        y_max = 0.8*max(cluster_y_max)+0.2*min(cluster_y_max)

        return x_min,y_min,x_max,y_max

    def Slide_Window(self,img,origial_window_width,origial_window_height,scaling_w_width,scaling_w_height,scaling_times,stride):
        '''
        Set the parameters of the sliding window
        img: Color images read by PIL
        origial_window_width,origial_window_height: Initial window width, height
        scaling_w_width,scaling_w_height: Window width, height per zoom
        scaling_times: Number of scaling
        stride: Window step per movement
        '''
        window_x = []
        window_y = []
        for i in tqdm(range(scaling_times)):
            window_width = origial_window_width + scaling_w_width * i 
            window_height = origial_window_height + scaling_w_height * i 
            # Get the starting coordinates of the image
            y = 0
            x = 0
            for row in range(int((size(img,0)-window_height+1)/stride)+1):
                y = stride*row
                for col in range(int((size(img,1)-window_width+1)/stride+1)):
                    x = stride*col
                    # Get window image
                    pic = img.crop((x,y,x+window_width,y+window_height))
                    pic = pic.resize((24,24))
                    # Get grayscale map and enhance contrast
                    gray = pic.convert('L')
                    gray = array(gray)
                    dist = histeq(gray)
                    integimg = integral(dist)

                    """Find the haar characteristics"""
                    # Find the normalization factor
                    imgsum = integimg[-1][-1]
                    sqsum = sum(np.dot(np.array(dist),np.array(dist)))
                    wh = 24 * 24
                    mean = imgsum/wh
                    sqmean = sqsum/wh
                    varNormFactor = math.sqrt(abs(mean*mean - sqmean))
                    # Get haar characteristics
                    # parameters: varNormFactor,dist, integimg,haarblock_width, haarblock_height, Scale_num
                    # Scale_num = window_size/haarblock_size，eg:4 = 24 / 6
                    haar_feature = harr(varNormFactor,dist, integimg, 6, 6, 4)
                    # Filter to get the useful 12 haar features by useful_feature_index
                    # Each feature corresponds to a strong classifier
                    for j in range(len(self.Useful_Feature_Index)):
                        # Judgment on each of the 12 features
                        useful_feature = []
                        useful_feature = haar_feature[useful_feature_index[j]]
                        result = self.Detect_Face(useful_feature,self.Split_Value[j],self.Alpha[j],self.Type[j])
                        # The result is -1, which means it is not a human face, and the judgment will not continue.
                        if result < 0:
                            break
                    # Cascade the classifiers to successively satisfy all strong classifiers for the face
                    # print(len(self.Useful_Feature_Index)-1)
                    if j == (len(self.Useful_Feature_Index)-1):
                        
                        rect = [x,y,x+window_width,y+window_height]
                        window_x.append(x)
                        window_x.append(x + window_width)
                        window_y.append(y)
                        window_y.append(y + window_height)

        cv2.destroyAllWindows()
        if len(window_x)>1:
            (x,y,xw,yh) = self.Cluster(window_x,window_y)
            rect = [x, y, xw, yh]
            self.draw_rectangle(test_img1,rect)
        elif len(window_x)==1:
            self.draw_rectangle(test_img1,rect)
        
        return rect


if __name__ == '__main__':

    """Obtain 12 classifiers for detecting faces"""
    Split_Value,Alpha,Type = Get_Classifier()
    # Instantiated face detection processor
    # This index value is printed by  /face_detection/Adaboost.py
    useful_feature_index = [781, 776, 686, 2812, 2842, 681, 2807, 791, 2282, 2847, 2802, 826]
    face_detect_processor = Face_Detect_Processor(Split_Value,Alpha,Type,useful_feature_index)#实例化个体
    print('Adaboost Classifier Ready')

    """Get the PCA features of the training set"""
    label_name = ['bihan','caoping','danshu','fanzhongyan','hanqi','huaiji','huirou','qiuhe','xinhe','zhaozhen']
    pca_processor = PCA_Processor(label_name)
    # take the first 5 feature vectors
    pca_processor.train_recognize(5)

    """Testing the effect of the algorithm"""
    img_folder = "test_image"
    img_name = os.listdir(img_folder)
    # read iamges
    print("Read Test Image...")
    correct_num = 0
    for num in tqdm(range(len(img_name))):

        image_path = img_folder + '/' + str(img_name[num])
        # Avoid different image formats
        img = Image.open(image_path)
        test_img1 = cv2.imread(image_path)

        # Find the face rectangle for each image
        # parameters: image; initial window_width; nitial window_height; window_scaling_width; window_scaling_height; scaling times；moving step of window
        rect = face_detect_processor.Slide_Window(img,400,400,50,50,5,20)
        #process_image = img.crop((rect[0],rect[1],rect[2]-rect[0],rect[3]-rect[1]))
        y_top = int(max(rect[1], 0))
        y_bot = int(min(rect[3], test_img1.shape[0]))
        x_left = int(max(rect[0], 0))
        x_right = int(min(rect[2], test_img1.shape[1]))
        test_img2 = test_img1
        test_img2 = cv2.cvtColor(test_img2, cv2.COLOR_BGR2GRAY)
        process_image = test_img2[y_top:y_bot+1,x_left:x_right+1]
        test_label = pca_processor.test_recognize(process_image,img_name[num])
        test_label1 = 'result:'+ label_name[test_label]+'|'+'label_result:'+img_name[num].split('_')[0]

        cv2.putText(test_img1,test_label1,(int(rect[0]),int(rect[1]-5)),cv2.FONT_HERSHEY_COMPLEX, 0.6, (128, 128, 0), 1)
        cv2.imshow(test_label1,test_img1)
        cv2.waitKey(10)
        print(label_name[test_label])
        print(img_name[num].split('_')[0])
        if label_name[test_label] == img_name[num].split('_')[0]:
            print(1)
            correct_num += 1
    accuracy = float(correct_num)/len(img_name)   
    print ('correct_num: %.2f%%'%(accuracy * 100))