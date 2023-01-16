import cv2
import sys 
sys.path.append("..") 
from face_align import *
from numpy import *
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import operator
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage

class PCA_Processor(object):
    def __init__(self,label_name):

        self.name_index = os.listdir("face_recognition/face")
        self.label_name = ['bihan','caoping','danshu','fanzhongyan','hanqi','huaiji','huirou','qiuhe','xinhe','zhaozhen']
        
    def _init_parameters(self,train_labels,train_dataset,eigenfaces,vector):
        
        self.Train_Labels = train_labels
        self.Train_Dataset = train_dataset
        self.Eigenfaces = eigenfaces
        self.Vector = vector

    def data_prerocess(self,img):
        data = np.array(img).flatten()#将图片转为一维
        data = list(data)
        return data

    #获取PCA特征值和特征向量
    def pca(self,data,k):
        data = float32(mat(data)) #将数据转化为矩阵
        trainNumber,trainsize = data.shape#取大小，trainNumber为图片的数量，trainsize为每张图的大小
        data_mean = mean(data,0)#对列求均值
        data_mean_all = tile(data_mean,(trainNumber,1))#将每列的平均值重复row行

        Z = data - data_mean_all
        T1 = Z*Z.T #协方差矩阵，使用矩阵计算，所以前面mat

        eigenvalues,eigenvectors = linalg.eig(T1) #特征值与特征向量   
        eigenvectors = list(eigenvectors.T)#特征向量矩阵的每列是一个特征向量

        vector = []
        for i in range(k):
            vector.append(eigenvectors[i])

        vectors = np.array(vector)
        vectors = np.mat(vectors).T
        vectors = Z.T*vectors
        for i in range(k): #特征向量归一化
            L = linalg.norm(vectors[:,i])
            vectors[:,i] = vectors[:,i]/L

        data_new = Z * vectors
        return data_new,vectors

    def recognize(self,test_labels,test_dataset):

        Matrix = self.Train_Dataset
        data_new = self.Eigenfaces
        test_data = float32(mat(test_dataset)) #将数据转化为矩阵
        testNumber,testsize = test_data.shape#取测试集的大小
        trainNumber,testsize = Matrix.shape

        #计算每个向量与平均向量的差
        data = float32(mat(Matrix))
        data_mean = mean(data,0)#对列求均值

        temp_face = test_data - tile(data_mean,(testNumber,1))#将每列的平均值重复row行   
        data_test_new = temp_face*self.Vector #得到测试脸在特征向量下的数据
        data_test_new = np.array(data_test_new) # mat change to array
        data_train_new = np.array(data_new)
        
        true_num = 0
        print('Start Recognition Test...')
        for i in tqdm(range(testNumber)):
            testFace = data_test_new[i,:]
            diffMat = data_train_new - tile(testFace,(trainNumber,1))
            sqDiffMat = diffMat**2
            sqDistances = sqDiffMat.sum(axis=1)
            sortedDistIndicies = sqDistances.argsort()#找到最短的欧式距离的索引
            result_label = self.Train_Labels[sortedDistIndicies[0]]
            if result_label == test_labels[i]:
                true_num += 1
        accuracy = float(true_num)/testNumber    
        print ('The classify accuracy is: %.2f%%'%(accuracy * 100))

        #KNN

        # scaler=StandardScaler()
        # train=scaler.fit_transform(data_train_new)#标准化处理
        # test=scaler.fit_transform(data_test_new)#标准化处理

        # # train=data_train_new#标准化处理
        # # test=data_test_new#标准化处理
        # labels = np.array(self.Train_Labels)
        # tlabels = np.array(test_labels)

        # knn=KNeighborsClassifier(n_neighbors=1,weights = 'distance')
        # knn.fit(train,labels)
        # y_pred=knn.predict(test)
        # print('model accuracy:',metrics.accuracy_score(tlabels,y_pred))



        # """确定经过过滤后的图片数目"""
        # perTotal, trainNumber = np.shape(eigenface)

        # """将每个样本投影到特征空间"""
        # projectedImage = eigenface.T * diffMatrix.T

        # projectedTestImage = eigenface.T * differenceTestImage.T

        # """按照欧式距离计算最匹配的人脸"""
        # distance = []
        # for i in range(0, trainNumber):
        #     q = projectedImage[:,i]
        #     temp = np.linalg.norm(projectedTestImage - q) #计算范数
        #     distance.append(temp)
        
        # minDistance = min(distance)
        # index = distance.index(minDistance)
    def train_recognize(self,k):
        
        #处理训练数据
        train_folder = "face_recognition/PCA_train"
        train_name = os.listdir(train_folder)
        dataset = []
        train_labels = []
        print("Start Recgonition Training...")
        for num  in tqdm(range(len(train_name))):
            label = train_name[num].split('_')[0]
            train_labels.append(self.label_name.index(label))
            img_path = train_folder + "/" + str(train_name[num])
            #图片校正
            img_cv2 = cv2.imread(img_path)
            img_index = self.name_index.index(train_name[num])
            face_aligment = FACE_ALIGMENT()
            angle,gray = face_aligment.Geometric_Normalization(img_cv2,img_index)
            #cv2.imshow('aaa',gray)
            #cv2.waitKey(1)
            #不进行图片校正
            #img = Image.open(img_path)
            #gray = img.convert('L')
            data_row = self.data_prerocess(gray)#将图片处理为一行数据
            dataset.append(data_row)

        
        dataset = np.array(dataset)
        dataset = double(dataset) / 255
        self.Eigenfaces,self.Vector = self.pca(dataset,k)
        self.Train_Labels = train_labels
        self.Train_Dataset = dataset
        print("Train Done!")


    def test_recognize(self,test_img,img_name):

        orig_img = cv2.imread("face_recognition/face"+'/'+str(img_name))

        data_new = self.Eigenfaces
        Matrix = self.Train_Dataset
        #test_img = transform.resize(test_img, [100, 100], mode='constant')
        #test_img = test_img.resize((100,100))
        #图片校正
        img_index = self.name_index.index(img_name)
        face_aligment = FACE_ALIGMENT()
        #通过原始图片获得转动角度
        angle,gray = face_aligment.Geometric_Normalization(orig_img,img_index)
        #test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        gray = ndimage.rotate(test_img, angle=+angle*180/np.pi)
        
        gray = transform.resize(gray, [100, 100], mode='constant')
        #gray = test_img.convert('L')
        data_row = self.data_prerocess(gray)#将图片处理为一行数据
        data_row = np.array(data_row)
        test_dataset = double(data_row) / 255 #光照补偿

        test_data = float32(mat(test_dataset)) #将数据转化为矩阵
        test_number,testsize = test_data.shape #取测试集的大小
        trainNumber,testsize = Matrix.shape

        #计算每个向量与平均向量的差
        data = float32(mat(Matrix))
        data_mean = mean(data,0)#对列求均值

        temp_face = test_data - data_mean
        data_test_new = temp_face*self.Vector #得到测试脸在特征向量下的数据
        data_test_new = np.array(data_test_new) # mat change to array
        data_train_new = np.array(data_new)
        
        print('Start Test...')
        
        testFace = data_test_new[0,:]
        diffMat = data_train_new - tile(testFace,(trainNumber,1))
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        sortedDistIndicies = sqDistances.argsort()#找到最短的欧式距离的索引
        result_label = self.Train_Labels[sortedDistIndicies[0]]

        return result_label


if __name__ == '__main__':
    label_name = ['bihan','caoping','danshu','fanzhongyan','hanqi','huaiji','huirou','qiuhe','xinhe','zhaozhen']
    pca = PCA_Processor(label_name)
    pca.train_recognize(5)
    
    #处理测试数据，得到测试标签
    testimage = "face_recognition/PCA_test"
    test_name = os.listdir(testimage)
    test_labels = []
    test_dataset = []

    for num in range(len(test_name)):
    
        img_path = testimage + "/" + str(test_name[num])
        #图片校正
        img_cv2 = cv2.imread(img_path)
        img_index = pca.name_index.index(test_name[num])
        face_aligment = FACE_ALIGMENT()
        angle,gray = face_aligment.Geometric_Normalization(img_cv2,img_index)
        #img = Image.open(img_path)
        #gray = img.convert('L')        
        data_row = pca.data_prerocess(gray)#将图片处理为一行数据
        test_dataset.append(data_row)

    test_dataset = np.array(test_dataset)
    test_dataset = double(test_dataset) / 255

    for num  in range(len(test_name)):
        label = test_name[num].split('_')[0]
        test_labels.append(label_name.index(label))
    
    pca.recognize(test_labels,test_dataset)
    

    #recognize(testimage,dataset,eigenfaces) 
    # eigenfaces = np.reshape(eigenfaces,(100,100))
    # print(eigenfaces.shape)
    # eigenfaces = np.array(eigenfaces)
    # image = Image.fromarray(eigenfaces*255)
    # image.show()

