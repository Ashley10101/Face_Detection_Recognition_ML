import pandas as pd
import numpy as np
import xlrd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import xlsxwriter


def create_data():
    
    label = []
    feature = []
    data = xlrd.open_workbook(r'HAAR Feature.xlsx')
    print('Read Data Done!')
    table = data.sheets()[0]
    
    for i in tqdm(range(table.nrows)):
        data = table.row_values(i)
        label.append(data[0])
        feature.append(data[1:])
    feature = np.array(feature)
    label = np.array(label)
    print('OK!')
    return feature, label



class AdaBoost(object):

    def __init__(self, weekClassifierNumber = 101):
        self.weekClassifierNumber = weekClassifierNumber
        
    def _init_parameters(self, features, labels):
        '''
        initialize the parameters
        N: the length of data
        n: the length of feature
        weights: Weights for each training data set
        alpha: Weights of each weak classifier
        '''
        self.X = features
        self.Y = labels
        self.N = len(features)
        self.n = 1 
        self.weights = np.ones(self.N)/len(features)
        self.alpha = np.zeros(self.weekClassifierNumber)
        self.minerror = 0.5
        self.weekClassifiers = {}

    def calculateAlpha(self,error):
        
        return float(0.5 * np.log((1 - error) / max(error , 1e-16)))


    def calculateWeight(self, alpha, originalWeight, y, g):
        weight = originalWeight * (np.e ** (-alpha * y * g))
        # w = originalWeight*np.exp(-alpha*y*g)
        # weight = w/w.sum()
        return weight

    def sign(self,value):
        if value > 0:
            return 1
        else:
            return -1

    def learn(self, x, featureValue):
        # if x >= featureValue:
        #     return self.sign(alpha * 1)
        # else:
        #     return self.sign(alpha * -1)
        if x >= featureValue:
            return 1
        else:
            return -1

    def weakClassify(self, X, val, type):   
        results = np.ones((X.shape[0], 1))
        if type == 'up':
            results[X <= val] = -1.0
        else:
            results[X > val] = -1.0
        results = np.array(results).flatten()
        return results

    def findBestCut(self, dataset, target, weights):
        #print(weights)
        # Take the eigenvalues v in order, larger than v is predicted to be 1 and less than v is predicted to be -1
        newWeights = []
        minerror = 0.5
        featureRow = dataset
        # Find the best segmentation value for each feature
        for featureIndex in range(len(featureRow)):
            # Forward and backward judgments
            for type in ['up', 'down']:
                featureValue = featureRow[featureIndex]

                resultRow = self.weakClassify(dataset, featureValue, type)
                resultRow = list(resultRow.reshape(-1))

                #error = np.dot(np.multiply(resultRow, target), weights.T)
                errArr = np.ones(len(resultRow))
                errArr[resultRow == target] = 0

                error = np.dot(errArr,weights)
                
                # If error is less than minerror, it means that this is the most qualified weak classifier G in this dataset weight, record the corresponding weak classifier weight and division value and the feature group where the division value is located and return
                if error < minerror:
                    
                    newWeights = []
                    # Get the weights of this classifier
                    minerror = error
                    self.minerror = error
                    alpha = self.calculateAlpha(error)
                    # print('alpha is :' + str(alpha))

                    for weightNumber in range(len(weights)):
                        #g = self.learn(featureRow[weightNumber], featureValue)
                        g = resultRow
                        newWeights.append(self.calculateWeight(alpha, weights[weightNumber], target[weightNumber], g[weightNumber]))

                    newWeights = newWeights / sum(newWeights)
                    featurevalue = featureValue
                    flag = type

                    # Returns the feature number, the best segmentation value
        return featurevalue, alpha, newWeights,flag

    def train(self, features, labels):

        self._init_parameters(features, labels)

        featureNumber = self.n

        for classifierSeq in range(self.weekClassifierNumber):
        # Cycle to find a good division
            #print(self.weights)
            if self.minerror == 0:
                break
            featureValue, alpha, newWeights,type = self.findBestCut(self.X, self.Y, self.weights)

            if len(newWeights) == 0:
                break
            # featureValue is the best division value of the feature
            self.weekClassifiers["WClassifier" + str(classifierSeq)] = [featureNumber, featureValue, alpha,type]
            self.weights = newWeights
            self.alpha[classifierSeq] = alpha
            pre_result = self.train_predict(self.X,classifierSeq)
            #print(pre_result)
            score = accuracy_score(self.Y,pre_result)
            if score > 0.99:
                break

        self.weekClassifierNumber = classifierSeq
        return 'train done!'
    
    def train_predict(self,feature,number):
        m = feature.shape[0]
        final_result = np.zeros(m)
        if number == 0:
            split_value = self.weekClassifiers["WClassifier" + str(0)][1] # Determine the feature value as a basis for classification
            alpha = self.weekClassifiers["WClassifier" + str(0)][2]
            type = self.weekClassifiers["WClassifier" + str(0)][3]
            result = self.weakClassify( feature, split_value, type)
            final_result += alpha * result


        for i in range(number):
            split_value = self.weekClassifiers["WClassifier" + str(i)][1] # Determine the feature value as a basis for classification
            alpha = self.weekClassifiers["WClassifier" + str(i)][2]
            type = self.weekClassifiers["WClassifier" + str(i)][3]
            result = self.weakClassify( feature, split_value, type)
            final_result += alpha * result
        
        return np.sign(final_result)

    def predict(self,feature):
        m = feature.shape[0]
        final_result = np.zeros(m)

        for i in range(self.weekClassifierNumber):
            split_value = self.weekClassifiers["WClassifier" + str(i)][1]#确定作为划分依据的特征值是多少
            alpha = self.weekClassifiers["WClassifier" + str(i)][2]
            type = self.weekClassifiers["WClassifier" + str(i)][3]
            result = self.weakClassify( feature, split_value, type)

            final_result += alpha * result
        
        return np.sign(final_result)

if __name__ == '__main__':
    X, y = create_data()

    # Train a classifier for each feature
    SCORE = []
    print('Start Training...')
    for index in tqdm(range(len(X[0]))):
        Train = X[:,index].flatten()
        X_train, X_test, y_train, y_test = train_test_split(Train, y, test_size=0.25, random_state=98)
        adaboost = AdaBoost()
        adaboost.train(X_train,y_train)
        test_predict = adaboost.predict(X_test)
        score = accuracy_score(y_test,test_predict)
        print(score)
        SCORE.append(score) 
    score_index = np.array(SCORE)
    sindex = list(np.argsort(-score_index))
    SCORE.sort(reverse=True)    
    print(SCORE)
    print(sindex)

    # sindex = [781, 776, 686, 2812, 2842, 681, 2807, 791, 2282, 2847, 2802, 826, 771, 2777, 691, 831, 2767, 2797, 696, 2837]
    # Create a new table to save strong classifiers
    workbook = xlsxwriter.Workbook('Strong Classifier.xlsx')
    worksheet = workbook.add_worksheet('sheet1')
    header = ["WClassifier"+ str(i) for i in range(100)]
    worksheet.write_row('A1',header)
    # Save the first 12 as useful strong classifiers
    print('Save 12 Strong Classifiers:')
    for i in tqdm(range(12)):
        featureindex = sindex[i]
        Train = X[:,featureindex].flatten()
        X_train, X_test, y_train, y_test = train_test_split(Train, y, test_size=0.25, random_state=98)
        adaboost = AdaBoost()
        adaboost.train(X_train,y_train)
        test_predict = adaboost.predict(X_test)
        score = accuracy_score(y_test,test_predict)
        print(score)
        Split_Value = []
        Alpha = []
        Type = []
        for j in range(adaboost.weekClassifierNumber):
            split_value = adaboost.weekClassifiers["WClassifier" + str(j)][1]
            Split_Value.append(split_value)
            alpha = adaboost.weekClassifiers["WClassifier" + str(j)][2]
            Alpha.append(alpha)
            type = adaboost.weekClassifiers["WClassifier" + str(j)][3]
            Type.append(type)

        worksheet.write_row('A'+str(4*i+2),Split_Value)
        worksheet.write_row('A'+str(4*i+3),Alpha)
        worksheet.write_row('A'+str(4*i+4),Type)

    workbook.close() 
    print('Save OK!')