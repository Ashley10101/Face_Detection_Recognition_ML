import cv2
import numpy as np
from numpy import *
import os
from PIL import Image
from skimage import exposure
import xlsxwriter
from tqdm import tqdm
import math

global dist
global integimg
global varNormFactor

# Get integral image
def integral(img):
    # The integral image has one more row and one more column than the original image, and the first row and the first column of the integral image are 0
    integimg = np.zeros( shape = (img.shape[0] + 1, img.shape[1] + 1), dtype = np.int32 )
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            integimg[i+1][j+1] = img[i][j] + integimg[i][j+1] + integimg[i+1][j] - integimg[i][j]
    # print('Done integral image!')
    return integimg

# Obtain Haar features at a single scale
def haar_onescale(varNormFactor,img,integimg,haarblock_width,haarblock_height):
    # Store Haar features with haar feature map, step equals to 1, no padding
    haarimg = np.zeros( shape = (img.shape[0] - haarblock_width + 1, img.shape[1] - haarblock_height + 1 ), dtype = np.int32 )
    haarimg_1 = haarimg
    haarimg_2 = haarimg
    haarimg_3 = haarimg
    haarimg_4 = haarimg
    haarimg_5 = haarimg
    haar_feature_onescale = []
    for i in range( haarimg.shape[0] ):
        for j in range( haarimg.shape[1] ): 
            # Map i,j back to the coordinates of the original graph
            m = haarblock_width + i
            n = haarblock_height + j

            haar_all = integimg[m][n] - integimg[m-haarblock_width][n] - integimg[m][n-haarblock_height] + integimg[m-haarblock_width][n-haarblock_height]
            
            # Compute a rectangle of type (1, 2): Black on top, white on bottom
            haar_black = integimg[m][n- int( haarblock_height/2 )] - integimg[m-haarblock_width][n-int( haarblock_height/2 )]- integimg[m][n-haarblock_height] + integimg[m-haarblock_width][n-haarblock_height]            
            haarimg_1[i][j] = 1 * haar_all - 2 * haar_black #1*all - 2*black = white - black
            haar_feature_onescale.append(haarimg_1[i][j]/varNormFactor)

            # Compute a rectangle of type (2, 1): Black on left, white on right
            haar_black = integimg[m- int(haarblock_width/2)][n] - integimg[m- int(haarblock_width/2)][n-haarblock_height]- integimg[m - haarblock_width ][n] + integimg[m-haarblock_width][n-haarblock_height]
            haarimg_2[i][j] = 1 * haar_all - 2 * haar_black
            haar_feature_onescale.append(haarimg_2[i][j]/varNormFactor)

            # Compute a rectangle of type (3, 1): Black in the middle and white on the left and right
            haar_white = integimg[m- int(haarblock_width/3)][n] - integimg[m- int(haarblock_width/3)][n-haarblock_height]- integimg[m - haarblock_width ][n] + integimg[m-haarblock_width][n-haarblock_height]
            haar_black_white = integimg[m- 2*int(haarblock_width/3)][n] - integimg[m- 2*int(haarblock_width/3)][n-haarblock_height]- integimg[m - haarblock_width ][n] + integimg[m-haarblock_width][n-haarblock_height]
            haar_black = haar_black_white - haar_white
            haarimg_3[i][j] = 1 * haar_all - 3 * haar_black # 1*all - 3*black = white - 2*black
            haar_feature_onescale.append(haarimg_3[i][j]/varNormFactor)

            # Compute a rectangle of type (1, 3): Black in the middle and white on the top and bottom
            haar_white = integimg[m][n- int( haarblock_height/3)] - integimg[m-haarblock_width][n-int( haarblock_height/3)]- integimg[m][n-haarblock_height] + integimg[m-haarblock_width][n-haarblock_height]
            haar_black_white = integimg[m][n- 2 * int( haarblock_height/3)] - integimg[m-haarblock_width][n-2 * int( haarblock_height/3)]- integimg[m][n-haarblock_height] + integimg[m-haarblock_width][n-haarblock_height]
            haar_black = haar_black_white - haar_white
            haarimg_4[i][j] = 1 * haar_all - 3 * haar_black
            haar_feature_onescale.append(haarimg_4[i][j]/varNormFactor)

            # Compute a rectangle of type (2, 2): Black in the middle and white on the top and bottom
            haar_black_1 = integimg[m- int(haarblock_width/2)][n- int( haarblock_height/2 )] - integimg[m-haarblock_width][n-int( haarblock_height/2 )]- integimg[m- int(haarblock_width/2)][n-haarblock_height] + integimg[m-haarblock_width][n-haarblock_height]
            haar_black_2 = integimg[m][n] - integimg[m][n-int( haarblock_height/2 )]- integimg[m- int(haarblock_width/2)][n] + integimg[m- int(haarblock_width/2)][n- int( haarblock_height/2 )]
            haar_black = haar_black_1 + haar_black_2
            haarimg_5[i][j] = 1 * haar_all - 2 * haar_black
            haar_feature_onescale.append(haarimg_5[i][j]/varNormFactor)


    # print('The current dimensionality of Haar features： {}'.format(len(haar_feature_onescale)))
    return haar_feature_onescale

# Obtain Haar features at full scale
def harr(varNormFactor,dist, integimg,haarblock_width, haarblock_height, Scale_num):
    feature = []
    haar_num = 0
    for i in range(Scale_num):
        # Equal magnification
        haarblock_width = i*haarblock_width + 6
        haarblock_height = i*haarblock_height + 6
        # print('The current Haarblock scale is:({}, {})'.format(haarblock_height, haarblock_width)) 
        haar_feature_onescale = haar_onescale(varNormFactor,dist, integimg, haarblock_width, haarblock_height)
        haar_num += len(haar_feature_onescale) 
        feature = feature + haar_feature_onescale
        #feature.append(haar_feature_onescale)
        haarblock_width = 6
        haarblock_height = 6
    # Calculate the total Haar feature dimension
    # print('Calculate the total Haar feature dimension')
    # print('The total dimensionality of Haar features is：{}'.format(haar_num))
    return feature

def histeq(im,nbr_bins = 256):
    """Histogram equalization of a grayscale image"""
    # Calculate the histogram of the image
    imhist,bins = histogram(im.flatten(),nbr_bins,normed= True)
    cdf = imhist.cumsum()   
    cdf = 255.0 * cdf / cdf[-1]
    # Compute new pixel values using linear interpolation of the cumulative distribution function
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape)



if __name__ == '__main__':

    # Convert images to grayscale sharpening
    img_folder = "../adaboost_image"
    img_name = os.listdir(img_folder)

    # Construct new table to store Haar features
    workbook = xlsxwriter.Workbook('HAAR Feature.xlsx')
    worksheet = workbook.add_worksheet('sheet1')    
    
    # Get the feature values of each image
    for num in tqdm(range(len(img_name))):
        data = []
        label = int(img_name[num][0])
        label = label * 2 - 1
        data.append(label)
        img_path = img_folder + "/" + str(img_name[num])
        img = Image.open(img_path)
        
        # Conversion to grayscale
        gray = img.convert('L')
        gray = array(gray)
        dist = histeq(gray)  

        # Extracting haar features
        haarblock_width = 6
        haarblock_height = 6
        width_limt = int( img.size[0] / haarblock_width )
        height_limt = int( img.size[1] / haarblock_height )
        Scale_num = min( height_limt, width_limt )
        integimg = integral(dist)

        # Find the normalization factor
        imgsum = integimg[-1][-1]
        sqsum = sum(np.dot(np.array(dist),np.array(dist)))
        wh = 24 * 24
        mean = imgsum/wh
        sqmean = sqsum/wh
        varNormFactor = math.sqrt(abs(mean*mean - sqmean))
        haar_feature = harr(haarblock_width, haarblock_height, Scale_num)#2900个特征

        # Store results
        data = data + haar_feature

        worksheet.write_row('A'+str(num+1),data)
        
    workbook.close() 
    print('Save OK!')



