#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


# -*- coding: utf-8 -*-
# """
# Created on Fri Mar 29 15:24:09 2019

# @author: MODAK
# """
# from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import matplotlib.pyplot as plt


def Canny(image):
      gray=cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
      blur=cv2.GaussianBlur(gray, (5, 5), 0)
      #5, 5 is feature scale matrix size. 
      canny=cv2.Canny(blur, 30, 120)
      #50=lower_threshold for gradient change same for 15 high
      return canny
    
def region_of_interests(image):
      height=image.shape[0]#0for no. of rows or height
      triangle=np.array([
          [(200, height), (1100, height), (550, 250)]
          ])
      mask=np.zeros_like(image)#same dimension complete black image
      cv2.fillPoly(mask, triangle, 255)#255 to fill the polygon in black mask with white region
      masked_img=cv2.bitwise_and(image, mask)#original image iwth white highlighted region
      return masked_img

def display_lines(image, lines):
      line_image=np.zeros_like(image)
      if lines is not None:#its not empty
        for line in lines:
          #print (line)#it's a 2D array we need to convert each line into a 1D
          #x1=line.reshape(4)#1d array with 4 elements
          x1, y1, x2, y2=line.reshape(4)#dividing them into points
          cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 7)#255,0,0 for bluecolour to represent line with thickness 10
      return line_image    
      
  
def average_slope_intercept(image, lines):
      left_fit=[]
      right_fit=[]
      for line in lines:
        x1, y1, x2, y2=line.reshape(4)#dividing them into points
        parameters=np.polyfit((x1,x2), (y1,y2), 1)# putting a 1 degree polynomial
        #print (parameters)# it brought a lot of no. of lines
        slope=parameters[0]
        intercept=parameters[1]
        if slope<0:
          left_fit.append((slope, intercept))
        else:
          right_fit.append((slope,intercept))
      left_fit_average=np.average(left_fit, axis=0)
      right_fit_average=np.average(right_fit, axis=0)
      left_line=make_coordinates(image, left_fit_average)
      right_line=make_coordinates(image, right_fit_average)
      return left_line, right_line
    
    
def make_coordinates(image, line_parameters):
      slope,intercept=line_parameters
      print(image.shape)
      print(slope)
      y1=image.shape[0]
      y2=int(y1*(3/5))
      x1=int((y1-intercept)/slope)
      x2=int((y2-intercept)/slope)
      p=np.array([x1 ,y1, x2 ,y2])
      return p

#testing on image
image_1=cv2.imread('Downloads/test_image.jpg')
lane_image=np.copy(image_1)
# cv2.imshow('iamge',lane_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

canny=Canny(lane_image)#1st black white
region=region_of_interests(canny)#2nd black white 
lines=cv2.HoughLinesP(region, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# 2 is for hough space y=mx+c or xcos0+ysin0=p
# 2 is 2 pixel for every 1 deg=pie/180 and 100 is min no. of lines to pass through same point and a blank array and maxLineGap for gap between lines to detect sepearate lines
averaged_lines=average_slope_intercept(lane_image, lines)
line_image=display_lines(lane_image,averaged_lines)#glorifying lines in 2nd black white image
combo_image=cv2.addWeighted(lane_image, 0.8, line_image, 1 , 1)#0.8 and 1 is weight assigned to it and reamining 1 is gamma value
# cv2.imshow('canny',canny)
# #print("--------------------------------------")
# cv2.imshow('new_iamge',combo_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#testing on video
cap=cv2.VideoCapture("Downloads/test2.mp4")
while(cap.isOpened()):
    _, frame=cap.read()
    canny=Canny(frame)#1st black white
    region=region_of_interests(canny)#2nd black white 
    lines=cv2.HoughLinesP(region, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # 2 is 2 pixel for every 1 deg=pie/180 and 100 is min no. of lines to pass through same point and a blank array and maxLineGap for gap between lines to detect sepearate lines
    averaged_lines=average_slope_intercept(frame, lines)
    line_image=display_lines(frame,averaged_lines)#glorifying lines in 2nd black white image
    combo_image=cv2.addWeighted(frame, 0.8, line_image, 1 , 1)#0.8 and 1 is weight assigned to it and reamining 1 is gamma value
    #cv2_imshow(canny)
    #print("--------------------------------------")
    
    cv2.imshow("windowName",combo_image)
    if cv2.waitKey(1) & 0xFF ==ord('q'):# displaying noraml speed 1ms/frame
        break
cap.release()
cv2.destroyAllWindows()

        



# In[ ]:




