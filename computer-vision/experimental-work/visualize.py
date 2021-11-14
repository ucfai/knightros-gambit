import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import streamlit as st
from PIL import Image


def colorChannels(image): #View Color Spaces

    choice = st.sidebar.selectbox('Color Channels', ["HSV","GRAY"])

    if(choice == "HSV"):
        image = cv.cvtColor(image,cv.COLOR_BGR2HSV)

    if(choice == "GRAY"):
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY) 

    return image

# Transforming the image 


def hough(image, edges):

    #dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    #lines: A vector that will store the parameters (r,θ) of the detected lines
    #rho : The resolution of the parameter r in pixels. We use 1 pixel.
    #theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    #threshold: The minimum number of intersections to "*detect*" a line
    #srn and stn: Default parameters to zero. Check OpenCV reference for more info.

    lines = cv.HoughLines(edges,1,np.pi/180,150, None, 0, 0)
 
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        image = cv.line(image, pt1, pt2, (0,0,0), 2, cv.LINE_AA)
    
    return image

def threshType(thresh):

    if(thresh == "Binary"):
        thresh = cv.THRESH_BINARY 

    if(thresh == "BinaryINV"):
        thresh = cv.THRESH_BINARY_INV

    if(thresh == "Trunc"):
        thresh = cv.THRESH_TRUNC

    if(thresh == "TOZERO"):
        thresh = cv.THRESH_TOZERO

    if(thresh == "TOZEROINV"):
        thresh = cv.THRESH_TOZERO_INV

    return thresh    


def thresholds(image): #

    thresh = st.sidebar.selectbox('Select', ["None","Binary","BinaryINV", "Trunc","TOZERO","TOZEROINV", "Adaptive"])

    if(thresh == "None"):
        return image

    if(thresh == "Adaptive"):

        maxVal = st.sidebar.slider("Max Value",min_value=1,value = 255, max_value=255)
        method = st.sidebar.selectbox('Select', ["Mean","Gaussian"])

        if(method == "Mean"):
            method = cv.ADAPTIVE_THRESH_MEAN_C
        if(method == "Gaussian"):
            method = cv.ADAPTIVE_THRESH_GAUSSIAN_C    

#        thresh = st.sidebar.selectbox('Treshold', ["None","Binary","BinaryINV", "Trunc","TOZERO","TOZEROINV"])
      #  thresh = threshType(thresh)
        block_size = st.sidebar.slider("Block Size",min_value=1,value = 11, max_value=255)
        constant = st.sidebar.slider("Constant",min_value=1,value = 5, max_value=100)
        image = cv.adaptiveThreshold(image,maxVal,method,cv.THRESH_BINARY_INV,block_size,constant)
        return image
     
   

    min = st.sidebar.slider("Minimum Pixel Intensity",min_value=1,value = 100, max_value=255)
    max = st.sidebar.slider("Maximum Pixel Intensity",min_value=1,value = 255, max_value=255)

    type = threshType(thresh)
    ret, image = cv.threshold(image,min,max,type)

    return image

def blurring(image):

    type = st.sidebar.selectbox("Blur Type",["None","Boxfilter","Gaussian","Median","Bilateral"])
 
    if(type == "None"):
        return image

    if(type == "Boxfilter"): #takes the average of of the pixels under the kernel

        w = st.sidebar.slider("Kernel Width",min_value=1,value = 5, max_value=255)
        h = st.sidebar.slider("Kernel Height",min_value=1,value = 5, max_value=255)

        blur_img = cv.blur(image,(w,h))

    if(type == "Gaussian"): # uses a gaussian kernel

        w = st.sidebar.slider("Kernel Width",min_value=1,value = 5, max_value=100)
        h = st.sidebar.slider("Kernel Height",min_value=1, value = 5, max_value=100)
        sigmaX = st.sidebar.slider("SigmaX",min_value=0, max_value=255)

        blur_img  = cv.GaussianBlur(image,(w,h),sigmaX)

    if(type == "Median"): #takes the median of all the pixels under the kernel area and central element is replaced with this median value
        size = st.sidebar.slider("Filter Size",min_value=1,value = 5, max_value=255)
        blur_img = cv.medianBlur(image,size)

    if(type == "Bilateral"):

        d = st.sidebar.slider("Pixel Diameter",min_value=1, max_value=255)
        SigmaColor = st.sidebar.slider("SigmaColor",min_value=1,value = 75, max_value=255)
        SigmaSpace = st.sidebar.slider("SigmaSpace",min_value=1,value = 75, max_value=255)
        blur_img = cv.bilateralFilter(image,d,sigmaColor,sigmaSpace)

    return blur_img

        #arguments - d ( diameter of each pixel in neighborhood)
        #sigmaColor value of sigma in color space. Greater value means colors farther aways to each other will start to get mixed
        #Sigmacolr-   Value of \sigma in the coordinate space. The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range.


def corners(image):

    choice = st.sidebar.selectbox('Edge Detection', ["None","Canny","Shi-Tomashi"])

    if(choice == "None"):
        return image

    if(choice == "Shi-Tomashi"):

        edges = cv.goodFeaturesToTrack(image,100,0.01,100)
        edges = np.int0(edges)
         
        radius = st.sidebar.slider("Radius",min_value=1,value = 10, max_value=255)
        color =  st.sidebar.slider("Color",min_value=1, max_value=255)

        for i in edges:
           
           x,y = i.ravel()
           image = cv.circle(image,(x,y),radius,(color,0,0))

        return image

    if(choice == "Canny"):

        minval = st.sidebar.slider("Minval",min_value=1,value = 50, max_value=255)
        maxval =  st.sidebar.slider("Maxval",min_value=1,value = 100, max_value=200)
        aperature = st.sidebar.slider("aperature",min_value=1,value = 3, max_value=50)
        edges = cv.Canny(image,minval,maxval,aperature)
        image = hough(image,edges)


        return image


def contour(image):

    square_sums = []

    choice = st.sidebar.selectbox("Contour",["None","Contour"])
    count = 0
    newcontours = []

    if(choice == "None") :
        return image

    contours,hierarchy = cv.findContours(image,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    #image = cv.cvtColor(image,cv.COLOR_GRAY2RGB)

    val = st.sidebar.slider("Perimeter Limit",min_value=1, max_value=10000)
    pixelSum = st.sidebar.slider("Sum",min_value=0,max_value=100000)
    square = st.sidebar.slider("Square",min_value=0,max_value=63)

    zeros = np.zeros(image.shape,dtype="uint8")

    for contour in contours:
        
        perimeter = cv.contourArea(contour, False)

        if(perimeter>val):

            newcontours.append(contour) #Retrieve all the contours
            count = count + 1 

    if(square!=0):    
        #print(newcontours[square])        
        mask = cv.drawContours(zeros,[newcontours[square]],-1,255,-1) #Retrieves a square on the board
        newimg = cv.bitwise_and(image,image,mask = mask)
        square_sums.append(np.sum(newimg))


        print(np.sum(newimg))
        image = newimg

    print(count)

    return image


#Luis - This function will help us to load the image // It is working 04/07/2021 - Fixed
def load_image():
    

    st.title("Chess board project")
    img_upload = st.file_uploader(label = "Upload a file", type = ['JPG','PNG']) #Here we are creating the button to upload
     #a file with the type JPG or PNG

    if img_upload is not None: #Then if the img_upload is not None; in other words, is correct. We are going to display
        im = Image.open(img_upload)
        img = np.array(im)
        st.write("The image was uploaded correctly")
        return img

    if img_upload is None: #If the image wasn't uploaded
        st.write("The image doesn't have the file type JPG or PNG")

#Luis//  Working on this perspective Transform - 04/07/2021
def PerspectiveTransform(image):
    
  choice = st.sidebar.selectbox("Perspective Transform",["None","Yes"])
  
  if(choice == "None"):
      return image
  
  else:
      image = cv.circle(image,(755,43),20,(255,0,0),2) #Here we are dtecting one image. 
      return image
#We have to find an effiecent way to detech all the coordinates from the corners
#In order to change the perspective of the image.

def chessBoardCorners(image):

    choice = st.sidebar.selectbox("Chessboard Corners",["None","Yes"])
    
    if(choice == "None"):
        return image

    ret, corners = cv.findChessboardCorners(image, (4,4), None)
    print(corners)
    #If found, draw corners
    if ret == True:
        print("test")
       
        
    image = cv.drawChessboardCorners(image, (5,5), corners, ret)
    return image
    

def chessBoardCorners(image):

    choice = st.sidebar.selectbox("Chessboard Corners",["None","Yes"])
    
    if(choice == "None"):
        return image

    ret, corners = cv.findChessboardCorners(image, (4,4), None)
    print(corners)
    #If found, draw corners
    if ret == True:
        print("test")
       
        
    image = cv.drawChessboardCorners(image, (5,5), corners, ret)
    return image
    

if __name__ == "__main__":


    #image = cv.imread("onlineboard.jpg")
    image = load_image()
    image = colorChannels(image)
    image = blurring(image)
    image = thresholds(image)
    image = corners(image)
    image = contour(image)
    image = chessBoardCorners(image)
    image = PerspectiveTransform(image)

    
    col =  st.beta_columns(1)
    col[0].image(image,use_column_width=True)

 
 

        

