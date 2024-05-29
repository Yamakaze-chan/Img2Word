#Document image orientation correction
#This approach is based on text orientation

#Assumption: Document image contains all text in same orientation

import cv2
import numpy as np
from skimage.filters import threshold_yen

debug = True

#Display image
def display(img, frameName="OpenCV Image"):
    if not debug:
        return
    h, w = img.shape[0:2]
    neww = 800
    newh = int(neww*(h/w))
    img = cv2.resize(img, (neww, newh))
    cv2.imshow(frameName, img)
    cv2.waitKey(0)

#rotate the image with given theta value
def rotate(img, theta):
    rows, cols = img.shape[0], img.shape[1]
    image_center = (cols/2, rows/2)
    
    M = cv2.getRotationMatrix2D(image_center,theta,1)

    abs_cos = abs(M[0,0])
    abs_sin = abs(M[0,1])

    bound_w = int(rows * abs_sin + cols * abs_cos)
    bound_h = int(rows * abs_cos + cols * abs_sin)

    M[0, 2] += bound_w/2 - image_center[0]
    M[1, 2] += bound_h/2 - image_center[1]

    # rotate orignal image to show transformation
    rotated = cv2.warpAffine(img,M,(bound_w,bound_h),borderValue=(255,255,255))
    return rotated


def slope(x1, y1, x2, y2):
    if x1 == x2:
        return 0
    slope = (y2-y1)/(x2-x1)
    theta = np.rad2deg(np.arctan(slope))
    return theta

def pre_process_image(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, threshold_yen(img), 255, cv2.THRESH_BINARY)
    bitwise = cv2.bitwise_not(thresh)
    erosion = cv2.erode(bitwise, np.ones((5, 5) ,np.uint8), iterations=5)
    dilation = cv2.dilate(bitwise, np.ones((7, 7) ,np.uint8), iterations=5)
    return dilation

def rotate_image(img):
    textImg = img.copy()

    small = pre_process_image(cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY))

    #find the gradient map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    # display(grad)

    #Binarize the gradient image
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # display(bw)

    #connect horizontally oriented regions
    #kernal value (9,1) can be changed to improved the text detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # display(connected)

    # using RETR_EXTERNAL instead of RETR_CCOMP
    # _ , contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #opencv >= 4.0



    mask = np.zeros(bw.shape, dtype=np.uint8)
    #display(mask)
    #cumulative theta value
    cummTheta = 0
    #number of detected text regions
    ct = 0
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        #fill the contour
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        #display(mask)
        #ratio of non-zero pixels in the filled region
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        #assume at least 45% of the area is filled if it contains text
        if r > 0.45 and w > 8 and h > 8:
            #cv2.rectangle(textImg, (x1, y), (x+w-1, y+h-1), (0, 255, 0), 2)

            rect = cv2.minAreaRect(contours[idx])
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(textImg,[box],0,(0,0,255),2)

            #we can filter theta as outlier based on other theta values
            #this will help in excluding the rare text region with different orientation from ususla value 
            theta = slope(box[0][0], box[0][1], box[1][0], box[1][1])
            cummTheta += theta
            ct +=1 
            #print("Theta", theta)
            
    #find the average of all cumulative theta value
    if ct == 0:
        orientation = cummTheta
    else:
        orientation = cummTheta/ct
    # print("Image orientation in degress: ", orientation)
    finalImage = rotate(img, orientation)
    # display(textImg, "Detectd Text minimum bounding box")
    # display(finalImage, "Deskewed Image")
    return [finalImage, orientation]

# if __name__ == "__main__":
#     filePath = r'test_img\31.jpg'
#     main(filePath)