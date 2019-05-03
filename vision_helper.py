import cv2
import numpy as np


def apply_hsv_mask(img,lower_mask,upper_mask):
    #might have to change this up to make it most useful for what type img, lower_mask etc is
    '''
    lower_mask and upper_mask are lists of 3 values representing h,s,v
    img is np array of bgr image (default opencv type)
    returns masked version of img as bgr
    '''
    hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_mask_array=np.array(lower_mask)
    upper_mask_array=np.array(upper_mask)
    masked_img=cv2.inRange(hsv_img,lower_mask_array,upper_mask_array)
    #cv2.imshow('Image',masked_img) #uncomment these to show pic each time
    #cv2.waitKey(0)
    return(masked_img)

def look_at_hsv(img):
    hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    print("B value is H, scale of 0-180")
    print("G value is S, scale of 0-255")
    print("R value is V, scale of 0-255")
    print("press any key on the picture to close it")
    #hsv_img=cv2.resize(hsv_img,(400,300))
    cv2.imshow("Image",hsv_img)
    cv2.waitKey(0)
