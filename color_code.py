import vision_helper
import cv2
import numpy as np

'''
order:
take picture
do gamma adjustment?
for each topping:
    look for topping in image:
        apply hsv
        find blobs (potentially using circle or whatever)
        find c.o.m. of blobs
with coordinates relative to camera, apply transfrom to get topping coordinates relative to base
send out the coordinates!
'''

#toppings here: (all of these need more testing and to work better with a better picture)
pep={ #seem to work pretty well
"name":"pepperoni",
"upper_hsv_mask":[175,140,90],
"lower_hsv_mask":[142,51,55],
}
piz={ #these values work well but different processing is needed for pizza
"name":"pizza",
"upper_hsv_mask":[8,220,170],
"lower_hsv_mask":[0,190,150],
}
ham={
"name":"ham",
"upper_hsv_mask":[180,190,210],
"lower_hsv_mask":[160,165,160],
}

toppings_list=[pep,ham]

def get_all_items(img):
    location={}
    for topping in toppings_list:
        location[topping["name"]]=find_topping_locations(img,topping)
    return(location)

def find_topping_locations(img,topping):
    #img is numpy array
    #topping is a dict from toppings_list
    #for a specific topping type, find the x,y,z coordinates of those toppings

    #get correct upper and lower values
    upper=topping["upper_hsv_mask"]
    lower=topping["lower_hsv_mask"]

    masked_img=vision_helper.apply_hsv_mask(img,lower,upper)
    #should be black and white coming out of this

    #erode and dilate to reduce noise
    denoised=denoise(masked_img) #works well for pepperoni,not so much for pizza

    #use contours to detect different blobs
    im2,contours,hierarchy=cv2.findContours(denoised,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("Image",im2)
    #print(contours)
    #print(hierarchy)
    #print(len(contours))

    im3,outer_contours,hi=cv2.findContours(denoised,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #if nested contours, get the outer ones only

    #print(len(outer_contours))
    locs=find_locs(contours)
    print(locs)
    cv2.waitKey(0)
    return(locs)
    #oof

def denoise(img):
    #erode first then dilate to reduce noice and
    size=3
    iters=1

    cv2.imshow("Image",img)
    cv2.waitKey(0)
    kernel = np.ones((size,size),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = iters)
    cv2.imshow("Image",erosion)
    cv2.waitKey(0)

    dilation=cv2.dilate(erosion,kernel,iterations=iters+2)
    cv2.imshow("Image",dilation)
    cv2.waitKey(0)
    return(dilation)

def find_locs(contours):
    averages=[]
    for blob in contours:
        x_avg=0
        y_avg=0
        for pair in blob:
            x_val=pair[0][0] #because weird nested list thing
            y_val=pair[0][1]
            x_avg+=x_val
            y_avg+=y_val
        #blob should be numpy array
        x_avg=x_avg/len(blob)
        y_avg=y_avg/len(blob)
        averages.append((x_avg,y_avg)) #add centroid for each blob
    return averages

img=cv2.imread('modules/test_Color.png')
img = np.array(img, dtype=np.uint8)

pizza=cv2.imread('modules/test2_Color.png')
pizza=np.array(pizza, dtype=np.uint8)

#find_topping_locations(pizza,ham)
#vision_helper.look_at_hsv(img)
#find_topping_locations(img,pep)
