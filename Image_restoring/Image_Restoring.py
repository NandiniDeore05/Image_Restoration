import cv2
import numpy as np

#Load our damaged photo
image = cv2.imread('abraham.jpg')
cv2.imshow('Original Damaged Photo', image)
cv2.waitKey(0)

#Load the photo where we've marked the damaged areas
marked_damages = cv2.imread('mask.jpg', 0)
cv2.imshow('Marked Damages', marked_damages)
cv2.waitKey(0)

#Lets make a mask out of the marked image by changing all colours that are notwhite, to black
ret, thresh1 = cv2.threshold(marked_damages, 254, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary', thresh1)
cv2.waitKey(0)

#Let's dilate (make thicker) the marks since thresholding have made them slightly narrower
kernel = np.ones((7,7), np.uint8)
mask = cv2.dilate(thresh1, kernel, iterations = 1)
cv2.imshow('Dilated Mask', mask)
cv2.imwrite("abraham_mask.png", mask)

cv2.waitKey(0)
restored = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

cv2.imshow('Restored', restored)
cv2.waitKey(0)
cv2.destroyAllWindows()

