import cv2

im = cv2.imread(r'F:\\4.2\\2 Machine learning laboratory 452\\OPENCV\\img.jpg')  
cv2.imshow('Original Image',im)  
cv2.imshow('Blurred Image', cv2.blur(im, (3,3)))  
cv2.waitKey(0)  
cv2.destroyAllWindows()  
