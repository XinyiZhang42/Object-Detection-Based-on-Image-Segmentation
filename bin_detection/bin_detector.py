'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''
import math
import numpy as np
from skimage import measure
import cv2
class BinDetector():
	def __init__(self):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		#'''
		self.mu = np.array([[113.91622572, 206.50479637, 160.97238248],
 [ 95.67741935,  52.97616487,  60.45967742],
 [ 36.38911829,  88.69808113, 155.77362157],
 [ 52.30760184, 133.73473752, 105.19736983],
 [ 67.48803014,  15.1526444,  172.78927538],
 [132.48281016, 155.67115097,  81.91479821],
 [145.94249685, 193.01765448, 208.60857503],
 [104.28818549, 109.78918151, 224.62268573],
 [111.76906844,  25.25324088, 230.08260476],
 [ 25.25,       130.38509874, 246.85682226]])
		self.sigma = np.array([[1.41120547e+01, 8.77918703e+02, 2.12284418e+03],
 [1.40234361e+03, 8.18073805e+02, 3.42142639e+02],
 [2.67957630e+03, 3.17192633e+03, 3.61938208e+03],
 [1.37363416e+02, 4.34116528e+03, 1.08501055e+03],
 [1.94666800e+03, 5.83112124e+01, 1.29454660e+03],
 [1.78969392e+01, 6.92435954e+02, 2.05369229e+03],
 [4.34819947e+03, 5.56532882e+03, 3.49797188e+03],
 [4.78285527e+01, 1.79100297e+03, 4.18704482e+02],
 [2.78521889e+02, 1.71849044e+02, 1.19315808e+03],
 [5.20994165e+00, 7.83096391e+03, 1.25377615e+02]])
		#'''
		#self.mu = np.load('./mu.npy')
		#self.sigma = np.load('./sigma.npy')
	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image WRONG:MASK IMAGE
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		# YOUR CODE HERE
		mu = self.mu
		sigma = self.sigma

		# convert from RGB TO HSV
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        
		# Replace this with your own approach
		num_classes = 10
		X = img
     
		# vectorize input image
		length = X.shape[0]*X.shape[1]
		vectorized_X = np.reshape(X, (length, 3))

		# initialize arrays to store predicted class scores and final image mask values
		predict_scores = np.zeros((vectorized_X.shape[0], num_classes))  
		y = np.zeros((vectorized_X.shape[0],1))

		#To save running time, remove exponential calculation
		for i in range(num_classes):  #iterate classes
			temp_sum = 0
			for j in range(3):  # iterate channels
				temp_sum += math.log((sigma[i,j]) ** 2) + (((vectorized_X[:,j] - mu[i,j]) ** 2) / (sigma[i,j] ** 2))
			predict_scores[:,i] = temp_sum
		y = np.argmin(predict_scores, axis=1) + 1
        
		# Reshape mask to get segmented image
		mask_img = np.reshape(y, (X.shape[0], X.shape[1]))
		#Generate binary image       
		mask_img = np.where(mask_img==1,1,0)

		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE

		mask = img
		#Convert bin blue to 255 for erode and dilated
		#Refer to https://stackoverflow.com/questions/57196047/how-to-detect-all-the-rectangular-boxes-in-the-given-image
		mask *= 255
		mask = mask.astype('uint8')
		kernel = np.ones((13,13), np.uint8)
		kernel2 = np.ones((5,5), np.uint8)
		erode = cv2.erode(mask, kernel, iterations = 1)
		dilation = cv2.dilate(erode, kernel2, iterations = 3)
		#Blue the image
		blurred = cv2.GaussianBlur(dilation, (3,3),0)
		thresh = cv2.threshold(blurred, 128, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		boxes = []


		contours= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
		for c in contours:
			x,y,w,h = cv2.boundingRect(c)
			area_ratio = cv2.contourArea(c)/(mask.shape[0]*mask.shape[1])
			#blue_area = np.sum(mask[y:y+h,x:x+w])/255
			color_ratio = cv2.contourArea(c)/(w*h)
			if 1.1<= h/w <=2 and area_ratio > 0.01 and color_ratio>0.5:
				boxes.append([x,y,x + w,y + h])


		boxes.sort()
		return boxes