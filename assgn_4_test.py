import cv2
import numpy as np
import os
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import glob
import pickle
import sys


def predict_result(imgx1,model,bow_extract,surf):
    try :
        feat = bow_extract.compute(imgx1,surf.detect(imgx1))
        res = model.predict(feat)
        return res
    except :
        return [2]

def main():
	args = (sys.argv)
	vocab_file= str(args[1])
	model_file= str(args[2])
	with open(vocab_file, 'rb') as handle:
		vocab = pickle.load(handle)
	model = joblib.load(model_file)
	surf = cv2.xfeatures2d.SURF_create(3000)
	bow_extract = cv2.BOWImgDescriptorExtractor(surf,cv2.BFMatcher(cv2.NORM_L2))
	bow_extract.setVocabulary( vocab )
	webcam = cv2.VideoCapture(0)
	ret, frame = webcam.read() # get first frame
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (30,300)
	fontScale              = 1
	fontColor              = (255,255,255)
	lineType               = 2
	bottomLeftCornerOfText1 = (50,300)
	color = [[255 ,0 ,0],[0,255,0]]
	res = ['not Temoc' , 'Temoc','']
	while ret:
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
		cv2.putText(img_gray,'It is a : {} '.format(res[predict_result(img_gray,model,bow_extract,surf)[0]]), 
				bottomLeftCornerOfText, 
				font, 
				fontScale,
				fontColor,
				lineType)
		cv2.imshow('frame',img_gray)
		ret, frame = webcam.read()


	webcam.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
