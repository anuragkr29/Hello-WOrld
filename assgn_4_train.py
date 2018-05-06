import cv2
import numpy as np
import os
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import glob
import pickle
def main():
	temoc_data = glob.glob('train/1*.jpg')
	other_data = glob.glob('train/2*.jpg')
	all_images_dir = glob.glob('train/*.jpg')
	bow = cv2.BOWKMeansTrainer(40)
	surf1 = cv2.xfeatures2d.SURF_create(2000)
	#cv2.BOWImgDescriptorExtractor
	for image_src in all_images_dir:
		imgx = cv2.imread(image_src,0)
		# Find keypoints and descriptors directly
		kp, des = surf1.detectAndCompute(imgx,None)
		bow.add(des);

	vocabulary = bow.cluster(); 
	with open('bow_pickle.pickle', 'wb') as f:
		pickle.dump(vocabulary ,f)
	bow_extract = cv2.BOWImgDescriptorExtractor(surf1,cv2.BFMatcher(cv2.NORM_L2))
	bow_extract.setVocabulary( vocabulary )
	#print ("bow vocab", np.shape(vocabulary), vocabulary)
	train_data = []



	arr1,labels1=get_surf_feature(temoc_data,1,surf1,bow_extract)
	arr2,labels2=get_surf_feature(other_data,0,surf1,bow_extract)

	feature_vector = np.concatenate((arr1,arr2))
	labelss = np.concatenate((labels1,labels2))

	clf1=SVC(gamma=1,C=10)

	clf1.fit(feature_vector,labelss)

	#ypred = clf.predict()
	joblib.dump(clf1, "trained_surf.model", compress=3)
	print('Files saved are : trained_surf.model , bow_pickle.pickle')

def get_surf_feature(image_dir,target_class,surf1,bow_extract):
    i=0
    arr_s=np.array([])
    labels=[]
    for image_src in image_dir:
        imgx = cv2.imread(image_src,0)
        kp, des = surf1.detectAndCompute(imgx,None)
        gg=bow_extract.compute(imgx, kp, des)
        if i==0:
            arr_s = np.vstack(gg)
            i=i+1
        else:
            arr_s = np.vstack((arr_s,gg))
        labels.append(target_class)
    return arr_s,labels
if __name__ == '__main__':
	main()