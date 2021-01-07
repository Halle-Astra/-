# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 21:48:21 2018
@author: Miracle
"""
import cv2
from matplotlib import  pyplot as plt 
import numpy  as np 
import time 
from PIL import Image 
import os 
from utils import process

def detectFace():
	type_need = '眯眼'
	if not os.path.exists(type_need):
		os.mkdir(type_need)
	#加载人脸检测的配置文件
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	#判断是否可行
	if eye_cascade.empty() and face_cascade.empty():
		raise IOError('Cannot load cascade classifier xml files!')
	#打开摄像头
	cap = cv2.VideoCapture(0)
	scaling_factor = 1.15
	if not cap.isOpened:
		raise IOError('Cannot open webcam!')
	time_begin = time.time()
	x_ls = []
	if not os.path.exists('w_h_ref.txt'):
		w_ref = 0
		h_ref = 0
	else:
		f = open('w_h_ref.txt')
		t = f.read().split()
		w_ref,h_ref = int(t[0]),int(t[1])
		f.close()
	w_ls = [] 
	h_ls = [] 
	cnt = 0 
	while True:
		ret,frame = cap.read()
		frame_ = frame
		if not ret:
			break
		frame = cv2.resize(frame,None,
		fx = scaling_factor,
		fy = scaling_factor,
		interpolation = cv2.INTER_LINEAR)
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		#获取脸部位置
		face_rects = face_cascade.detectMultiScale(gray)
		#获取脸部地址
		for (x,y,w,h) in face_rects:
			roi_gray = gray[y+int(0.3*h):int(y+0.6*h),x:x+h]
			roi_color = frame[y+int(0.3*h):int(y+0.6*h),x:x+h]
			
			roi_gray = (roi_gray-roi_gray.mean()/np.std(roi_gray)).astype(np.uint8)
			
			eyes = eye_cascade.detectMultiScale(roi_gray)
		if w_ref == 0 :
			for (x_eye,y_eye,w_eye,h_eye) in eyes:
				center = (int(x_eye+0.5*w_eye),int(y_eye+0.5*h_eye))
				cv2.rectangle(roi_color,(x_eye,y_eye),(x_eye+w_eye,y_eye+h_eye),(0,0,255),3)
				if len(w_ls)<40:
					w_ls.append(w_eye)
					h_ls.append(h_eye)
				else :
					time_end = time.time()
					w_ref = np.mean(w_ls)
					h_ref = np.mean(h_ls) #得到了这两个参数后w_ref的判断就进不来了
					w_ref = int(w_ref)
					h_ref = int(h_ref)
					f = open('w_h_ref.txt','w')
					f.write(f'{w_ref}\n{h_ref}')
					f.close()
		else:
			roi_bk = roi_color.copy()
			for (x_eye,y_eye,w_eye,h_eye) in eyes:
				center = (int(x_eye+0.5*w_eye),int(y_eye+0.5*h_eye))
				eye_img = roi_bk[center[1]-int(0.5*h_ref):center[1]+int(0.5*h_ref),center[0]-int(0.75*w_ref):center[0]+int(0.75*w_ref)]
				cv2.rectangle(roi_color,(x_eye,y_eye),(x_eye+w_ref,y_eye+h_ref),(0,0,255),3)
				try:
					eye_img = process(eye_img)
				except:
					continue
				prop = eye_img.sum()/eye_img.size
				if prop > 0.7 or prop<0.65:
					cv2.putText(roi_color,'open'+str(prop),center,cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
				else:
					cv2.putText(roi_color,'close'+str(prop),center,cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
				cnt+=1

				
		cv2.imshow('detecting eye',frame)
		if cv2.waitKey(1) == 27:
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	detectFace()
