import cv2
import config as cfg
import os
import numpy as np
import xml.etree.ElementTree as ET
flag=0
epoch=0
cell_size=7
img_size=448
batch_size=1
annotation_pwd='/home/ubuntu/VOCdevkit/VOC2007/Annotations'
img_pwd='/home/ubuntu/VOCdevkit/VOC2007/JPEGImages'
CLASS_dict=dict(zip(cfg.CLASSES,range(len(cfg.CLASSES))))


def get_annotation(index):
	'''from annotations get xml bbox and classes  '''
	img_name = os.path.join(img_pwd, index + '.jpg')
        img = cv2.imread(img_name)
        h_ratio = 1.0 * img_size / img.shape[0]
        w_ratio = 1.0 * img_size / img.shape[1]

        label = np.zeros((cell_size,cell_size, 25))
        img_xml = os.path.join(annotation_pwd, index + '.xml')
        tree = ET.parse(img_xml)
        objs = tree.findall('object') 

	for obj in objs:
	    bbox=obj.find('bndbox')
	    x1=float(bbox.find('xmin').text)*w_ratio
	    x2=float(bbox.find('xmax').text)*w_ratio
	    y1=float(bbox.find('ymin').text)*h_ratio
            y2=float(bbox.find('ymax').text)*h_ratio
 	    cls_indx=CLASS_dict[obj.find('name').text]

	    box=[(x1+x2)/2.0,(y1+y2)/2.0,x2-x1,y2-y1]
            x_box_indx=int(box[0]*cell_size/img_size)
	    y_box_indx=int(box[1]*cell_size/img_size)
	    if  label[y_box_indx,x_box_indx,0]==1:
		   continue

	    label[y_box_indx,x_box_indx,0]=1
	    label[y_box_indx,x_box_indx,1:5]=box
	    label[y_box_indx,x_box_indx,5+cls_indx]=1
	return label,len(objs)

 
def all_imgs_pre(img_name):
	img=cv2.imread(img_name)
	img=cv2.resize(img,(img_size,img_size))
	img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
	img=(img/225.0)*2.0-1.0
	return img

def get_batch_input():
	ALL_labels_list=get_ALL_labels_list()
	global epoch,flag
	imgs_input=np.zeros((batch_size,img_size,img_size,3))
	labels_input=np.zeros((batch_size,cell_size,cell_size,25))
	count=0
	while count<batch_size:
		im_name=ALL_labels_list[flag]['img__name']
		imgs_input[count,:,:,:]=all_imgs_pre(im_name)
		labels_input[count,:,:,:]=ALL_labels_list[flag]['labelground']
		count+=1
		flag+=1
		if flag>=len(ALL_labels_list):
		      np.random.shuffle(ALL_labels_list)
		      flag=0
		      epoch+=1
	return imgs_input,labels_input


def get_ALL_labels_list():
	textname='/home/ubuntu/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
        with open(textname,'r') as f:
              img_index=[x.strip() for x in f.readlines()]
        ALL_labels_list=[]
        for index in img_index:
             label,num=get_annotation(index)
             if num==0:
                continue
             img_name=os.path.join(img_pwd,index+'.jpg')
             ALL_labels_list.append({'img__name':img_name,'labelground':label})
	return ALL_labels_list

if __name__=='__main__':
	a,b=get_batch_input()
	print a,b
