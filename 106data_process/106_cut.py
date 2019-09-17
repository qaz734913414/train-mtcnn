import cv2
import numpy as np

img_path = '../data/'
out_img_path = '../data_cut/'

with open('../box_landmark.txt', 'r') as f:
    landmark_lines = f.readlines()
f = open('../cut_landmarks.txt','w')
num = len(landmark_lines)
for i in range(num):
    line = landmark_lines[i].split('\n')[0]
    landmark_splits = line.split()
    img = cv2.imread(img_path+'/'+landmark_splits[0])
    width = img.shape[1]
    height = img.shape[0]
    box = np.array(landmark_splits[1:5],dtype=np.float32)
    x1 = box[2]
    y1 = box[1]
    h = box[0]
    w = box[3]
    bbox = box.copy()
    bbox[0] = x1
    bbox[1] = y1
    bbox[2] = w
    bbox[3] = h
    landmark = np.array(landmark_splits[5:217],dtype=np.float32)
    bbox_ori = bbox.copy()
    off_x = int(max(0,x1-1.5*w))
    off_y = int(max(0,y1-1.5*h))
    max_x = int(min(width,x1+w*2.5))
    max_y = int(min(height,y1+h*2.5))
    bbox[0] = max(0,x1)
    bbox[1] = max(0,y1)	
    bbox[2] = x1+w - bbox[0]
    bbox[3] = y1+h - bbox[1]
    bbox_ori[0] -= off_x
    bbox_ori[1] -= off_y
    for j in range(106):
        landmark[j*2] -= off_x
        landmark[j*2+1] -= off_y
    if bbox[2] >= 20 and bbox[3] >= 20 and bbox[0]+bbox[2] <= width and bbox[1]+bbox[3] <= height:
        cut_width = max_x-off_x
        cut_height = max_y-off_y
        cut_img = img[off_y:max_y,off_x:max_x,:]
        if cut_width > 200 and cut_height > 200:
            scale = 200.0/min(cut_width,cut_height)
            dst_width = int(cut_width*scale)
            dst_height = int(cut_height*scale)
            cut_img = cv2.resize(cut_img,(dst_width,dst_height),interpolation=cv2.INTER_LINEAR)
            bbox_ori *= scale
            landmark *= scale
        out_img_name = 'face%05d_%s'%(i,landmark_splits[0])
        cv2.imwrite(out_img_path+'/'+out_img_name,cut_img)
        line = out_img_name+' '
        for idx in range(106):
            line = line + '%.1f %.1f '%(landmark[idx*2],landmark[idx*2+1])
        line = line + '\n'
        f.write(line)
    if (i+1)%100 == 0:
        print i+1
	
f.close()