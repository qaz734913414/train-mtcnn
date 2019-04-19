import cv2
import numpy as np


def get_bbox(landmark):
    x_coords = landmark[0::2]
    y_coords = landmark[1::2]
    x1 = min(x_coords)
    x2 = max(x_coords)
    y1 = min(y_coords)
    y2 = max(y_coords)
    return x1,y1, x2-x1, y2-y1
	
img_path = '../migu_106points/'
out_img_path = '../migu_106points_cut/'

with open('../migu_106points/cut_landmarks_migu.txt', 'r') as f:
    landmark_lines = f.readlines()
f = open('../migu_106points_cut/cut_landmarks_migu.txt','w')
num = len(landmark_lines)
for i in range(num):
    line = landmark_lines[i].split('\n')[0]
    landmark_splits = line.split()
    img = cv2.imread(img_path+'/'+landmark_splits[0])
    width = img.shape[1]
    height = img.shape[0]
    landmark = np.array(landmark_splits[1:213],dtype=np.float32)
    x1,y1,w,h = get_bbox(landmark)
    #print([x1,y1,w,h])
    off_x = int(max(0,x1-0.8*w))
    off_y = int(max(0,y1-0.5*h))
    max_x = int(min(width,x1+w*1.8))
    max_y = int(min(height,y1+h*1.5))
    bbox = np.empty([4],dtype=np.float32)
    bbox_ori = np.array([x1,y1,w,h],dtype=np.float32)
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
        out_img_name = landmark_splits[0]
        #print out_img_name
        cv2.imwrite(out_img_path+'/'+out_img_name,cut_img)
        line = out_img_name+' '
        for idx in range(106):
            line = line + '%.1f %.1f '%(landmark[idx*2],landmark[idx*2+1])
        line = line + '\n'
        f.write(line)
    if (i+1)%100 == 0:
        print i+1
	
f.close()