import cv2
import threading
from tools import image_processing
import numpy as np
import numpy.random as npr
import math
import os,sys
sys.path.append(os.getcwd())
from config import config
import tools.image_processing as image_processing

h = np.arange(0, config.HeatMapSize, 1)
w = np.arange(0, config.HeatMapSize, 1)
ww, hh = np.meshgrid(w, h)
ww = np.tile(ww,[14,1,1])
hh = np.tile(hh,[14,1,1])
	
class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.ims, self.landmarks = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.ims, self.landmarks
        except Exception:
            return None

def get_minibatch_thread(imdb, im_size):
    num_images = len(imdb)
    processed_ims = list()
    landmark_reg_target = list()
    #print(num_images)
    for i in range(num_images):
        im,landmark = augment_for_one_image(imdb[i],im_size)
        im_tensor = image_processing.transform(im,True)
        processed_ims.append(im_tensor)
        landmark_reg_target.append(landmark)

    return processed_ims, landmark_reg_target

def get_minibatch(imdb, im_size, thread_num = 4):

    num_images = len(imdb)
    thread_num = max(1,thread_num)
    num_per_thread = math.ceil(float(num_images)/thread_num)
    #print(num_per_thread)
    threads = []
    for t in range(thread_num):
        start_idx = int(num_per_thread*t)
        end_idx = int(min(num_per_thread*(t+1),num_images))
        cur_imdb = [imdb[i] for i in range(start_idx, end_idx)]
        cur_thread = MyThread(get_minibatch_thread,(cur_imdb,im_size))
        threads.append(cur_thread)
    for t in range(thread_num):
        threads[t].start()

    processed_ims = list()
    landmark_reg_target = list()

    for t in range(thread_num):
        cur_process_ims, cur_landmark_reg_target = threads[t].get_result()
        processed_ims = processed_ims + cur_process_ims
        landmark_reg_target = landmark_reg_target + cur_landmark_reg_target    
    
    im_array = np.vstack(processed_ims)
    landmark_target_array = np.vstack(landmark_reg_target)
    
    data = {'data': im_array}
    label = {}
    label['landmark_target'] = landmark_target_array

    return data, label

def augment_for_one_image(annotation_line, size):
    annotation = annotation_line.strip().split()
    img_path = config.root+'/data/%s/'%config.landmark_img_set+annotation[0]
    #print img_path
    img = cv2.imread(img_path)
    width = img.shape[1]
    height = img.shape[0]
    bbox = np.array(annotation[1:5],dtype=np.float32)
    landmark_x = np.array(annotation[5:47:3],dtype=np.float32)
    landmark_y = np.array(annotation[6:47:3],dtype=np.float32)
    vis = np.array(annotation[7:47:3],dtype=np.float32)
    x1,y1,w,h = bbox
    x2 = x1+w
    y2 = y1+h
    cx = 0.5*(x1+x2)
    cy = 0.5*(y1+y2)

    cur_angle = npr.randint(int(config.min_rot_angle),int(config.max_rot_angle)+1)
    rot_landmark_x,rot_landmark_y = image_processing.rotateLandmark14(cx,cy,landmark_x,landmark_y, cur_angle,1)
    cur_size_w = int(npr.randint(10, 16)*0.1*w)
    cur_size_h = int(npr.randint(10, 16)*0.1*h)
    
    # delta here is the offset of box center
    delta_x = npr.randint(-int(cur_size_w * 0.20), int(cur_size_w * 0.20)+1)
    delta_y = npr.randint(-int(cur_size_h * 0.20), int(cur_size_h * 0.20)+1)
				
    nx1 = int(max(x1 + cur_size_w / 2 + delta_x - cur_size_w / 2, 0))
    ny1 = int(max(y1 + cur_size_h / 2 + delta_y - cur_size_h / 2, 0))
    nx2 = nx1 + cur_size_w
    ny2 = ny1 + cur_size_h
    nx1 = max(0,nx1)
    ny1 = max(0,ny1)
    nx2 = min(width, nx2)
    ny2 = min(height,ny2)
    cur_size_w = int(nx2 - nx1)
    cur_size_h = int(ny2 - ny1)
    cur_size = max(cur_size_w, cur_size_h)
	
    cropped_im = np.zeros([cur_size,cur_size,3])

    pad_x = cur_size - cur_size_w
    pad_y = cur_size - cur_size_h
    pad_x_left = int(pad_x*0.5)
    pad_y_up = int(pad_y*0.5)
        
    rot_img,_,_ = image_processing.rotateWithLandmark14(img,cx,cy,landmark_x,landmark_y, cur_angle,1)   
    cropped_im[pad_y_up:(pad_y_up+cur_size_h), pad_x_left:(pad_x_left+cur_size_w), :] = rot_img[ny1 : ny2, nx1 : nx2, :]
    resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)
	
    for i in range(vis.shape[0]):
        if vis[i] == 1:
            if rot_landmark_x[i] < nx1 or rot_landmark_x[i] > nx1 or rot_landmark_y[i] < ny1 or rot_landmark_y[i] > ny2:
                vis[i] = 0
    
    offset_x = (rot_landmark_x - nx1 + pad_x_left +0.5)/float(cur_size)
    offset_y = (rot_landmark_y - ny1 + pad_y_up + 0.5)/float(cur_size)
        

	
    heatmap = landmark_to_heatmap(offset_x, offset_y, vis, config.HeatMapSigma)
    
    
    if config.enable_blur:
        kernel_size = npr.randint(-5,13)*2+1
        if kernel_size >= 3:
            blur_im = cv2.GaussianBlur(resized_im,(kernel_size,kernel_size),0)
            resized_im = blur_im
    
    return resized_im,heatmap
	
def landmark_to_heatmap(landmark_x, landmark_y, vis, sigma):
    sigma2 = sigma*sigma
    heatmap = np.empty([15*config.HeatMapStage,config.HeatMapSize,config.HeatMapSize],dtype=np.float32)
    
    cx = landmark_x*config.HeatMapSize-0.5
    cy = landmark_y*config.HeatMapSize-0.5
    cx = cx[:,np.newaxis,np.newaxis]
    cy = cy[:,np.newaxis,np.newaxis]
    cx = np.tile(cx,[1,config.HeatMapSize,config.HeatMapSize])
    cy = np.tile(cy,[1,config.HeatMapSize,config.HeatMapSize])
    vv = vis[:,np.newaxis,np.newaxis]
    vv = np.tile(vv,[1,config.HeatMapSize,config.HeatMapSize])
    
    ww1 = ww - cx
    hh1 = hh - cy
    dis2 = ww1**2+hh1**2
        
    heatmap[0:14,:,:] = np.exp(-dis2/sigma2)*vv
    heatmap[14,:,:] = 1 - np.max(heatmap[0:14,:,:],axis=0)  
    for i in range(config.HeatMapStage-1):
        heatmap[15*(i+1):15*(i+2),:,:] = heatmap[0:15,:,:]	
    return heatmap.flatten()
  
 
