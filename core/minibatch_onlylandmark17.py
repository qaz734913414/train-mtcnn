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
    thread_num = max(2,thread_num)
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
    landmark = np.array(annotation[5:39],dtype=np.float32)
    landmark_x = landmark[0::2]
    landmark_y = landmark[1::2]
    
    x1,y1,w,h = bbox
    x2 = x1 + w
    y2 = y1 + h	
    x1 += npr.randint(-100,101)*0.001*w
    y1 += npr.randint(-100,101)*0.001*h
    x2 += npr.randint(-100,101)*0.001*w
    y2 += npr.randint(-100,101)*0.001*h
    ny1 = int(max(0,y1))
    ny2 = int(min(height,y2))
    nx1 = int(max(0,x1))
    nx2 = int(min(width,x2))
    w = nx2-nx1
    h = ny2-ny1
    cur_size = max(w,h)
    pad_w = cur_size - w
    pad_h = cur_size - h
    pad_w_left = int(0.5*pad_w)
    pad_h_up = int(0.5*pad_h)
    cropped_im = np.zeros((cur_size,cur_size,3))

    #print ny1,ny2,nx1,nx2
    cropped_im[pad_h_up:(pad_h_up+h),pad_w_left:(pad_w_left+w),:] = img[ny1 : ny2, nx1 : nx2, :]
    resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)
    offset_x = (landmark_x - nx1 + pad_w_left +0.5)/float(cur_size)
    offset_y = (landmark_y - ny1 + pad_h_up +0.5)/float(cur_size)
    
    
    landmark[0::2] = offset_x
    landmark[1::2] = offset_y
    
    
    if config.enable_blur:
        #kernel_size = npr.randint(-5,4)*2+1
        kernel_size = npr.randint(-5,13)*2+1
        if kernel_size >= 3:
            blur_im = cv2.GaussianBlur(resized_im,(kernel_size,kernel_size),0)
            resized_im = blur_im
    
    return resized_im,landmark