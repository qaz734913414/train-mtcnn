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
    img_path = annotation[0] + '.jpg'
    #print img_path
    img = cv2.imread(img_path)
    landmark = np.array(annotation[1:213],dtype=np.float32)
    
    return img,landmark