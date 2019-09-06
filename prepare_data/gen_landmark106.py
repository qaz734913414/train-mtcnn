import numpy as np
import cv2
import threading
import argparse
import math
import os,sys
import numpy.random as npr
from utils import IoU
sys.path.append(os.getcwd())
from config import config
import tools.image_processing as image_processing

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.landmark_names = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.landmark_names
        except Exception:
            return None

def gen_landmark_minibatch_thread(size, start_idx, annotation_lines, imdir, landmark_save_dir, base_num):
    num_images = len(annotation_lines)
    landmark_names = list()
    for i in range(num_images):
        cur_annotation_line = annotation_lines[i].strip().split()
        im_path = cur_annotation_line[0]
        landmarks = np.array(cur_annotation_line[1:213],dtype=np.float32)
        img = cv2.imread(os.path.join(imdir, im_path))
        cur_landmark_names = gen_landmark_for_one_image(size, start_idx+i, img, landmark_save_dir, landmarks, base_num)
        landmark_names = landmark_names + cur_landmark_names


    return landmark_names


def gen_landmark_minibatch(size, start_idx, annotation_lines, imdir, landmark_save_dir, base_num, thread_num = 4):
    num_images = len(annotation_lines)
    num_per_thread = math.ceil(float(num_images)/thread_num)
    threads = []
    for t in range(thread_num):
        cur_start_idx = int(num_per_thread*t)
        cur_end_idx = int(min(num_per_thread*(t+1),num_images))
        cur_annotation_lines = annotation_lines[cur_start_idx:cur_end_idx]
        cur_thread = MyThread(gen_landmark_minibatch_thread,(size, start_idx+cur_start_idx, cur_annotation_lines,
                                                        imdir, landmark_save_dir, base_num))
        threads.append(cur_thread)
    for t in range(thread_num):
        threads[t].start()

    landmark_names = list()
    
    for t in range(thread_num):
        cur_landmark_names = threads[t].get_result()
        landmark_names = landmark_names + cur_landmark_names

    return landmark_names
	

def gen_landmark_for_one_image(size, idx, img, landmark_save_dir, landmarks, base_num = 1):
    landmark_names = list()
    landmark_num = 0
    
    width = img.shape[1]
    height = img.shape[0]
    landmark_x = landmarks[0::2]
    landmark_y = landmarks[1::2]
    max_x = max(landmark_x)
    min_x = min(landmark_x)
    max_y = max(landmark_y)
    min_y = min(landmark_y)
    cx = 0.5*(max_x+min_x)
    cy = 0.5*(max_y+min_y)
    w = max_x-min_x
    h = max_y-min_y
    bbox_size = max(h,w)
    x1 = int(cx - bbox_size*0.5)
    y1 = int(cy - bbox_size*0.5)
    w = bbox_size
    h = bbox_size
    
    init_rot = 0
    if config.landmark106_migu_init_rot:
        eye_cx = 0.25*(landmark_x[52]+landmark_x[55]+landmark_x[58]+landmark_x[61])
        eye_cy = 0.25*(landmark_y[52]+landmark_y[55]+landmark_y[58]+landmark_y[61])
        mouth_cx = 0.25*(landmark_x[84]+landmark_x[96]+landmark_x[100]+landmark_x[90])
        mouth_cy = 0.25*(landmark_y[84]+landmark_y[96]+landmark_y[100]+landmark_y[90])
        dir_x = mouth_cx - eye_cx
        dir_y = mouth_cy - eye_cy
        init_rot = 90 - math.atan2(dir_y, dir_x)/math.pi*180
		
    cur_angle = npr.randint(int(config.min_rot_angle - init_rot),int(config.max_rot_angle - init_rot)+1)
    try_num = 0
    force_accept = 0
	
    while landmark_num < base_num:
        try_num += 1
        if try_num > base_num*1000:
            force_accept = 1
            break
        rot_landmark_x,rot_landmark_y = image_processing.rotateLandmark106(cx,cy,landmark_x,landmark_y, cur_angle,1)
        rot_max_x = max(rot_landmark_x)
        rot_min_x = min(rot_landmark_x)
        rot_max_y = max(rot_landmark_y)
        rot_min_y = min(rot_landmark_y)
        rot_cx = 0.5*(rot_max_x+rot_min_x)
        rot_cy = 0.5*(rot_max_y+rot_min_y)
        rot_w = rot_max_x-rot_min_x
        rot_h = rot_max_y-rot_min_y
        rot_bbox_size = max(rot_h,rot_w)
        rot_x1 = int(rot_cx - rot_bbox_size*0.5)
        rot_y1 = int(rot_cy - rot_bbox_size*0.5)
        rot_w = rot_bbox_size
        rot_h = rot_bbox_size
        #cur_size = int(npr.randint(10, 21)*0.1*rot_bbox_size)
        #cur_size = int(npr.randint(10, 16)*0.1*rot_bbox_size)
        cur_size = int(npr.randint(110, 126)*0.01*rot_bbox_size)
        #up_border_size = int(-cur_size*0.15)
        #down_border_size = int(-cur_size*0.15)
        #left_border_size = int(-cur_size*0.15)
        #right_border_size = int(-cur_size*0.15)
        up_border_size = int(cur_size*0.05)
        down_border_size = int(cur_size*0.05)
        left_border_size = int(cur_size*0.05)
        right_border_size = int(cur_size*0.05)

        # delta here is the offset of box center
        #delta_x = npr.randint(-int(rot_w * 0.35), int(rot_w * 0.35)+1)
        #delta_y = npr.randint(-int(rot_h * 0.35), int(rot_h * 0.35)+1)
        #delta_x = npr.randint(-int(rot_w * 0.20), int(rot_w * 0.20)+1)
        #delta_y = npr.randint(-int(rot_h * 0.20), int(rot_h * 0.20)+1)
        delta_x = npr.randint(-int(rot_w * 0.02), int(rot_w * 0.02)+1)
        delta_y = npr.randint(-int(rot_h * 0.02), int(rot_h * 0.02)+1)
		
		
        nx1 = int(max(x1 + rot_w / 2 + delta_x - cur_size / 2, 0))
        ny1 = int(max(y1 + rot_h / 2 + delta_y - cur_size / 2, 0))
        nx2 = nx1 + cur_size
        ny2 = ny1 + cur_size

        if nx2 > width or ny2 > height:
            continue
        ignore = 0
        max_rot_landmark_x = max(rot_landmark_x)
        min_rot_landmark_x = min(rot_landmark_x)
        max_rot_landmark_y = max(rot_landmark_y)
        min_rot_landmark_y = min(rot_landmark_y)
        if min_rot_landmark_x < nx1+left_border_size or max_rot_landmark_x >= nx1 + cur_size-right_border_size:
            ignore = 1
        if min_rot_landmark_y < ny1+up_border_size or max_rot_landmark_y >= ny1 + cur_size-down_border_size:
            ignore = 1
												
        if ignore == 1:
            continue
        landmark_x_dis = max_rot_landmark_x - min_rot_landmark_x
        landmark_y_dis = max_rot_landmark_y - min_rot_landmark_y
        tmp_dis = landmark_x_dis*landmark_x_dis + landmark_y_dis*landmark_y_dis
        #if tmp_dis < 0.64*cur_size*cur_size:
        if tmp_dis < 1.00*cur_size*cur_size:
            continue
			
        offset_x = (rot_landmark_x - nx1+0.5)/float(cur_size)
        offset_y = (rot_landmark_y - ny1+0.5)/float(cur_size)
        
        rot_img,_,_ = image_processing.rotateWithLandmark106(img,cx,cy,landmark_x,landmark_y, cur_angle,1)
        cropped_im = rot_img[ny1 : ny2, nx1 : nx2, :]
        resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)
        
        if config.landmark106_migu_random_flip:
            flip_val = npr.randint(0,2)
            if flip_val == 1:
                tmp_arr1_x, tmp_arr1_y = offset_x[0:33], offset_y[0:33]
                offset_x[0:33], offset_y[0:33] = 1 - tmp_arr1_x[::-1], tmp_arr1_y[::-1]
                tmp_arr1_x, tmp_arr1_y = offset_x[33:43], offset_y[33:43]
                offset_x[33:43], offset_y[33:43] = 1 - tmp_arr1_x[::-1], tmp_arr1_y[::-1]
                tmp_arr1_x, tmp_arr1_y = offset_x[43:47], offset_y[43:47]
                offset_x[43:47], offset_y[43:47] = 1 - tmp_arr1_x, tmp_arr1_y
                tmp_arr1_x, tmp_arr1_y = offset_x[47:52], offset_y[47:52]
                offset_x[47:52], offset_y[47:52] = 1 - tmp_arr1_x[::-1], tmp_arr1_y[::-1]
                tmp_arr1_x, tmp_arr1_y, tmp_arr2_x, tmp_arr2_y = offset_x[52:56], offset_y[52:56], offset_x[58:62], offset_y[58:62]
                offset_x[52:56], offset_y[52:56], offset_x[58:62], offset_y[58:62] = 1 - tmp_arr2_x[::-1], tmp_arr2_y[::-1], 1 - tmp_arr1_x[::-1], tmp_arr1_y[::-1]
                tmp_arr1_x, tmp_arr1_y, tmp_arr2_x, tmp_arr2_y = offset_x[56:58], offset_y[56:58], offset_x[62:64], offset_y[62:64]
                offset_x[56:58], offset_y[56:58], offset_x[62:64], offset_y[62:64] = 1 - tmp_arr2_x[::-1], tmp_arr2_y[::-1], 1 - tmp_arr1_x[::-1], tmp_arr1_y[::-1]
                tmp_arr1_x, tmp_arr1_y = offset_x[64:72], offset_y[64:72]
                offset_x[64:72], offset_y[64:72] = 1 - tmp_arr1_x[::-1], tmp_arr1_y[::-1]
                tmp_arr1_x, tmp_arr1_y, tmp_arr2_x, tmp_arr2_y = offset_x[72:75], offset_y[72:75], offset_x[75:78], offset_y[75:78]
                offset_x[72:75], offset_y[72:75], offset_x[75:78], offset_y[75:78] = 1 - tmp_arr2_x, tmp_arr2_y, 1 - tmp_arr1_x, tmp_arr1_y
                offset_x[78], offset_y[78], offset_x[79], offset_y[79] = 1 - offset_x[79], offset_y[79], 1 - offset_x[78], offset_y[78]
                offset_x[80], offset_y[80], offset_x[81], offset_y[81] = 1 - offset_x[81], offset_y[81], 1 - offset_x[80], offset_y[80]
                offset_x[82], offset_y[82], offset_x[83], offset_y[83] = 1 - offset_x[83], offset_y[83], 1 - offset_x[82], offset_y[82]
                tmp_arr1_x, tmp_arr1_y = offset_x[84:91], offset_y[84:91]
                offset_x[84:91], offset_y[84:91] = 1 - tmp_arr1_x[::-1], tmp_arr1_y[::-1]
                tmp_arr1_x, tmp_arr1_y = offset_x[91:96], offset_y[91:96]
                offset_x[91:96], offset_y[91:96] = 1 - tmp_arr1_x[::-1], tmp_arr1_y[::-1]
                tmp_arr1_x, tmp_arr1_y = offset_x[96:101], offset_y[96:101]
                offset_x[96:101], offset_y[96:101] = 1 - tmp_arr1_x[::-1], tmp_arr1_y[::-1]
                tmp_arr1_x, tmp_arr1_y = offset_x[101:104], offset_y[101:104]
                offset_x[101:104], offset_y[101:104] = 1 - tmp_arr1_x[::-1], tmp_arr1_y[::-1]
                offset_x[104], offset_y[104], offset_x[105], offset_y[105] = 1 - offset_x[105], offset_y[105], 1 - offset_x[104], offset_y[104]
                resized_im = resized_im[:,::-1,:]
        
        
        if config.enable_black_border:
            black_size = npr.randint(0,int(size*0.5))
            if npr.randint(0,2) == 0:
                resized_im[:,0:black_size,:] = 128
            else:
                resized_im[:,(size-black_size):size,:] = 128

        save_file = '%s/%d_%d.jpg'%(landmark_save_dir,idx,landmark_num)
        if cv2.imwrite(save_file, resized_im):
            line = '%s/%d_%d'%(landmark_save_dir,idx,landmark_num)
            for j in range(106):
                line = line + ' %.7f %.7f'%(offset_x[j],offset_y[j])
            landmark_names.append(line)
            landmark_num += 1

    return landmark_names

def gen_landmark(image_set, size=20, base_num = 1, thread_num = 4):
    anno_file = "%s/data/mtcnn/imglists/%s.txt"%(config.root,image_set)
    imdir = "%s/data/%s"%(config.root,config.landmark_img_set)
    landmark_save_dir = "%s/prepare_data/%d/landmark106"%(config.root,size)
    
    save_dir = "%s/prepare_data/%d"%(config.root,size)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(landmark_save_dir):
        os.mkdir(landmark_save_dir)
    f1 = open(os.path.join(save_dir, 'landmark106.txt'), 'w')

    with open(anno_file, 'r') as f:
        annotation_lines = f.readlines()
    
    num = len(annotation_lines)
    print "%d pics in total" % num
    batch_size = thread_num*10
    landmark_num = 0
    start_idx = 0
    while start_idx < num:
        end_idx = min(start_idx+batch_size,num)
        cur_annotation_lines = annotation_lines[start_idx:end_idx]
        landmark_names = gen_landmark_minibatch(size, start_idx, cur_annotation_lines,
                                            imdir, landmark_save_dir, base_num, thread_num)
        cur_landmark_num = len(landmark_names)
        for i in range(cur_landmark_num):
            f1.write(landmark_names[i]+'\n')
        landmark_num += cur_landmark_num
        start_idx = end_idx
        print '%s images done, landmark106: %d'%(end_idx,landmark_num)

    f1.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train proposal net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_set', dest='image_set', help='training set',
                        default='106data_merge', type=str)
    parser.add_argument('--size', dest='size', help='112, 96, 80, 64', default='112', type=str)
    parser.add_argument('--base_num', dest='base_num', help='base num', default='1', type=str)
    parser.add_argument('--thread_num', dest='thread_num', help='thread num', default='4', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    gen_landmark(args.image_set, int(args.size), int(args.base_num), int(args.thread_num))