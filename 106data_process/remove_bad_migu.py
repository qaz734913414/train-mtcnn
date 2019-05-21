
import os
import cv2
import numpy as np

def is_in_bad_lists(bad_lists, cur_str):
    num_bad = len(bad_lists)
    flag = False
    for i in range(num_bad):
        if cur_str == bad_lists[i]:
            flag = True
            break
    return flag

if __name__ == '__main__':
    with open("part3.txt", "r") as f:
        lines = f.readlines()
		
    with open("bad0.txt","r") as f:
        bad_lines = f.readlines()
		
    num_bad = len(bad_lines)
    bad_lists = list()
    for i in range(num_bad):
        id = int(bad_lines[i].split('\n')[0])
        bad_lists.append('%06d'%id)
		
    num_lines = len(lines)
    with open("part3_clean.txt","w") as f:
        for i in range(num_lines):
            cur_line = lines[i]
            cur_file = cur_line.split()[0]
            cur_splits = cur_file.split('/')
            split_num = len(cur_splits)
            real_filename = cur_splits[split_num-1].split('.jpg')[0]
            annotation = cur_line.split()
            landmark = np.array(annotation[1:213],dtype=np.float32)
            landmark_x = landmark[0::2]
            landmark_y = landmark[1::2]
            max_x = max(landmark_x)
            min_x = min(landmark_x)
            max_y = max(landmark_y)
            min_y = min(landmark_y)
            h = max_x-min_x
            w = max_y-min_y
            bbox_size = max(h,w)
            if bbox_size >= 96:
                if not is_in_bad_lists(bad_lists,real_filename):
                    f.write(cur_line)