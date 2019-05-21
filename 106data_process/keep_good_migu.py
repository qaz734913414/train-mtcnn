
import os
import cv2
import numpy as np

def is_in_good_lists(good_lists, cur_str):
    num_good = len(good_lists)
    flag = False
    for i in range(num_good):
        if cur_str == good_lists[i]:
            flag = True
            break
    return flag

if __name__ == '__main__':
    with open("migu_part.txt", "r") as f:
        lines = f.readlines()
		
    with open("keep_info.txt","r") as f:
        keep_lines = f.readlines()
    
    num_keep = len(keep_lines)
    good_lists = list()
    for i in range(num_keep):
        cur_line = keep_lines[i].split('\n')[0]
        cur_line = cur_line.replace('.jpg_','.jpg ')
        cur_line = cur_line.replace('.jpeg_','.jpeg ')
        cur_line = cur_line.replace('.JPG_','.JPG ')
        cur_line = cur_line.replace('.JPEG_','.JPEG ')
        splits = cur_line.split()
        cur_file = splits[0]
        flag = int(splits[1])
        if flag == 1:
            good_lists.append(cur_file)
		
    num_lines = len(lines)
    with open("migu_part_clean.txt","w") as f:
        for i in range(num_lines):
            cur_line = lines[i]
            cur_file = cur_line.split()[0]
            
            if is_in_good_lists(good_lists,cur_file):
                f.write(cur_line)