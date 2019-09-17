
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
    with open("migu_true.txt", "r") as f:
        lines = f.readlines()
		
    num_lines = len(lines)
    with open("migu_part.txt","w") as f:
        for i in range(num_lines):
            part_id = int(i / 10000)
            cur_line = lines[i]
            splits = cur_line.split()
            cur_file = splits[0]
            out_file = 'migu_part%d'%part_id+'/%06d.jpg'%i
            im = cv2.imread(cur_file)
            cv2.imwrite(out_file,im)
            f.write(out_file)
            for j in range(212):
                f.write(' '+splits[j+1])
            f.write('\n')