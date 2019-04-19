"""
Image visualization result for 106 landmarks
Created by Jacky LUO
Usage: python vis.py 
"""
import os
import cv2
import numpy as np
   	
if __name__ == '__main__':

    out_f = open('../106data_merge_migu.txt','w')
   
    map_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,
                42,43,44,45,46,51,52,53,54,58,59,60,61,62,66,67,69,70,71,73,75,76,78,79,80,82,
	            41,40,39,38,50,49,48,47,68,72,74,77,81,83,55,65,56,64,57,63,
				84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105]
    print len(map_idx)
    with open("../106data_merge.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            # image_path, 106-landmarks
            image_path = line[0]
            landmarks = np.array(line[1:],dtype=np.float32)
            
            line = image_path+' '

            for i in range(106):
                idx = int(map_idx[i])
                line = line + '%.1f %.1f '%(landmarks[idx*2],landmarks[idx*2+1])
            line = line + '\n'
            out_f.write(line)

	
    