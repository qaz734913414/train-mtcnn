
import os
import cv2
import numpy as np
   	
if __name__ == '__main__':

    out_f = open('../clean0_landmarks_jd.txt','w')
   
    map_idx = [1,9,10,11,12,13,14,15,16,2,3,4,5,6,7,8,0,24,23,22,21,20,19,18,32,31,30,29,28,27,26,25,17,
	            43,48,49,51,50,46,47,45,44,
				102,103,104,105,101,100,99,98,97,
				72,73,74,86,75,76,77,78,79,80,85,84,83,82,81,
				35,41,40,42,39,37,33,36,34,
				89,95,94,96,93,91,87,90,88,
				52,64,63,71,67,68,61,58,59,53,56,55,
				65,66,62,70,69,57,60,54,
				38,92]
    print len(map_idx)
    with open("../clean0_landmarks.txt") as f:
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

	
    