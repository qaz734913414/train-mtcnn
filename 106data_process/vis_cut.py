"""
Image visualization result for 106 landmarks
Created by Jacky LUO
Usage: python vis.py 
"""
import os
import cv2
import numpy as np

base_path = '../data_cut/'
save_path = '../vis_cut/'

if __name__ == '__main__':
    with open("../cut_landmarks.txt") as f:
        lines = f.readlines()
        line_id = 0
        for line in lines:
            line = line.strip().split()
            # image_path, 106-landmarks
            image_path = line[0]
            landmarks = np.array(line[1:],dtype=np.float32)

            # visualization
            img = cv2.imread(os.path.join(base_path, image_path))
            h = img.shape[0]
            w = img.shape[1]
            
            for i in range(106):
                cv2.circle(img, (int(landmarks[i*2]), int(landmarks[i*2+1])), 2, (255, 0, 0), -1)
                cv2.imwrite(os.path.join(save_path, image_path),img)
            line_id = line_id+1
            if line_id%100 == 0
                print line_id			