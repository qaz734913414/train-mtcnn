
import os
import cv2
import numpy as np

base_path = '../data/'
scale = int(5)
if __name__ == '__main__':
    with open("../box_landmark.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            # image_path, box, 106-landmarks
            image_path = line[0]
            b_box = line[1: 5]
            landmarks = line[5:]

            # visualization
            img = cv2.imread(os.path.join(base_path, image_path))
            h = img.shape[0]
            w = img.shape[1]
            img =cv2.resize(img,(w*scale,h*scale))
            print scale
            off_x = int(b_box[2])*scale
            off_y = int(b_box[1])*scale
            rect_h = int(b_box[0])*scale
            rect_w = int(b_box[3])*scale
            print off_x,off_y,rect_w,rect_h
            cv2.rectangle(img, (int(off_x), int(off_y)),
                          (int(off_x+rect_w), int(off_y+rect_h)), (0, 0, 255), 2)
            landmarks = np.reshape(landmarks, (-1, 2))
            idx = 0
            font = cv2.FONT_HERSHEY_SIMPLEX
            for (x, y) in landmarks:
                cv2.circle(img, (int(x)*scale, int(y)*scale), 2, (255, 0, 0), -1)
                cv2.putText(img, '%d'%idx, (int(x)*scale, int(y)*scale), font, 0.8, (0, 255, 0), 2)
                idx = idx+1
            # cv2.imwrite("image.png", img)
            cv2.imshow("image", img)
            cv2.imwrite('save.jpg',img)
            cv2.waitKey(0)
