
import os
import cv2
import numpy as np

base_path = '../data_cut/'
save_path = '../clean0/'
vis_save_path = '../clean0_vis/'
err_file = '../Error_list0.txt'
min_size_thresh = 60
def load_error_lists():
    with open(err_file) as f:
        err_lines = f.readlines()
    err_idx = np.array(err_lines,dtype=np.int32)
    err_idx.sort()
    return err_idx
	
def is_in_set(set, val):
    num = len(set)
    if num == 0:
        return False
    low, high = 0, num-1
    while low <= high:
        mid = int((low+high)/2)
        #print low,high,mid
        if set[mid] == val:
            return True
        elif set[mid] > val:
            high = mid - 1
        else:
            low = mid + 1
			
    return False
        	
if __name__ == '__main__':
    err_idx = load_error_lists()
    #print err_idx
    out_f = open('../clean0_landmarks.txt','w')
    print 'hello1'
    with open("../cut_landmarks.txt") as f:
        lines = f.readlines()
        line_id = 0
		
        for line in lines:
            line_id = line_id+1
            if line_id%10 == 0:
                print line_id
            line = line.strip().split()
            # image_path, 106-landmarks
            image_path = line[0]
            image_id = np.array(image_path.split('_')[0][4:],dtype=np.int32)
            if is_in_set(err_idx,image_id):
                continue
            #print image_id	
            landmarks = np.array(line[1:],dtype=np.float32)
            min_x = min(landmarks[0::2])
            max_x = max(landmarks[0::2])
            min_y = min(landmarks[1::2])
            max_y = max(landmarks[1::2])
            if max_x - min_x < min_size_thresh and max_y-min_y < min_size_thresh:
                continue

            # visualization
            img = cv2.imread(os.path.join(base_path, image_path))
            h = img.shape[0]
            w = img.shape[1]
			
            if min_x < 0 or min_y < 0 or max_x > w or max_y > h:
                continue
            
            out_img_name = 'face%05d_.jpg'%image_id
            cv2.imwrite(os.path.join(save_path, out_img_name),img)
			
            for i in range(106):
                cv2.circle(img, (int(landmarks[i*2]), int(landmarks[i*2+1])), 2, (255, 0, 0), -1)
                cv2.imwrite(os.path.join(vis_save_path, out_img_name),img)
				
            line = out_img_name+' '
            for idx in range(106):
                line = line + '%.1f %.1f '%(landmarks[idx*2],landmarks[idx*2+1])
            line = line + '\n'
            out_f.write(line)

	
    