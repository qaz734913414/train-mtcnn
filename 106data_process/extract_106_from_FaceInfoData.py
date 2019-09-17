import os
import json
import cv2
import random
import shutil

def extract_106(in_file, out_file):
    with open(out_file, 'w') as fk:
        data_file = open(in_file, 'r')
        lines = data_file.readlines()
        print('num_lines = %d\n'%(len(lines)))
        for line in lines:
            line = line.replace('\n', '')
            data = json.loads(line)
            img = data['filePath']
            write_line = img
            faces = data['stMobile106']
            if len(faces) > 1:
                continue
            for face in faces:
                points_106 = face['face106']['pointsArray']
                for point in points_106:
                    px = float(point['x'])
                    py = float(point['y'])
                    npx = str('%.1f'%float(px))
                    npy = str('%.1f'%float(py))
                    write_line += ' ' + npx + ' ' + npy
            
            write_line += '\n'
            fk.write(write_line)
        data_file.close()


if __name__ == '__main__':
    cwd = os.getcwd() 
    #print(cwd)
    in_file = 'D:/BaiduYunDownload/part20_FaceInfoData.txt'
    out_file = 'celeba_part20.txt'
    extract_106(in_file, out_file)

