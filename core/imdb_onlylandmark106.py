import os
import numpy as np

class IMDB(object):
    def __init__(self, name, image_set, root_path, dataset_path, mode='train', min_size = -1):
        self.name = name + '_' + image_set
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path
        self.mode = mode
        self.min_size = min_size
        self.annotations = self.load_annotations()

    def get_annotations(self):
        return self.annotations

    def load_annotations(self):

        annotation_file = os.path.join(self.data_path, 'imglists', self.image_set + '.txt')
        assert os.path.exists(annotation_file), 'annotations not found at {}'.format(annotation_file)
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()
        if self.min_size > 0:
            num = len(annotations)
            selected_annotations = list()
            for i in range(num):
                annotation = annotations[i].split()
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
                if bbox_size >= self.min_size:
                    selected_annotations.append(annotations[i])
            print('%d/%d selected\n'%(len(selected_annotations),num))
            return selected_annotations
        else:
            return annotations