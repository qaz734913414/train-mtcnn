from tools.load_model import load_param
from tools.image_processing import transform
from core.symbol import L106_Net112
import cv2
import numpy as np
import mxnet as mx

sym = L106_Net112('test')
pretrained='model/lnet106_112'
epoch=4070
data_size=112
imshow_size=640
ctx = mx.cpu()
args, auxs = load_param(pretrained, epoch, convert=True, ctx=ctx)
#print(args)
#print(auxs)
data_shapes = {'data': (1, 3, data_size, data_size)}
img=cv2.imread('./00_.jpg')
img=cv2.resize(img,(data_size,data_size))
print(img.shape)
newimg1 = transform(img,False)
args['data'] = mx.nd.array(newimg1, ctx)
executor = sym.simple_bind(ctx, grad_req='null', **dict(data_shapes))#mx.cpu(), x=(5,4), grad_req='null'
executor.copy_params_from(args, auxs)
out_list = [[] for _ in range(len(executor.outputs))]
executor.forward(is_train=False)
for o_list, o_nd in zip(out_list, executor.outputs):
    o_list.append(o_nd.asnumpy())
out = list()
for o in out_list:
    out.append(np.vstack(o))
landmarks=out[0]
for j in range(int(len(landmarks)/2)):
    if(landmarks[2 * j]>1):
        landmarks[2 * j] = 1
    if (landmarks[2 * j] < 0):
        landmarks[2 * j] = 0
    if (landmarks[2 * j + 1] > 1):
        landmarks[2 * j + 1] = 1
    if (landmarks[2 * j + 1] < 0):
        landmarks[2 * j + 1] = 0

imshow_img=cv2.resize(img,(imshow_size,imshow_size))
landmarks=landmarks*imshow_size
landmarks=np.reshape(landmarks,-1)
#print(len(landmarks))
for j in range(int(len(landmarks)/2)):
    cv2.circle(imshow_img, (int(landmarks[2 * j]), (int(landmarks[2 * j + 1]))), 2, (0, 0, 255))
cv2.imshow("landmarks_106",imshow_img)
cv2.waitKey(0)

