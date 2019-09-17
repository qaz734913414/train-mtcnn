import mxnet as mx
import numpy as np
from config import config

class NegativeMiningOperator(mx.operator.CustomOp):
    def __init__(self, cls_ohem=config.CLS_OHEM, cls_ohem_ratio=config.CLS_OHEM_RATIO,
            bbox_ohem=config.BBOX_OHEM, bbox_ohem_ratio=config.BBOX_OHEM_RATIO):
        super(NegativeMiningOperator, self).__init__()
        self.cls_ohem = cls_ohem
        self.cls_ohem_ratio = cls_ohem_ratio
        self.bbox_ohem = bbox_ohem
        self.bbox_ohem_ratio = bbox_ohem_ratio

    def forward(self, is_train, req, in_data, out_data, aux):
        cls_prob = in_data[0].asnumpy() # batchsize x 2 x 1 x 1
        bbox_pred = in_data[2].asnumpy() # batchsize x 4
        label = in_data[1].asnumpy().astype(int) # batchsize x 1
        bbox_target = in_data[3].asnumpy() # batchsize x 4

        self.assign(out_data[0], req[0], in_data[0])
        self.assign(out_data[2], req[2], in_data[2])

        # cls
        cls_prob = cls_prob.reshape(-1, 2)
        valid_inds = np.where(label > -1)[0]
        cls_keep = np.zeros(cls_prob.shape[0])

        if self.cls_ohem:
            keep_num = int(len(valid_inds) * self.cls_ohem_ratio)
            cls_valid = cls_prob[valid_inds, :]
            label_valid = label.flatten()[valid_inds]

            cls = cls_valid[np.arange(len(valid_inds)), label_valid] + config.EPS
            log_loss = - np.log(cls)
            keep = np.argsort(log_loss)[::-1][:keep_num]
            cls_keep[valid_inds[keep]] = 1
        else:
            cls_keep[valid_inds] = 1
        self.assign(out_data[1], req[1], mx.nd.array(cls_keep))

        # bbox
        valid_inds = np.where(abs(label) == 1)[0]
        bbox_keep = np.zeros(cls_prob.shape[0])

        if self.bbox_ohem:
            keep_num = int(len(valid_inds) * self.bbox_ohem_ratio)
            bbox_valid = bbox_pred[valid_inds, :]
            bbox_target_valid = bbox_target[valid_inds, :]
            square_error = np.sum((bbox_valid - bbox_target_valid)**2, axis=1)
            keep = np.argsort(square_error)[::-1][:keep_num]
            bbox_keep[valid_inds[keep]] = 1
        else:
            bbox_keep[valid_inds] = 1
        self.assign(out_data[3], req[3], mx.nd.array(bbox_keep))


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        cls_prob = in_data[0].asnumpy() # batchsize x 2 x 1 x 1
        label = in_data[1].asnumpy().astype(int) # batchsize x 1
        bbox_pred = in_data[2].asnumpy() # batchsize x 4
        bbox_target = in_data[3].asnumpy() # batchsize x 4
        cls_keep = out_data[1].asnumpy().reshape(-1, 1)
        cls_keep_repeat = np.repeat(cls_keep,2,axis=1)
        bbox_keep = out_data[3].asnumpy().reshape(-1, 1)
        bbox_keep_repeat = np.repeat(bbox_keep,4,axis=1)
        
        cls_target = np.zeros(cls_prob.shape)
        for i in range(cls_prob.shape[0]):
            if label[i] >= 0:
                cls_target[i][label[i]] = 1
        cls_grad = cls_prob - cls_target
        cls_grad /= len(np.where(cls_keep == 1)[0])
        cls_grad *= cls_keep_repeat * 1

        bbox_grad = 2*(bbox_pred - bbox_target)
        bbox_grad /= len(np.where(bbox_keep == 1)[0])
        bbox_grad *= bbox_keep_repeat * 2
        #print('backward')
        cls_grad = cls_grad.reshape(in_data[0].shape)
        self.assign(in_grad[0], req[0], mx.nd.array(cls_grad))
        self.assign(in_grad[2], req[2], mx.nd.array(bbox_grad))


@mx.operator.register("negativemining")
class NegativeMiningProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(NegativeMiningProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['cls_prob', 'label', 'bbox_pred', 'bbox_target']

    def list_outputs(self):
        return ['cls_out', 'cls_keep', 'bbox_out', 'bbox_keep']

    def infer_shape(self, in_shape):
        keep_shape = (in_shape[0][0], )
        return in_shape, [in_shape[0], keep_shape, in_shape[2], keep_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return NegativeMiningOperator()
