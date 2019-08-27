import mxnet as mx
import negativemining
import negativemining_landmark
import negativemining_onlylandmark
import negativemining_onlylandmark10
import negativemining_onlylandmark17
import negativemining_onlylandmark106
import negativemining_onlylandmark106_heatmap
import negativemining_onlylandmark14_heatmap
from config import config

#def P_Net16_v0(mode='train'):
def P_Net16(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 16 x 16
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), stride=(2,2),  num_filter=16, name="conv1")#16/7
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2,2), num_filter=16, num_group=16, name="conv2_dw")#7/3
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=24, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")

    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=24, num_group=24, name="conv3_dw")#3/1
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net16_v1(mode='train'):
#def P_Net16(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 16 x 16
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=8, name="conv1")#16/15
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2,2), num_filter=8, num_group=8, name="conv2_dw")#15/7
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=16, num_group=16, name="conv3_dw")#7/3
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=24, num_group=24, name="conv4_dw")#3/1
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net20_p(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=8, name="conv1")#20/19
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3),stride=(2,2), num_filter=8, num_group=8, name="conv2_dw")#19/9
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    bn2 = mx.sym.BatchNorm(data=conv2_sep, name='bn2', fix_gamma=False,momentum=0.9)
    prelu2 = mx.symbol.LeakyReLU(data=bn2, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=16, num_group=16, name="conv3_dw")#9/4
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep")
    bn3 = mx.sym.BatchNorm(data=conv3_sep, name='bn3', fix_gamma=False,momentum=0.9)
    prelu3 = mx.symbol.LeakyReLU(data=bn3, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=24, num_group=24, name="conv4_dw")#4/3
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=32, name="conv4_sep")
    bn4 = mx.sym.BatchNorm(data=conv4_sep, name='bn4', fix_gamma=False,momentum=0.9)
    prelu4 = mx.symbol.LeakyReLU(data=bn4, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=32, num_group=32, name="conv5_dw")#3/1
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")	
	
    fc1_dw = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=64, name="fc1_dw")#1/1
    fc1_bn_dw = mx.sym.BatchNorm(data=fc1_dw, name='fc1_bn_dw', fix_gamma=False,momentum=0.9)
    fc1_relu_dw = mx.symbol.LeakyReLU(data=fc1_bn_dw, act_type="prelu", name="fc1_relu_dw")
	
    conv4_1 = mx.symbol.Convolution(data=fc1_relu_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
	
    conv5_bb_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=32, num_group=32, name="conv5_bb_dw")#3/1
    bn5_bb_dw = mx.sym.BatchNorm(data=conv5_bb_dw, name='bn5_bb_dw', fix_gamma=False,momentum=0.9)
    prelu5_bb_dw = mx.symbol.LeakyReLU(data=bn5_bb_dw, act_type="prelu", name="prelu5_bb_dw")	
	
    fc1_bb_dw = mx.symbol.Convolution(data=prelu5_bb_dw, kernel=(1, 1), num_filter=64, name="fc1_bb_dw")#1/1
    fc1_bb_bn_dw = mx.sym.BatchNorm(data=fc1_bb_dw, name='fc1_bb_bn_dw', fix_gamma=False,momentum=0.9)
    fc1_bb_relu_dw = mx.symbol.LeakyReLU(data=fc1_bb_bn_dw, act_type="prelu", name="fc1_bb_relu_dw")

    
    conv4_2 = mx.symbol.Convolution(data=fc1_bb_relu_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group
	
def P_Net20_v00(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), stride=(2,2), num_filter=8, name="conv1")#20/9
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3),stride=(2,2), num_filter=8, num_group=8, name="conv2_dw")#9/4
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(2, 2),num_filter=16, num_group=16, name="conv3_dw")#4/3
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=24, num_group=24, name="conv4_dw")#3/1
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

#def P_Net20_v0(mode='train'):
def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), stride=(2,2), num_filter=8, name="conv1")#20/9
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), num_filter=8, num_group=8, name="conv2_dw")#9/7
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=16, num_group=16, name="conv3_dw")#7/3
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=24, num_group=24, name="conv4_dw")#3/1
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group
	
def P_Net20_v1(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=8, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3),stride=(2,2), num_filter=8, num_group=8, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=8, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=8, num_group=8, name="conv3_dw")#9/4
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=16, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=16, num_group=16, name="conv4_dw")#4/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=24, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=24, num_group=24, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group
	
def P_Net20_v2(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=8, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3),stride=(2,2), num_filter=8, num_group=8, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=16, num_group=16, name="conv3_dw")#9/4
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=24, num_group=24, name="conv4_dw")#4/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=24, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=24, num_group=24, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net20_v3(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=8, name="conv1")#20/18
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool1") #18/9
    conv2_sep = mx.symbol.Convolution(data=pool1, kernel=(1, 1), num_filter=16, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=16, num_group=16, name="conv3_dw")#9/4
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=24, num_group=24, name="conv4_dw")#4/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=32, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=32, num_group=32, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net20_v4(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=16, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2,2), num_filter=16, num_group=16, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), stride=(2,2), num_filter=32, num_group=32, name="conv3_dw")#9/4
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=32, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=32, num_group=32, name="conv4_dw")#4/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=64, num_group=64, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net20_v5(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=16, name="conv1")#20/18
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool1") #18/9
    conv2_sep = mx.symbol.Convolution(data=pool1, kernel=(1, 1), num_filter=24, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=24, num_group=24, name="conv3_dw")#9/7
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=32, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), stride=(2,2), num_filter=32, num_group=32, name="conv4_dw")#7/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=64, num_group=64, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net20_v6(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=16, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2,2), num_filter=16, num_group=16, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=32, num_group=32, name="conv3_dw")#9/7
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=48, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), stride=(2,2), num_filter=48, num_group=48, name="conv4_dw")#7/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=64, num_group=64, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net20_v7(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=16, name="conv1")#20/18
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool1") #18/9
    conv2_sep = mx.symbol.Convolution(data=pool1, kernel=(1, 1), num_filter=24, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(2, 2), num_filter=24, num_group=24, name="conv3_dw")#9/8
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=32, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    pool2 = mx.symbol.Pooling(data=prelu3, pool_type="max", pooling_convention="full", kernel=(2, 2), stride=(2, 2), name="pool2") #8/4
    conv4_sep = mx.symbol.Convolution(data=pool2, kernel=(1, 1), num_filter=64, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(2, 2), num_filter=64, num_group=64, name="conv5_dw")#4/3
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=64, name="conv5_sep")
    prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")

    conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=64, num_group=64, name="conv6_dw")#3/1
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group
	
def P_Net20_v8(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=32, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3),stride=(2,2), num_filter=32, num_group=32, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=32, num_group=32, name="conv3_dw")#9/4
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=64, num_group=64, name="conv4_dw")#4/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=128, num_group=128, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def P_Net20_s2v1(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=8, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2,2), num_filter=8, num_group=8, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=16, num_group=16, name="conv3_dw")#9/7
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=16, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=16, num_group=16, name="conv4_dw")#7/5
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=24, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=24, num_group=24, name="conv5_dw")#5/3
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=32, name="conv5_sep")
    prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")
	
    conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=32, num_group=32, name="conv6_dw")#3/1
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group
	
def P_Net20_s2v2(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=8, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2,2), num_filter=8, num_group=8, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=16, num_group=16, name="conv3_dw")#9/7
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=24, num_group=24, name="conv4_dw")#7/5
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=32, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=32, num_group=32, name="conv5_dw")#5/3
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=64, name="conv5_sep")
    prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")
	
    conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=64, num_group=64, name="conv6_dw")#3/1
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group
	
def P_Net20_s2v3(mode='train'):
#def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=16, name="conv1")#20/19
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2,2), num_filter=16, num_group=16, name="conv2_dw")#19/9
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=24, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=24, num_group=24, name="conv3_dw")#9/7
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=32, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=32, num_group=32, name="conv4_dw")#7/5
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=48, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=48, num_group=48, name="conv5_dw")#5/3
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=64, name="conv5_sep")
    prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")
	
    conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=64, num_group=64, name="conv6_dw")#3/1
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")

    conv4_1 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    bn4_1 = mx.sym.BatchNorm(data=conv4_1, name='bn4_1', fix_gamma=False,momentum=0.9)
    conv4_2 = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=4, name="conv4_2")
    bn4_2 = mx.sym.BatchNorm(data=conv4_2, name='bn4_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=bn4_1, mode="channel", name="cls_prob")
        bbox_pred = bn4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        
    else:
        conv4_1_reshape = mx.symbol.Reshape(data = bn4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = bn4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred,bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group
	
def R_Net_p0(mode='train'):
#def R_Net(mode='train'):
    """
    Refine Network
    input shape 3 x 24 x 24
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=8, name="conv1") #24/23
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    relu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="relu1")
    
    conv2_dw = mx.symbol.Convolution(data=relu1, kernel=(3, 3), stride=(2, 2), num_filter=8, num_group=8, name="conv2_dw")#23/11
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    relu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="relu2_dw")
    conv2_sep = mx.symbol.Convolution(data=relu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    relu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="relu2")
    
    conv3_dw = mx.symbol.Convolution(data=relu2, kernel=(3, 3), stride=(2, 2), num_filter=16, num_group=16, name="conv3_dw")#11/5
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    relu3_dw = mx.sym.Activation(data=bn3_dw, act_type="relu", name='relu3_dw')
    conv3_sep = mx.symbol.Convolution(data=relu3_dw, kernel=(1, 1), num_filter=32, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    relu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="relu3")
    
    conv4_dw = mx.symbol.Convolution(data=relu3, kernel=(3, 3), num_filter=32, num_group=32, name="conv4_dw")#5/3
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    relu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="relu4_dw")
    conv4_sep = mx.symbol.Convolution(data=relu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    relu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="relu4")

    conv5_dw = mx.symbol.Convolution(data=relu4, kernel=(3, 3), num_filter=64, num_group=64, name="conv5_dw")#3/1
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    relu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="relu5_dw")
	
    fc1_dw = mx.symbol.Convolution(data=relu5_dw, kernel=(1, 1), num_filter=128, name="fc1_dw")#1/1
    fc1_bn_dw = mx.sym.BatchNorm(data=fc1_dw, name='fc1_bn_dw', fix_gamma=False,momentum=0.9)
    fc1_relu_dw = mx.symbol.LeakyReLU(data=fc1_bn_dw, act_type="prelu", name="fc1_relu_dw")
	
    fc2_dw = mx.symbol.Convolution(data=fc1_relu_dw, kernel=(1, 1), num_filter=128, name="fc2_dw")#1/1
    fc2_bn_dw = mx.sym.BatchNorm(data=fc2_dw, name='fc2_bn_dw', fix_gamma=False,momentum=0.9)
    fc2_relu_dw = mx.symbol.LeakyReLU(data=fc2_bn_dw, act_type="prelu", name="fc2_relu_dw")
	
    conv5_1 = mx.symbol.FullyConnected(data=fc2_relu_dw, num_hidden=2, name="conv5_1")
    bn5_1 = mx.sym.BatchNorm(data=conv5_1, name='bn5_1', fix_gamma=False,momentum=0.9)
    cls_prob = mx.symbol.SoftmaxOutput(data=bn5_1, label=label, use_ignore=True, name="cls_prob")
	
    conv4_bb_dw = mx.symbol.Convolution(data=relu3, kernel=(3, 3), num_filter=32, num_group=32, name="conv4_bb_dw")#5/3
    bn4_bb_dw = mx.sym.BatchNorm(data=conv4_bb_dw, name='bn4_bb_dw', fix_gamma=False,momentum=0.9)
    relu4_bb_dw = mx.symbol.LeakyReLU(data=bn4_bb_dw, act_type="prelu", name="relu4_bb_dw")
    conv4_bb_sep = mx.symbol.Convolution(data=relu4_bb_dw, kernel=(1, 1), num_filter=64, name="conv4_bb_sep")
    bn4_bb_sep = mx.sym.BatchNorm(data=conv4_bb_sep, name='bn4_bb_sep', fix_gamma=False,momentum=0.9)
    relu4_bb = mx.symbol.LeakyReLU(data=bn4_bb_sep, act_type="prelu", name="relu4_bb")

    conv5_bb_dw = mx.symbol.Convolution(data=relu4_bb, kernel=(3, 3), num_filter=64, num_group=64, name="conv5_bb_dw")#3/1
    bn5_bb_dw = mx.sym.BatchNorm(data=conv5_bb_dw, name='bn5_bb_dw', fix_gamma=False,momentum=0.9)
    relu5_bb_dw = mx.symbol.LeakyReLU(data=bn5_bb_dw, act_type="prelu", name="relu5_bb_dw")
	
    fc1_bb_dw = mx.symbol.Convolution(data=relu5_bb_dw, kernel=(1, 1), num_filter=128, name="fc1_bb_dw")#1/1
    fc1_bn_bb_dw = mx.sym.BatchNorm(data=fc1_bb_dw, name='fc1_bn_bb_dw', fix_gamma=False,momentum=0.9)
    fc1_relu_bb_dw = mx.symbol.LeakyReLU(data=fc1_bn_bb_dw, act_type="prelu", name="fc1_relu_bb_dw")
	
    fc2_bb_dw = mx.symbol.Convolution(data=fc1_relu_bb_dw, kernel=(1, 1), num_filter=128, name="fc2_bb_dw")#1/1
    fc2_bn_bb_dw = mx.sym.BatchNorm(data=fc2_bb_dw, name='fc2_bn_bb_dw', fix_gamma=False,momentum=0.9)
    fc2_relu_bb_dw = mx.symbol.LeakyReLU(data=fc2_bn_bb_dw, act_type="prelu", name="fc2_relu_bb_dw")

    conv5_2 = mx.symbol.FullyConnected(data=fc2_relu_bb_dw, num_hidden=4, name="conv5_2")
    bn5_2 = mx.sym.BatchNorm(data=conv5_2, name='bn5_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        bbox_pred = bn5_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=bn5_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                op_type='negativemining', name="negative_mining")

        group = mx.symbol.Group([out])
    return group

def R_Net_p(mode='train'):
#def R_Net(mode='train'):
    """
    Refine Network
    input shape 3 x 24 x 24
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=16, name="conv1") #24/23
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    relu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="relu1")
    
    conv2_dw = mx.symbol.Convolution(data=relu1, kernel=(3, 3), stride=(2, 2), num_filter=16, num_group=16, name="conv2_dw")#23/11
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    relu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="relu2_dw")
    conv2_sep = mx.symbol.Convolution(data=relu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    relu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="relu2")
    
    conv3_dw = mx.symbol.Convolution(data=relu2, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw")#11/5
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    relu3_dw = mx.sym.Activation(data=bn3_dw, act_type="relu", name='relu3_dw')
    conv3_sep = mx.symbol.Convolution(data=relu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    relu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="relu3")
    
    conv4_dw = mx.symbol.Convolution(data=relu3, kernel=(3, 3), num_filter=64, num_group=64, name="conv4_dw")#5/3
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    relu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="relu4_dw")
    conv4_sep = mx.symbol.Convolution(data=relu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    relu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="relu4")

    conv5_dw = mx.symbol.Convolution(data=relu4, kernel=(3, 3), num_filter=128, num_group=128, name="conv5_dw")#3/1
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    relu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="relu5_dw")
	
    fc1_dw = mx.symbol.Convolution(data=relu5_dw, kernel=(1, 1), num_filter=128, name="fc1_dw")#1/1
    fc1_bn_dw = mx.sym.BatchNorm(data=fc1_dw, name='fc1_bn_dw', fix_gamma=False,momentum=0.9)
    fc1_relu_dw = mx.symbol.LeakyReLU(data=fc1_bn_dw, act_type="prelu", name="fc1_relu_dw")
	
    fc2_dw = mx.symbol.Convolution(data=fc1_relu_dw, kernel=(1, 1), num_filter=128, name="fc2_dw")#1/1
    fc2_bn_dw = mx.sym.BatchNorm(data=fc2_dw, name='fc2_bn_dw', fix_gamma=False,momentum=0.9)
    fc2_relu_dw = mx.symbol.LeakyReLU(data=fc2_bn_dw, act_type="prelu", name="fc2_relu_dw")
	
    conv5_1 = mx.symbol.FullyConnected(data=fc2_relu_dw, num_hidden=2, name="conv5_1")
    bn5_1 = mx.sym.BatchNorm(data=conv5_1, name='bn5_1', fix_gamma=False,momentum=0.9)
    cls_prob = mx.symbol.SoftmaxOutput(data=bn5_1, label=label, use_ignore=True, name="cls_prob")
	
    conv4_bb_dw = mx.symbol.Convolution(data=relu3, kernel=(3, 3), num_filter=64, num_group=64, name="conv4_bb_dw")#5/3
    bn4_bb_dw = mx.sym.BatchNorm(data=conv4_bb_dw, name='bn4_bb_dw', fix_gamma=False,momentum=0.9)
    relu4_bb_dw = mx.symbol.LeakyReLU(data=bn4_bb_dw, act_type="prelu", name="relu4_bb_dw")
    conv4_bb_sep = mx.symbol.Convolution(data=relu4_bb_dw, kernel=(1, 1), num_filter=128, name="conv4_bb_sep")
    bn4_bb_sep = mx.sym.BatchNorm(data=conv4_bb_sep, name='bn4_bb_sep', fix_gamma=False,momentum=0.9)
    relu4_bb = mx.symbol.LeakyReLU(data=bn4_bb_sep, act_type="prelu", name="relu4_bb")

    conv5_bb_dw = mx.symbol.Convolution(data=relu4_bb, kernel=(3, 3), num_filter=128, num_group=128, name="conv5_bb_dw")#3/1
    bn5_bb_dw = mx.sym.BatchNorm(data=conv5_bb_dw, name='bn5_bb_dw', fix_gamma=False,momentum=0.9)
    relu5_bb_dw = mx.symbol.LeakyReLU(data=bn5_bb_dw, act_type="prelu", name="relu5_bb_dw")
	
    fc1_bb_dw = mx.symbol.Convolution(data=relu5_bb_dw, kernel=(1, 1), num_filter=128, name="fc1_bb_dw")#1/1
    fc1_bn_bb_dw = mx.sym.BatchNorm(data=fc1_bb_dw, name='fc1_bn_bb_dw', fix_gamma=False,momentum=0.9)
    fc1_relu_bb_dw = mx.symbol.LeakyReLU(data=fc1_bn_bb_dw, act_type="prelu", name="fc1_relu_bb_dw")
	
    fc2_bb_dw = mx.symbol.Convolution(data=fc1_relu_bb_dw, kernel=(1, 1), num_filter=128, name="fc2_bb_dw")#1/1
    fc2_bn_bb_dw = mx.sym.BatchNorm(data=fc2_bb_dw, name='fc2_bn_bb_dw', fix_gamma=False,momentum=0.9)
    fc2_relu_bb_dw = mx.symbol.LeakyReLU(data=fc2_bn_bb_dw, act_type="prelu", name="fc2_relu_bb_dw")

    conv5_2 = mx.symbol.FullyConnected(data=fc2_relu_bb_dw, num_hidden=4, name="conv5_2")
    bn5_2 = mx.sym.BatchNorm(data=conv5_2, name='bn5_2', fix_gamma=False,momentum=0.9)
    if mode == 'test':
        bbox_pred = bn5_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=bn5_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                op_type='negativemining', name="negative_mining")

        group = mx.symbol.Group([out])
    return group
	
#def R_Net_v1(mode='train'):
def R_Net(mode='train'):
    """
    Refine Network
    input shape 3 x 24 x 24
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=32, name="conv1") #24/23
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv2_dw")#23/11
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")

    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw")#11/5
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=64, num_group=64, name="conv4_dw")#5/3
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=128, num_group=128, name="conv5_dw")#3/1
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
	
    conv5_1 = mx.symbol.FullyConnected(data=prelu5_dw, num_hidden=2, name="conv5_1")
    bn5_1 = mx.sym.BatchNorm(data=conv5_1, name='bn5_1', fix_gamma=False,momentum=0.9)
    conv5_2 = mx.symbol.FullyConnected(data=prelu5_dw, num_hidden=4, name="conv5_2")
    bn5_2 = mx.sym.BatchNorm(data=conv5_2, name='bn5_2', fix_gamma=False,momentum=0.9)
    cls_prob = mx.symbol.SoftmaxOutput(data=bn5_1, label=label, use_ignore=True, name="cls_prob")
    if mode == 'test':
        bbox_pred = bn5_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=bn5_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                op_type='negativemining', name="negative_mining")

        group = mx.symbol.Group([out])
    return group
	
def R_Net_v2(mode='train'):
#def R_Net(mode='train'):
    """
    Refine Network
    input shape 3 x 24 x 24
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=32, name="conv1") #24/22
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=32, num_group=32, name="conv2_dw")#22/21
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep")
    prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")

    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw")#21/10
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=32, name="conv3_sep")
    prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=32, num_group=32, name="conv4_dw")#10/9
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep")
    prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), stride=(2, 2), num_filter=64, num_group=64, name="conv5_dw")#9/4
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=64, name="conv5_sep")
    prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")
	
    conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(2, 2), num_filter=64, num_group=64, name="conv6_dw")#4/3
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=128, name="conv6_sep")
    prelu6 = mx.symbol.LeakyReLU(data=conv6_sep, act_type="prelu", name="prelu6")

    conv7_dw = mx.symbol.Convolution(data=prelu6, kernel=(3, 3), num_filter=128, num_group=128, name="conv7_dw")#3/1
    prelu7_dw = mx.symbol.LeakyReLU(data=conv7_dw, act_type="prelu", name="prelu7_dw")
	
    conv5_1 = mx.symbol.FullyConnected(data=prelu7_dw, num_hidden=2, name="conv5_1")
    bn5_1 = mx.sym.BatchNorm(data=conv5_1, name='bn5_1', fix_gamma=False,momentum=0.9)
    conv5_2 = mx.symbol.FullyConnected(data=prelu7_dw, num_hidden=4, name="conv5_2")
    bn5_2 = mx.sym.BatchNorm(data=conv5_2, name='bn5_2', fix_gamma=False,momentum=0.9)
    cls_prob = mx.symbol.SoftmaxOutput(data=bn5_1, label=label, use_ignore=True, name="cls_prob")
    if mode == 'test':
        bbox_pred = bn5_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=bn5_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                op_type='negativemining', name="negative_mining")

        group = mx.symbol.Group([out])
    return group

def O_Net_p0(mode="train", with_landmark = False):
#def O_Net(mode="train", with_landmark = False):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2),num_filter=8, name="conv1") #48/47
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    relu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="relu1")
    
    conv2_dw = mx.symbol.Convolution(data=relu1, kernel=(3, 3), stride=(2, 2), num_filter=8, num_group=8, name="conv2_dw") #47/23
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    relu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="relu2_dw")
    conv2_sep = mx.symbol.Convolution(data=relu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    relu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="relu2")

    conv3_dw = mx.symbol.Convolution(data=relu2, kernel=(3, 3), stride=(2, 2), num_filter=16, num_group=16, name="conv3_dw") #23/11
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    relu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="relu3_dw")
    conv3_sep = mx.symbol.Convolution(data=relu3_dw, kernel=(1, 1), num_filter=32, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    relu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="relu3")

    conv4_dw = mx.symbol.Convolution(data=relu3, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv4_dw") #11/5
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    relu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="relu4_dw")
    conv4_sep = mx.symbol.Convolution(data=relu4_dw, kernel=(1, 1), num_filter=32, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    relu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="relu4")
	
    conv5_dw = mx.symbol.Convolution(data=relu4, kernel=(3, 3), num_filter=32, num_group=32, name="conv5_dw") #5/3
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    relu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="relu5_dw")
    conv5_sep = mx.symbol.Convolution(data=relu5_dw, kernel=(1, 1), num_filter=64, name="conv5_sep")
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    relu5 = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="relu5")
    
    conv6_dw = mx.symbol.Convolution(data=relu5, kernel=(3, 3), num_filter=64, num_group=64, name="conv6_dw") #3/1
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    relu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="relu6_dw")
	
    fc1 = mx.symbol.Convolution(data=relu6_dw, kernel=(1, 1), num_filter=128, name="fc1")#1/1
    fc1_bn = mx.sym.BatchNorm(data=fc1, name='fc1_bn', fix_gamma=False,momentum=0.9)
    fc1_relu = mx.symbol.LeakyReLU(data=fc1_bn, act_type="prelu", name="fc1_relu")
	
    fc2 = mx.symbol.Convolution(data=fc1_relu, kernel=(1, 1), num_filter=128, name="fc2")#1/1
    fc2_bn = mx.sym.BatchNorm(data=fc2, name='fc2_bn', fix_gamma=False,momentum=0.9)
    fc2_relu = mx.symbol.LeakyReLU(data=fc2_bn, act_type="prelu", name="fc2_relu")
	
    conv6_1 = mx.symbol.FullyConnected(data=fc2_relu, num_hidden=2, name="conv6_1")
    bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)
    cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
	
    conv5_bb_dw = mx.symbol.Convolution(data=relu4, kernel=(3, 3), num_filter=32, num_group=32, name="conv5_bb_dw") #5/3
    bn5_bb_dw = mx.sym.BatchNorm(data=conv5_bb_dw, name='bn5_bb_dw', fix_gamma=False,momentum=0.9)
    relu5_bb_dw = mx.symbol.LeakyReLU(data=bn5_bb_dw, act_type="prelu", name="relu5_bb_dw")
    conv5_bb_sep = mx.symbol.Convolution(data=relu5_bb_dw, kernel=(1, 1), num_filter=64, name="conv5_bb_sep")
    bn5_bb_sep = mx.sym.BatchNorm(data=conv5_bb_sep, name='bn5_bb_sep', fix_gamma=False,momentum=0.9)
    relu5_bb = mx.symbol.LeakyReLU(data=bn5_bb_sep, act_type="prelu", name="relu5_bb")
	
    conv6_bb_dw = mx.symbol.Convolution(data=relu5_bb, kernel=(3, 3), num_filter=64, num_group=64, name="conv6_bb_dw") #3/1
    bn6_bb_dw = mx.sym.BatchNorm(data=conv6_bb_dw, name='bn6_bb_dw', fix_gamma=False,momentum=0.9)
    relu6_bb_dw = mx.symbol.LeakyReLU(data=bn6_bb_dw, act_type="prelu", name="relu6_bb_dw")
	
    fc1_bb = mx.symbol.Convolution(data=relu6_bb_dw, kernel=(1, 1), num_filter=128, name="fc1_bb")#1/1
    fc1_bn_bb = mx.sym.BatchNorm(data=fc1_bb, name='fc1_bn_bb', fix_gamma=False,momentum=0.9)
    fc1_relu_bb = mx.symbol.LeakyReLU(data=fc1_bn_bb, act_type="prelu", name="fc1_relu_bb")
	
    fc2_bb = mx.symbol.Convolution(data=fc1_relu_bb, kernel=(1, 1), num_filter=128, name="fc2_bb")#1/1
    fc2_bn_bb = mx.sym.BatchNorm(data=fc2_bb, name='fc2_bn_bb', fix_gamma=False,momentum=0.9)
    fc2_relu_bb = mx.symbol.LeakyReLU(data=fc2_bn_bb, act_type="prelu", name="fc2_relu_bb")
	
    conv6_2 = mx.symbol.FullyConnected(data=fc2_relu_bb, num_hidden=4, name="conv6_2")	
    bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)
    if mode == "test":
        bbox_pred = bn6_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                 grad_scale=1, name="bbox_pred")
        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                           op_type='negativemining', name="negative_mining")
        group = mx.symbol.Group([out])
    return group

def O_Net_p(mode="train", with_landmark = False):
#def O_Net(mode="train", with_landmark = False):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2),num_filter=16, name="conv1") #48/47
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    relu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="relu1")
    
    conv2_dw = mx.symbol.Convolution(data=relu1, kernel=(3, 3), stride=(2, 2), num_filter=16, num_group=16, name="conv2_dw") #47/23
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    relu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="relu2_dw")
    conv2_sep = mx.symbol.Convolution(data=relu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    relu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="relu2")

    conv3_dw = mx.symbol.Convolution(data=relu2, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw") #23/11
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    relu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="relu3_dw")
    conv3_sep = mx.symbol.Convolution(data=relu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    relu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="relu3")

    conv4_dw = mx.symbol.Convolution(data=relu3, kernel=(3, 3), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw") #11/5
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    relu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="relu4_dw")
    conv4_sep = mx.symbol.Convolution(data=relu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    relu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="relu4")
	
    conv5_dw = mx.symbol.Convolution(data=relu4, kernel=(3, 3), num_filter=64, num_group=64, name="conv5_dw") #5/3
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    relu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="relu5_dw")
    conv5_sep = mx.symbol.Convolution(data=relu5_dw, kernel=(1, 1), num_filter=128, name="conv5_sep")
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    relu5 = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="relu5")
    
    conv6_dw = mx.symbol.Convolution(data=relu5, kernel=(3, 3), num_filter=128, num_group=128, name="conv6_dw") #3/1
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    relu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="relu6_dw")
	
    fc1 = mx.symbol.Convolution(data=relu6_dw, kernel=(1, 1), num_filter=128, name="fc1")#1/1
    fc1_bn = mx.sym.BatchNorm(data=fc1, name='fc1_bn', fix_gamma=False,momentum=0.9)
    fc1_relu = mx.symbol.LeakyReLU(data=fc1_bn, act_type="prelu", name="fc1_relu")
	
    fc2 = mx.symbol.Convolution(data=fc1_relu, kernel=(1, 1), num_filter=128, name="fc2")#1/1
    fc2_bn = mx.sym.BatchNorm(data=fc2, name='fc2_bn', fix_gamma=False,momentum=0.9)
    fc2_relu = mx.symbol.LeakyReLU(data=fc2_bn, act_type="prelu", name="fc2_relu")
	
    conv6_1 = mx.symbol.FullyConnected(data=fc2_relu, num_hidden=2, name="conv6_1")
    bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)
    cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
	
    conv5_bb_dw = mx.symbol.Convolution(data=relu4, kernel=(3, 3), num_filter=64, num_group=64, name="conv5_bb_dw") #5/3
    bn5_bb_dw = mx.sym.BatchNorm(data=conv5_bb_dw, name='bn5_bb_dw', fix_gamma=False,momentum=0.9)
    relu5_bb_dw = mx.symbol.LeakyReLU(data=bn5_bb_dw, act_type="prelu", name="relu5_bb_dw")
    conv5_bb_sep = mx.symbol.Convolution(data=relu5_bb_dw, kernel=(1, 1), num_filter=128, name="conv5_bb_sep")
    bn5_bb_sep = mx.sym.BatchNorm(data=conv5_bb_sep, name='bn5_bb_sep', fix_gamma=False,momentum=0.9)
    relu5_bb = mx.symbol.LeakyReLU(data=bn5_bb_sep, act_type="prelu", name="relu5_bb")
	
    conv6_bb_dw = mx.symbol.Convolution(data=relu5_bb, kernel=(3, 3), num_filter=128, num_group=128, name="conv6_bb_dw") #3/1
    bn6_bb_dw = mx.sym.BatchNorm(data=conv6_bb_dw, name='bn6_bb_dw', fix_gamma=False,momentum=0.9)
    relu6_bb_dw = mx.symbol.LeakyReLU(data=bn6_bb_dw, act_type="prelu", name="relu6_bb_dw")
	
    fc1_bb = mx.symbol.Convolution(data=relu6_bb_dw, kernel=(1, 1), num_filter=128, name="fc1_bb")#1/1
    fc1_bn_bb = mx.sym.BatchNorm(data=fc1_bb, name='fc1_bn_bb', fix_gamma=False,momentum=0.9)
    fc1_relu_bb = mx.symbol.LeakyReLU(data=fc1_bn_bb, act_type="prelu", name="fc1_relu_bb")
	
    fc2_bb = mx.symbol.Convolution(data=fc1_relu_bb, kernel=(1, 1), num_filter=128, name="fc2_bb")#1/1
    fc2_bn_bb = mx.sym.BatchNorm(data=fc2_bb, name='fc2_bn_bb', fix_gamma=False,momentum=0.9)
    fc2_relu_bb = mx.symbol.LeakyReLU(data=fc2_bn_bb, act_type="prelu", name="fc2_relu_bb")
	
    conv6_2 = mx.symbol.FullyConnected(data=fc2_relu_bb, num_hidden=4, name="conv6_2")	
    bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)
    if mode == "test":
        bbox_pred = bn6_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                 grad_scale=1, name="bbox_pred")
        out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                           op_type='negativemining', name="negative_mining")
        group = mx.symbol.Group([out])
    return group
	
#def O_Net_v1(mode="train", with_landmark = False):
def O_Net(mode="train", with_landmark = False):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    if with_landmark:
        type_label = mx.symbol.Variable(name="type_label")
        landmark_target = mx.symbol.Variable(name="landmark_target")
        conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3),pad=(1,1), num_filter=32, name="conv1")
        bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
        prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
	
        conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=32, num_group=32, name="conv2_dw", no_bias=True)
        bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
        prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
        conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=64, name="conv2_sep", no_bias=True)
        bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
        prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

        conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv3_dw", no_bias=True)
        bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
        prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
        conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep", no_bias=True)
        bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
        prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

        conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw", no_bias=True)
        bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
        prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
        conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep", no_bias=True)
        bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
        prelu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4")
	
        conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=128, num_group=128, name="conv5_dw", no_bias=True)
        bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
        prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
        conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=256, name="conv5_sep", no_bias=True)
        bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
        prelu5 = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5")
    
        conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=256, num_group=256, name="conv6_dw", no_bias=True)
        bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
        prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
	
        conv6_1 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=2, name="conv6_1")
        bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)

        conv6_2 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=4, name="conv6_2")	
        bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)

        conv6_3 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=10, name="conv6_3")	
        bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
        if mode == "test":
            cls_prob = mx.symbol.SoftmaxActivation(data=bn6_1, mode="channel", name="cls_prob")
            bbox_pred = bn6_2
            landmark_pred = bn6_3
            group = mx.symbol.Group([cls_prob, bbox_pred, landmark_pred])
        else:
            cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
            bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")
            landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                     grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                landmark_pred=landmark_pred, landmark_target=landmark_target, 
                                type_label=type_label, op_type='negativemining_landmark', name="negative_mining")
            group = mx.symbol.Group([out])
    else:
        conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2),num_filter=32, name="conv1") #48/47
        prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
	
        conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv2_dw") #47/23
        prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
        conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep")
        prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")

        conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw") #23/11
        prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
        conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep")
        prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

        conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw") #11/5
        prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
        conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep")
        prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")
	
        conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=64, num_group=64, name="conv5_dw") #5/3
        prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
        conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=128, name="conv5_sep")
        prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")
    
        conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=128, num_group=128, name="conv6_dw") #3/1
        prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")
	
        conv6_1 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=2, name="conv6_1")
        bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)

        conv6_2 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=4, name="conv6_2")	
        bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)
        if mode == "test":
            cls_prob = mx.symbol.SoftmaxActivation(data=bn6_1,  name="cls_prob")
            bbox_pred = bn6_2
            group = mx.symbol.Group([cls_prob, bbox_pred])
        else:
            cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
            bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")
            out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                               op_type='negativemining', name="negative_mining")
            group = mx.symbol.Group([out])
    return group

def O_Net_v2(mode="train", with_landmark = False):
#def O_Net(mode="train", with_landmark = False):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    if with_landmark:
        type_label = mx.symbol.Variable(name="type_label")
        landmark_target = mx.symbol.Variable(name="landmark_target")
        conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3),pad=(1,1), num_filter=32, name="conv1")
        bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
        prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
	
        conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=32, num_group=32, name="conv2_dw", no_bias=True)
        bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
        prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
        conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=64, name="conv2_sep", no_bias=True)
        bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
        prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

        conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv3_dw", no_bias=True)
        bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
        prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
        conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep", no_bias=True)
        bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
        prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

        conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw", no_bias=True)
        bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
        prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
        conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep", no_bias=True)
        bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
        prelu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4")
	
        conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=128, num_group=128, name="conv5_dw", no_bias=True)
        bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
        prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
        conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=256, name="conv5_sep", no_bias=True)
        bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
        prelu5 = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5")
    
        conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=256, num_group=256, name="conv6_dw", no_bias=True)
        bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
        prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
	
        conv6_1 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=2, name="conv6_1")
        bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)

        conv6_2 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=4, name="conv6_2")	
        bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)

        conv6_3 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=10, name="conv6_3")	
        bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
        if mode == "test":
            cls_prob = mx.symbol.SoftmaxActivation(data=bn6_1, mode="channel", name="cls_prob")
            bbox_pred = bn6_2
            landmark_pred = bn6_3
            group = mx.symbol.Group([cls_prob, bbox_pred, landmark_pred])
        else:
            cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
            bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")
            landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                     grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                landmark_pred=landmark_pred, landmark_target=landmark_target, 
                                type_label=type_label, op_type='negativemining_landmark', name="negative_mining")
            group = mx.symbol.Group([out])
    else:
        conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2),num_filter=32, name="conv1") #48/47
        prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
	
        conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv2_dw") #47/23
        prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
        conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=64, name="conv2_sep")
        prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")

        conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), stride=(2, 2), num_filter=64, num_group=64, name="conv3_dw") #23/11
        prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
        conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep")
        prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")

        conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw") #11/5
        prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
        conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep")
        prelu4 = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4")
	
        conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=128, num_group=128, name="conv5_dw") #5/3
        prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
        conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=256, name="conv5_sep")
        prelu5 = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5")
    
        conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=256, num_group=256, name="conv6_dw") #3/1
        prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")
	
        conv6_1 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=2, name="conv6_1")
        bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)

        conv6_2 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=4, name="conv6_2")	
        bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)
        if mode == "test":
            cls_prob = mx.symbol.SoftmaxActivation(data=bn6_1,  name="cls_prob")
            bbox_pred = bn6_2
            group = mx.symbol.Group([cls_prob, bbox_pred])
        else:
            cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
            bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")
            out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                               op_type='negativemining', name="negative_mining")
            group = mx.symbol.Group([out])
    return group
	
def O_Net_v3(mode="train", with_landmark = False):
#def O_Net(mode="train", with_landmark = False):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")
    if with_landmark:
        type_label = mx.symbol.Variable(name="type_label")
        landmark_target = mx.symbol.Variable(name="landmark_target")
        conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3),pad=(1,1), num_filter=32, name="conv1")
        bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
        prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
	
        conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=32, num_group=32, name="conv2_dw", no_bias=True)
        bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
        prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
        conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=64, name="conv2_sep", no_bias=True)
        bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
        prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

        conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv3_dw", no_bias=True)
        bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
        prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
        conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep", no_bias=True)
        bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
        prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

        conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw", no_bias=True)
        bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
        prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
        conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep", no_bias=True)
        bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
        prelu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4")
	
        conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=128, num_group=128, name="conv5_dw", no_bias=True)
        bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
        prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
        conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=256, name="conv5_sep", no_bias=True)
        bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
        prelu5 = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5")
    
        conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(3, 3), num_filter=256, num_group=256, name="conv6_dw", no_bias=True)
        bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
        prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
	
        conv6_1 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=2, name="conv6_1")
        bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)

        conv6_2 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=4, name="conv6_2")	
        bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)

        conv6_3 = mx.symbol.FullyConnected(data=prelu6_dw, num_hidden=10, name="conv6_3")	
        bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
        if mode == "test":
            cls_prob = mx.symbol.SoftmaxActivation(data=bn6_1, mode="channel", name="cls_prob")
            bbox_pred = bn6_2
            landmark_pred = bn6_3
            group = mx.symbol.Group([cls_prob, bbox_pred, landmark_pred])
        else:
            cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
            bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")
            landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                     grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                                landmark_pred=landmark_pred, landmark_target=landmark_target, 
                                type_label=type_label, op_type='negativemining_landmark', name="negative_mining")
            group = mx.symbol.Group([out])
    else:
        conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3),num_filter=32, name="conv1", no_bias=True) #48/46
        bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
        prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
	
        conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=32, num_group=32, name="conv2_dw", no_bias=True) #46/45
        bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
        prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
        conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep", no_bias=True)
        bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
        prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

        conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw", no_bias=True) #45/22
        bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
        prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
        conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep", no_bias=True)
        bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
        prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

        conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(2, 2), num_filter=64, num_group=64, name="conv4_dw", no_bias=True) #22/21
        bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
        prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
        conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=64, name="conv4_sep", no_bias=True)
        bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
        prelu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4")
	
        conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), stride=(2, 2), num_filter=64, num_group=64, name="conv5_dw", no_bias=True) #21/10
        bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
        prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
        conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=64, name="conv5_sep", no_bias=True)
        bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
        prelu5 = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5")
    
        conv6_dw = mx.symbol.Convolution(data=prelu5, kernel=(2, 2), num_filter=64, num_group=64, name="conv6_dw", no_bias=True) #10/9
        bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
        prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
        conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=64, name="conv6_sep", no_bias=True)
        bn6_sep = mx.sym.BatchNorm(data=conv6_sep, name='bn6_sep', fix_gamma=False,momentum=0.9)
        prelu6 = mx.symbol.LeakyReLU(data=bn6_sep, act_type="prelu", name="prelu6")
		
        conv7_dw = mx.symbol.Convolution(data=prelu6, kernel=(3, 3), stride=(2, 2), num_filter=64, num_group=64, name="conv7_dw", no_bias=True) #9/4
        bn7_dw = mx.sym.BatchNorm(data=conv7_dw, name='bn7_dw', fix_gamma=False,momentum=0.9)
        prelu7_dw = mx.symbol.LeakyReLU(data=bn7_dw, act_type="prelu", name="prelu7_dw")
        conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=128, name="conv7_sep", no_bias=True)
        bn7_sep = mx.sym.BatchNorm(data=conv7_sep, name='bn7_sep', fix_gamma=False,momentum=0.9)
        prelu7 = mx.symbol.LeakyReLU(data=bn7_sep, act_type="prelu", name="prelu7")
		
        conv8_dw = mx.symbol.Convolution(data=prelu7, kernel=(2, 2), num_filter=128, num_group=128, name="conv8_dw", no_bias=True) #4/3
        bn8_dw = mx.sym.BatchNorm(data=conv8_dw, name='bn8_dw', fix_gamma=False,momentum=0.9)
        prelu8_dw = mx.symbol.LeakyReLU(data=bn8_dw, act_type="prelu", name="prelu8_dw")
        conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=256, name="conv8_sep", no_bias=True)
        bn8_sep = mx.sym.BatchNorm(data=conv8_sep, name='bn8_sep', fix_gamma=False,momentum=0.9)
        prelu8 = mx.symbol.LeakyReLU(data=bn8_sep, act_type="prelu", name="prelu8")

        conv9_dw = mx.symbol.Convolution(data=prelu8, kernel=(3, 3), num_filter=256, num_group=256, name="conv9_dw", no_bias=True) #3/1
        bn9_dw = mx.sym.BatchNorm(data=conv9_dw, name='bn9_dw', fix_gamma=False,momentum=0.9)
        prelu9_dw = mx.symbol.LeakyReLU(data=bn9_dw, act_type="prelu", name="prelu9_dw")
	
        conv6_1 = mx.symbol.FullyConnected(data=prelu9_dw, num_hidden=2, name="conv6_1")
        bn6_1 = mx.sym.BatchNorm(data=conv6_1, name='bn6_1', fix_gamma=False,momentum=0.9)

        conv6_2 = mx.symbol.FullyConnected(data=prelu9_dw, num_hidden=4, name="conv6_2")	
        bn6_2 = mx.sym.BatchNorm(data=conv6_2, name='bn6_2', fix_gamma=False,momentum=0.9)
        if mode == "test":
            cls_prob = mx.symbol.SoftmaxActivation(data=bn6_1, name="cls_prob")
            bbox_pred = bn6_2
            group = mx.symbol.Group([cls_prob, bbox_pred])
        else:
            cls_prob = mx.symbol.SoftmaxOutput(data=bn6_1, label=label, use_ignore=True, name="cls_prob")
            bbox_pred = mx.symbol.LinearRegressionOutput(data=bn6_2, label=bbox_target,
                                                     grad_scale=1, name="bbox_pred")
            out = mx.symbol.Custom(cls_prob=cls_prob, label=label, bbox_pred=bbox_pred, bbox_target=bbox_target, 
                               op_type='negativemining', name="negative_mining")
            group = mx.symbol.Group([out])
    return group

lnet_basenum=32
#def L_Net_v1(mode="train"):
def L_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=lnet_basenum, name="conv1") #48/46
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=lnet_basenum, num_group=lnet_basenum, name="conv2_dw") #46/45
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet_basenum, name="conv2_sep")
    prelu2_sep = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet_basenum, num_group=lnet_basenum, name="conv3_dw") #45/22
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet_basenum*2, name="conv3_sep")
    prelu3_sep = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(2, 2), num_filter=lnet_basenum*2, num_group=lnet_basenum*2, name="conv4_dw") #22/21
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet_basenum*2, name="conv4_sep")
    prelu4_sep = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet_basenum*2, num_group=lnet_basenum*2, name="conv5_dw") #21/10
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet_basenum*4, name="conv5_sep")
    prelu5_sep = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5_sep")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(2, 2), num_filter=lnet_basenum*4,num_group=lnet_basenum*4, name="conv6_dw") #10/9
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet_basenum*4, name="conv6_sep")
    prelu6_sep = mx.symbol.LeakyReLU(data=conv6_sep, act_type="prelu", name="prelu6_sep")
	
    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet_basenum*4,num_group=lnet_basenum*4, name="conv7_dw") #9/4
    prelu7_dw = mx.symbol.LeakyReLU(data=conv7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv7_sep")
    prelu7_sep = mx.symbol.LeakyReLU(data=conv7_sep, act_type="prelu", name="prelu7_sep")

    conv8_dw = mx.symbol.Convolution(data=prelu7_sep, kernel=(2, 2), num_filter=lnet_basenum*8,num_group=lnet_basenum*8, name="conv8_dw") #4/3
    prelu8_dw = mx.symbol.LeakyReLU(data=conv8_dw, act_type="prelu", name="prelu8_dw")
    conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv8_sep")
    prelu8_sep = mx.symbol.LeakyReLU(data=conv8_sep, act_type="prelu", name="prelu8_sep")

    conv9_dw = mx.symbol.Convolution(data=prelu8_sep, kernel=(3, 3), num_filter=lnet_basenum*8,num_group=lnet_basenum*8, name="conv9_dw") #3/1
    prelu9_dw = mx.symbol.LeakyReLU(data=conv9_dw, act_type="prelu", name="prelu9_dw")
    conv9_sep = mx.symbol.Convolution(data=prelu9_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv9_sep")
    prelu9_sep = mx.symbol.LeakyReLU(data=conv9_sep, act_type="prelu", name="prelu9_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu9_sep, num_hidden=10, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)

    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        if config.use_landmark10:
            target_x1 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=0, end=1)
            target_x2 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=1, end=2)
            target_x3 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=2, end=3)
            target_x4 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=3, end=4)
            target_x5 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=4, end=5)
            target_y1 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=5, end=6)
            target_y2 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=6, end=7)
            target_y3 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=7, end=8)
            target_y4 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=8, end=9)
            target_y5 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=9, end=10)
            bn_x1 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=0, end=1)
            bn_x2 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=1, end=2)
            bn_x3 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=2, end=3)
            bn_x4 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=3, end=4)
            bn_x5 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=4, end=5)
            bn_y1 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=5, end=6)
            bn_y2 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=6, end=7)
            bn_y3 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=7, end=8)
            bn_y4 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=8, end=9)
            bn_y5 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=9, end=10)
            pred_x1 = mx.symbol.LinearRegressionOutput(data=bn_x1, label=target_x1, grad_scale=1, name="pred_x1")			
            pred_x2 = mx.symbol.LinearRegressionOutput(data=bn_x2, label=target_x2, grad_scale=1, name="pred_x2")			
            pred_x3 = mx.symbol.LinearRegressionOutput(data=bn_x3, label=target_x3, grad_scale=1, name="pred_x3")			
            pred_x4 = mx.symbol.LinearRegressionOutput(data=bn_x4, label=target_x4, grad_scale=1, name="pred_x4")			
            pred_x5 = mx.symbol.LinearRegressionOutput(data=bn_x5, label=target_x5, grad_scale=1, name="pred_x5")			
            pred_y1 = mx.symbol.LinearRegressionOutput(data=bn_y1, label=target_y1, grad_scale=1, name="pred_y1")			
            pred_y2 = mx.symbol.LinearRegressionOutput(data=bn_y2, label=target_y2, grad_scale=1, name="pred_y2")			
            pred_y3 = mx.symbol.LinearRegressionOutput(data=bn_y3, label=target_y3, grad_scale=1, name="pred_y3")			
            pred_y4 = mx.symbol.LinearRegressionOutput(data=bn_y4, label=target_y4, grad_scale=1, name="pred_y4")			
            pred_y5 = mx.symbol.LinearRegressionOutput(data=bn_y5, label=target_y5, grad_scale=1, name="pred_y5")			
            out = mx.symbol.Custom(pred_x1=pred_x1,pred_x2=pred_x2,pred_x3=pred_x3,pred_x4=pred_x4,pred_x5=pred_x5,
                                pred_y1=pred_y1,pred_y2=pred_y2,pred_y3=pred_y3,pred_y4=pred_y4,pred_y5=pred_y5,
                                target_x1=target_x1,target_x2=target_x2,target_x3=target_x3,target_x4=target_x4,target_x5=target_x5,
                                target_y1=target_y1,target_y2=target_y2,target_y3=target_y3,target_y4=target_y4,target_y5=target_y5,
                                op_type='negativemining_onlylandmark10', name="negative_mining") 
            group = mx.symbol.Group([out])
        else:
            landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                     grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                                op_type='negativemining_onlylandmark', name="negative_mining")
            group = mx.symbol.Group([out])
        
    return group

def L_Net_v2(mode="train"):	
#def L_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=lnet_basenum, name="conv1") #48/46
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=lnet_basenum, num_group=lnet_basenum, name="conv2_dw") #46/45
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet_basenum, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2_sep = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet_basenum, num_group=lnet_basenum, name="conv3_dw") #45/22
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet_basenum*2, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3_sep = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(2, 2), num_filter=lnet_basenum*2, num_group=lnet_basenum*2, name="conv4_dw") #22/21
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet_basenum*2, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4_sep = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet_basenum*2, num_group=lnet_basenum*2, name="conv5_dw") #21/10
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet_basenum*4, name="conv5_sep")
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    prelu5_sep = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5_sep")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(2, 2), num_filter=lnet_basenum*4,num_group=lnet_basenum*4, name="conv6_dw") #10/9
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet_basenum*4, name="conv6_sep")
    bn6_sep = mx.sym.BatchNorm(data=conv6_sep, name='bn6_sep', fix_gamma=False,momentum=0.9)
    prelu6_sep = mx.symbol.LeakyReLU(data=bn6_sep, act_type="prelu", name="prelu6_sep")
	
    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet_basenum*4,num_group=lnet_basenum*4, name="conv7_dw") #9/4
    bn7_dw = mx.sym.BatchNorm(data=conv7_dw, name='bn7_dw', fix_gamma=False,momentum=0.9)
    prelu7_dw = mx.symbol.LeakyReLU(data=bn7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv7_sep")
    bn7_sep = mx.sym.BatchNorm(data=conv7_sep, name='bn7_sep', fix_gamma=False,momentum=0.9)
    prelu7_sep = mx.symbol.LeakyReLU(data=bn7_sep, act_type="prelu", name="prelu7_sep")

    conv8_dw = mx.symbol.Convolution(data=prelu7_sep, kernel=(2, 2), num_filter=lnet_basenum*8,num_group=lnet_basenum*8, name="conv8_dw") #4/3
    bn8_dw = mx.sym.BatchNorm(data=conv8_dw, name='bn8_dw', fix_gamma=False,momentum=0.9)
    prelu8_dw = mx.symbol.LeakyReLU(data=bn8_dw, act_type="prelu", name="prelu8_dw")
    conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv8_sep")
    bn8_sep = mx.sym.BatchNorm(data=conv8_sep, name='bn8_sep', fix_gamma=False,momentum=0.9)
    prelu8_sep = mx.symbol.LeakyReLU(data=bn8_sep, act_type="prelu", name="prelu8_sep")

    conv9_dw = mx.symbol.Convolution(data=prelu8_sep, kernel=(3, 3), num_filter=lnet_basenum*8,num_group=lnet_basenum*8, name="conv9_dw") #3/1
    bn9_dw = mx.sym.BatchNorm(data=conv9_dw, name='bn9_dw', fix_gamma=False,momentum=0.9)
    prelu9_dw = mx.symbol.LeakyReLU(data=bn9_dw, act_type="prelu", name="prelu9_dw")
    conv9_sep = mx.symbol.Convolution(data=prelu9_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv9_sep")
    bn9_sep = mx.sym.BatchNorm(data=conv9_sep, name='bn9_sep', fix_gamma=False,momentum=0.9)
    prelu9_sep = mx.symbol.LeakyReLU(data=bn9_sep, act_type="prelu", name="prelu9_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu9_sep, num_hidden=10, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)

    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        if config.use_landmark10:
            target_x1 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=0, end=1)
            target_x2 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=1, end=2)
            target_x3 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=2, end=3)
            target_x4 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=3, end=4)
            target_x5 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=4, end=5)
            target_y1 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=5, end=6)
            target_y2 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=6, end=7)
            target_y3 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=7, end=8)
            target_y4 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=8, end=9)
            target_y5 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=9, end=10)
            bn_x1 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=0, end=1)
            bn_x2 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=1, end=2)
            bn_x3 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=2, end=3)
            bn_x4 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=3, end=4)
            bn_x5 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=4, end=5)
            bn_y1 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=5, end=6)
            bn_y2 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=6, end=7)
            bn_y3 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=7, end=8)
            bn_y4 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=8, end=9)
            bn_y5 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=9, end=10)
            pred_x1 = mx.symbol.LinearRegressionOutput(data=bn_x1, label=target_x1, grad_scale=1, name="pred_x1")			
            pred_x2 = mx.symbol.LinearRegressionOutput(data=bn_x2, label=target_x2, grad_scale=1, name="pred_x2")			
            pred_x3 = mx.symbol.LinearRegressionOutput(data=bn_x3, label=target_x3, grad_scale=1, name="pred_x3")			
            pred_x4 = mx.symbol.LinearRegressionOutput(data=bn_x4, label=target_x4, grad_scale=1, name="pred_x4")			
            pred_x5 = mx.symbol.LinearRegressionOutput(data=bn_x5, label=target_x5, grad_scale=1, name="pred_x5")			
            pred_y1 = mx.symbol.LinearRegressionOutput(data=bn_y1, label=target_y1, grad_scale=1, name="pred_y1")			
            pred_y2 = mx.symbol.LinearRegressionOutput(data=bn_y2, label=target_y2, grad_scale=1, name="pred_y2")			
            pred_y3 = mx.symbol.LinearRegressionOutput(data=bn_y3, label=target_y3, grad_scale=1, name="pred_y3")			
            pred_y4 = mx.symbol.LinearRegressionOutput(data=bn_y4, label=target_y4, grad_scale=1, name="pred_y4")			
            pred_y5 = mx.symbol.LinearRegressionOutput(data=bn_y5, label=target_y5, grad_scale=1, name="pred_y5")			
            out = mx.symbol.Custom(pred_x1=pred_x1,pred_x2=pred_x2,pred_x3=pred_x3,pred_x4=pred_x4,pred_x5=pred_x5,
                                pred_y1=pred_y1,pred_y2=pred_y2,pred_y3=pred_y3,pred_y4=pred_y4,pred_y5=pred_y5,
                                target_x1=target_x1,target_x2=target_x2,target_x3=target_x3,target_x4=target_x4,target_x5=target_x5,
                                target_y1=target_y1,target_y2=target_y2,target_y3=target_y3,target_y4=target_y4,target_y5=target_y5,
                                op_type='negativemining_onlylandmark10', name="negative_mining") 
            group = mx.symbol.Group([out])
        else:
            landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                     grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                                op_type='negativemining_onlylandmark', name="negative_mining")
            group = mx.symbol.Group([out])
        
    return group
	
#def L_Net64_v3(mode="train"):
def L_Net64(mode="train"):
    """
    Refine Network
    input shape 3 x 64 x 64
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=lnet_basenum, name="conv1") #64/63
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2, 2), num_filter=lnet_basenum, num_group=lnet_basenum, name="conv2_dw") #63/31
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet_basenum*2, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2_sep = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet_basenum*2, num_group=lnet_basenum*2, name="conv3_dw") #31/15
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet_basenum*4, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3_sep = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet_basenum*4, num_group=lnet_basenum*4, name="conv4_dw") #15/7
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet_basenum*4, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4_sep = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet_basenum*4, num_group=lnet_basenum*4, name="conv5_dw") #7/3
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv5_sep")
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    prelu5_sep = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5_sep")

    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(3, 3), num_filter=lnet_basenum*8,num_group=lnet_basenum*8, name="conv6_dw") #3/1
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet_basenum*8, name="conv6_sep")
    bn6_sep = mx.sym.BatchNorm(data=conv6_sep, name='bn6_sep', fix_gamma=False,momentum=0.9)
    prelu6_sep = mx.symbol.LeakyReLU(data=bn6_sep, act_type="prelu", name="prelu6_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu6_sep, num_hidden=10, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        if config.use_landmark10:
            target_x1 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=0, end=1)
            target_x2 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=1, end=2)
            target_x3 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=2, end=3)
            target_x4 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=3, end=4)
            target_x5 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=4, end=5)
            target_y1 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=5, end=6)
            target_y2 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=6, end=7)
            target_y3 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=7, end=8)
            target_y4 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=8, end=9)
            target_y5 = mx.symbol.slice_axis(data = landmark_target, axis=1, begin=9, end=10)
            bn_x1 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=0, end=1)
            bn_x2 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=1, end=2)
            bn_x3 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=2, end=3)
            bn_x4 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=3, end=4)
            bn_x5 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=4, end=5)
            bn_y1 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=5, end=6)
            bn_y2 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=6, end=7)
            bn_y3 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=7, end=8)
            bn_y4 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=8, end=9)
            bn_y5 = mx.symbol.slice_axis(data = bn6_3, axis=1, begin=9, end=10)
            pred_x1 = mx.symbol.LinearRegressionOutput(data=bn_x1, label=target_x1, grad_scale=1, name="pred_x1")			
            pred_x2 = mx.symbol.LinearRegressionOutput(data=bn_x2, label=target_x2, grad_scale=1, name="pred_x2")			
            pred_x3 = mx.symbol.LinearRegressionOutput(data=bn_x3, label=target_x3, grad_scale=1, name="pred_x3")			
            pred_x4 = mx.symbol.LinearRegressionOutput(data=bn_x4, label=target_x4, grad_scale=1, name="pred_x4")			
            pred_x5 = mx.symbol.LinearRegressionOutput(data=bn_x5, label=target_x5, grad_scale=1, name="pred_x5")			
            pred_y1 = mx.symbol.LinearRegressionOutput(data=bn_y1, label=target_y1, grad_scale=1, name="pred_y1")			
            pred_y2 = mx.symbol.LinearRegressionOutput(data=bn_y2, label=target_y2, grad_scale=1, name="pred_y2")			
            pred_y3 = mx.symbol.LinearRegressionOutput(data=bn_y3, label=target_y3, grad_scale=1, name="pred_y3")			
            pred_y4 = mx.symbol.LinearRegressionOutput(data=bn_y4, label=target_y4, grad_scale=1, name="pred_y4")			
            pred_y5 = mx.symbol.LinearRegressionOutput(data=bn_y5, label=target_y5, grad_scale=1, name="pred_y5")			
            out = mx.symbol.Custom(pred_x1=pred_x1,pred_x2=pred_x2,pred_x3=pred_x3,pred_x4=pred_x4,pred_x5=pred_x5,
                                pred_y1=pred_y1,pred_y2=pred_y2,pred_y3=pred_y3,pred_y4=pred_y4,pred_y5=pred_y5,
                                target_x1=target_x1,target_x2=target_x2,target_x3=target_x3,target_x4=target_x4,target_x5=target_x5,
                                target_y1=target_y1,target_y2=target_y2,target_y3=target_y3,target_y4=target_y4,target_y5=target_y5,
                                op_type='negativemining_onlylandmark10', name="negative_mining") 
            group = mx.symbol.Group([out])
        else:
            landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                     grad_scale=1, name="landmark_pred")
            out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                                op_type='negativemining_onlylandmark', name="negative_mining")
            group = mx.symbol.Group([out])
        
    return group
	
lnet106_basenum=32
#def L106_Net_v1(mode="train"):
def L106_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=lnet106_basenum, name="conv1") #48/46
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv2_dw") #46/45
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet106_basenum, name="conv2_sep")
    prelu2_sep = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv3_dw") #45/22
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv3_sep")
    prelu3_sep = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(2, 2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv4_dw") #22/21
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv4_sep")
    prelu4_sep = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv5_dw") #21/10
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv5_sep")
    prelu5_sep = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5_sep")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv6_dw") #10/9
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv6_sep")
    prelu6_sep = mx.symbol.LeakyReLU(data=conv6_sep, act_type="prelu", name="prelu6_sep")
	
    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv7_dw") #9/4
    prelu7_dw = mx.symbol.LeakyReLU(data=conv7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv7_sep")
    prelu7_sep = mx.symbol.LeakyReLU(data=conv7_sep, act_type="prelu", name="prelu7_sep")

    conv8_dw = mx.symbol.Convolution(data=prelu7_sep, kernel=(2, 2), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv8_dw") #4/3
    prelu8_dw = mx.symbol.LeakyReLU(data=conv8_dw, act_type="prelu", name="prelu8_dw")
    conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv8_sep")
    prelu8_sep = mx.symbol.LeakyReLU(data=conv8_sep, act_type="prelu", name="prelu8_sep")

    conv9_dw = mx.symbol.Convolution(data=prelu8_sep, kernel=(3, 3), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv9_dw") #3/1
    prelu9_dw = mx.symbol.LeakyReLU(data=conv9_dw, act_type="prelu", name="prelu9_dw")
    conv9_sep = mx.symbol.Convolution(data=prelu9_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv9_sep")
    prelu9_sep = mx.symbol.LeakyReLU(data=conv9_sep, act_type="prelu", name="prelu9_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu9_sep, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group
	
def L106_Net_v2(mode="train"):
#def L106_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=lnet106_basenum, name="conv1") #48/46
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv2_dw") #46/45
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet106_basenum, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2_sep = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv3_dw") #45/22
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3_sep = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(2, 2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv4_dw") #22/21
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4_sep = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv5_dw") #21/10
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv5_sep")
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    prelu5_sep = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5_sep")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv6_dw") #10/9
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv6_sep")
    bn6_sep = mx.sym.BatchNorm(data=conv6_sep, name='bn6_sep', fix_gamma=False,momentum=0.9)
    prelu6_sep = mx.symbol.LeakyReLU(data=bn6_sep, act_type="prelu", name="prelu6_sep")
	
    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv7_dw") #9/4
    bn7_dw = mx.sym.BatchNorm(data=conv7_dw, name='bn7_dw', fix_gamma=False,momentum=0.9)
    prelu7_dw = mx.symbol.LeakyReLU(data=bn7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv7_sep")
    bn7_sep = mx.sym.BatchNorm(data=conv7_sep, name='bn7_sep', fix_gamma=False,momentum=0.9)
    prelu7_sep = mx.symbol.LeakyReLU(data=bn7_sep, act_type="prelu", name="prelu7_sep")

    conv8_dw = mx.symbol.Convolution(data=prelu7_sep, kernel=(2, 2), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv8_dw") #4/3
    bn8_dw = mx.sym.BatchNorm(data=conv8_dw, name='bn8_dw', fix_gamma=False,momentum=0.9)
    prelu8_dw = mx.symbol.LeakyReLU(data=bn8_dw, act_type="prelu", name="prelu8_dw")
    conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv8_sep")
    bn8_sep = mx.sym.BatchNorm(data=conv8_sep, name='bn8_sep', fix_gamma=False,momentum=0.9)
    prelu8_sep = mx.symbol.LeakyReLU(data=bn8_sep, act_type="prelu", name="prelu8_sep")

    conv9_dw = mx.symbol.Convolution(data=prelu8_sep, kernel=(3, 3), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv9_dw") #3/1
    bn9_dw = mx.sym.BatchNorm(data=conv9_dw, name='bn9_dw', fix_gamma=False,momentum=0.9)
    prelu9_dw = mx.symbol.LeakyReLU(data=bn9_dw, act_type="prelu", name="prelu9_dw")
    conv9_sep = mx.symbol.Convolution(data=prelu9_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv9_sep")
    bn9_sep = mx.sym.BatchNorm(data=conv9_sep, name='bn9_sep', fix_gamma=False,momentum=0.9)
    prelu9_sep = mx.symbol.LeakyReLU(data=bn9_sep, act_type="prelu", name="prelu9_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu9_sep, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group
	
#def L106_Net96_v1(mode="train"):
def L106_Net96(mode="train"):
    """
    Refine Network
    input shape 3 x 96 x 96
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=lnet106_basenum, name="conv1") #96/94
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv2_dw") #94/93
    prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet106_basenum, name="conv2_sep")
    prelu2_sep = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv3_dw") #93/46
    prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv3_sep")
    prelu3_sep = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(2, 2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv4_dw") #46/45
    prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv4_sep")
    prelu4_sep = mx.symbol.LeakyReLU(data=conv4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv5_dw") #45/22
    prelu5_dw = mx.symbol.LeakyReLU(data=conv5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv5_sep")
    prelu5_sep = mx.symbol.LeakyReLU(data=conv5_sep, act_type="prelu", name="prelu5_sep")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(2, 2), num_filter=lnet106_basenum*2,num_group=lnet106_basenum*2, name="conv6_dw") #22/21
    prelu6_dw = mx.symbol.LeakyReLU(data=conv6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv6_sep")
    prelu6_sep = mx.symbol.LeakyReLU(data=conv6_sep, act_type="prelu", name="prelu6_sep")
	
    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum*2,num_group=lnet106_basenum*2, name="conv7_dw") #21/10
    prelu7_dw = mx.symbol.LeakyReLU(data=conv7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv7_sep")
    prelu7_sep = mx.symbol.LeakyReLU(data=conv7_sep, act_type="prelu", name="prelu7_sep")
	
    conv8_dw = mx.symbol.Convolution(data=prelu7_sep, kernel=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv8_dw") #10/9
    prelu8_dw = mx.symbol.LeakyReLU(data=conv8_dw, act_type="prelu", name="prelu8_dw")
    conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv8_sep")
    prelu8_sep = mx.symbol.LeakyReLU(data=conv8_sep, act_type="prelu", name="prelu8_sep")

    conv9_dw = mx.symbol.Convolution(data=prelu8_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv9_dw") #9/4
    prelu9_dw = mx.symbol.LeakyReLU(data=conv9_dw, act_type="prelu", name="prelu9_dw")
    conv9_sep = mx.symbol.Convolution(data=prelu9_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv9_sep")
    prelu9_sep = mx.symbol.LeakyReLU(data=conv9_sep, act_type="prelu", name="prelu9_sep")

    conv10_dw = mx.symbol.Convolution(data=prelu9_sep, kernel=(2, 2), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv10_dw") #4/3
    prelu10_dw = mx.symbol.LeakyReLU(data=conv10_dw, act_type="prelu", name="prelu10_dw")
    conv10_sep = mx.symbol.Convolution(data=prelu10_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv10_sep")
    prelu10_sep = mx.symbol.LeakyReLU(data=conv10_sep, act_type="prelu", name="prelu10_sep")

    conv11_dw = mx.symbol.Convolution(data=prelu10_sep, kernel=(3, 3), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv11_dw") #3/1
    prelu11_dw = mx.symbol.LeakyReLU(data=conv11_dw, act_type="prelu", name="prelu11_dw")
    conv11_sep = mx.symbol.Convolution(data=prelu11_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv11_sep")
    prelu11_sep = mx.symbol.LeakyReLU(data=conv11_sep, act_type="prelu", name="prelu11_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu11_sep, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group
	
def L106_Net96_v2(mode="train"):
#def L106_Net96(mode="train"):
    """
    Refine Network
    input shape 3 x 96 x 96
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=lnet106_basenum, name="conv1") #96/94
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(2, 2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv2_dw") #94/93
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet106_basenum, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2_sep = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv3_dw") #93/46
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3_sep = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(2, 2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv4_dw") #46/45
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4_sep = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv5_dw") #45/22
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv5_sep")
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    prelu5_sep = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5_sep")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(2, 2), num_filter=lnet106_basenum*2,num_group=lnet106_basenum*2, name="conv6_dw") #22/21
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv6_sep")
    bn6_sep = mx.sym.BatchNorm(data=conv6_sep, name='bn6_sep', fix_gamma=False,momentum=0.9)
    prelu6_sep = mx.symbol.LeakyReLU(data=bn6_sep, act_type="prelu", name="prelu6_sep")
	
    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum*2,num_group=lnet106_basenum*2, name="conv7_dw") #21/10
    bn7_dw = mx.sym.BatchNorm(data=conv7_dw, name='bn7_dw', fix_gamma=False,momentum=0.9)
    prelu7_dw = mx.symbol.LeakyReLU(data=bn7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv7_sep")
    bn7_sep = mx.sym.BatchNorm(data=conv7_sep, name='bn7_sep', fix_gamma=False,momentum=0.9)
    prelu7_sep = mx.symbol.LeakyReLU(data=bn7_sep, act_type="prelu", name="prelu7_sep")
	
    conv8_dw = mx.symbol.Convolution(data=prelu7_sep, kernel=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv8_dw") #10/9
    bn8_dw = mx.sym.BatchNorm(data=conv8_dw, name='bn8_dw', fix_gamma=False,momentum=0.9)
    prelu8_dw = mx.symbol.LeakyReLU(data=bn8_dw, act_type="prelu", name="prelu8_dw")
    conv8_sep = mx.symbol.Convolution(data=prelu8_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv8_sep")
    bn8_sep = mx.sym.BatchNorm(data=conv8_sep, name='bn8_sep', fix_gamma=False,momentum=0.9)
    prelu8_sep = mx.symbol.LeakyReLU(data=bn8_sep, act_type="prelu", name="prelu8_sep")
	
    conv9_dw = mx.symbol.Convolution(data=prelu8_sep, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum*4,num_group=lnet106_basenum*4, name="conv9_dw") #9/4
    bn9_dw = mx.sym.BatchNorm(data=conv9_dw, name='bn9_dw', fix_gamma=False,momentum=0.9)
    prelu9_dw = mx.symbol.LeakyReLU(data=bn9_dw, act_type="prelu", name="prelu9_dw")
    conv9_sep = mx.symbol.Convolution(data=prelu9_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv9_sep")
    bn9_sep = mx.sym.BatchNorm(data=conv9_sep, name='bn9_sep', fix_gamma=False,momentum=0.9)
    prelu9_sep = mx.symbol.LeakyReLU(data=bn9_sep, act_type="prelu", name="prelu9_sep")

    conv10_dw = mx.symbol.Convolution(data=prelu9_sep, kernel=(2, 2), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv10_dw") #4/3
    bn10_dw = mx.sym.BatchNorm(data=conv10_dw, name='bn10_dw', fix_gamma=False,momentum=0.9)
    prelu10_dw = mx.symbol.LeakyReLU(data=bn10_dw, act_type="prelu", name="prelu10_dw")
    conv10_sep = mx.symbol.Convolution(data=prelu10_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv10_sep")
    bn10_sep = mx.sym.BatchNorm(data=conv10_sep, name='bn10_sep', fix_gamma=False,momentum=0.9)
    prelu10_sep = mx.symbol.LeakyReLU(data=bn10_sep, act_type="prelu", name="prelu10_sep")

    conv11_dw = mx.symbol.Convolution(data=prelu10_sep, kernel=(3, 3), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv11_dw") #3/1
    bn11_dw = mx.sym.BatchNorm(data=conv11_dw, name='bn11_dw', fix_gamma=False,momentum=0.9)
    prelu11_dw = mx.symbol.LeakyReLU(data=bn11_dw, act_type="prelu", name="prelu11_dw")
    conv11_sep = mx.symbol.Convolution(data=prelu11_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv11_sep")
    bn11_sep = mx.sym.BatchNorm(data=conv11_sep, name='bn11_sep', fix_gamma=False,momentum=0.9)
    prelu11_sep = mx.symbol.LeakyReLU(data=bn11_sep, act_type="prelu", name="prelu11_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu11_sep, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group
	
	
def L106_Net96_v3(mode="train"):
#def L106_Net96(mode="train"):
    """
    Refine Network
    input shape 3 x 96 x 96
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=lnet106_basenum, name="conv1") #96/95
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv2_dw") #95/47
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2_sep = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv3_dw") #47/23
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3_sep = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*4, num_group=lnet106_basenum*4, name="conv4_dw") #23/11
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4_sep = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*4, num_group=lnet106_basenum*4, name="conv5_dw") #11/5
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv5_sep")
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    prelu5_sep = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5_sep")
    
    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(3, 3), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv6_dw") #5/3
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv6_sep")
    bn6_sep = mx.sym.BatchNorm(data=conv6_sep, name='bn6_sep', fix_gamma=False,momentum=0.9)
    prelu6_sep = mx.symbol.LeakyReLU(data=bn6_sep, act_type="prelu", name="prelu6_sep")

    conv7_dw = mx.symbol.Convolution(data=prelu6_sep, kernel=(3, 3), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv7_dw") #3/1
    bn7_dw = mx.sym.BatchNorm(data=conv7_dw, name='bn7_dw', fix_gamma=False,momentum=0.9)
    prelu7_dw = mx.symbol.LeakyReLU(data=bn7_dw, act_type="prelu", name="prelu7_dw")
    conv7_sep = mx.symbol.Convolution(data=prelu7_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv7_sep")
    bn7_sep = mx.sym.BatchNorm(data=conv7_sep, name='bn7_sep', fix_gamma=False,momentum=0.9)
    prelu7_sep = mx.symbol.LeakyReLU(data=bn7_sep, act_type="prelu", name="prelu7_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu7_sep, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group
	
#def L106_Net64_v3(mode="train"):
def L106_Net64(mode="train"):
    """
    Refine Network
    input shape 3 x 64 x 64
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(2, 2), num_filter=lnet106_basenum, name="conv1") #64/63
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), stride=(2, 2), num_filter=lnet106_basenum, num_group=lnet106_basenum, name="conv2_dw") #63/31
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=lnet106_basenum*2, name="conv2_sep")
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2_sep = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2_sep")
	
    conv3_dw = mx.symbol.Convolution(data=prelu2_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*2, num_group=lnet106_basenum*2, name="conv3_dw") #31/15
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv3_sep")
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3_sep = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3_sep")
    
    conv4_dw = mx.symbol.Convolution(data=prelu3_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*4, num_group=lnet106_basenum*4, name="conv4_dw") #15/7
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=lnet106_basenum*4, name="conv4_sep")
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4_sep = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4_sep")

    conv5_dw = mx.symbol.Convolution(data=prelu4_sep, kernel=(3, 3), stride=(2,2), num_filter=lnet106_basenum*4, num_group=lnet106_basenum*4, name="conv5_dw") #7/3
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
    conv5_sep = mx.symbol.Convolution(data=prelu5_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv5_sep")
    bn5_sep = mx.sym.BatchNorm(data=conv5_sep, name='bn5_sep', fix_gamma=False,momentum=0.9)
    prelu5_sep = mx.symbol.LeakyReLU(data=bn5_sep, act_type="prelu", name="prelu5_sep")

    conv6_dw = mx.symbol.Convolution(data=prelu5_sep, kernel=(3, 3), num_filter=lnet106_basenum*8,num_group=lnet106_basenum*8, name="conv6_dw") #3/1
    bn6_dw = mx.sym.BatchNorm(data=conv6_dw, name='bn6_dw', fix_gamma=False,momentum=0.9)
    prelu6_dw = mx.symbol.LeakyReLU(data=bn6_dw, act_type="prelu", name="prelu6_dw")
    conv6_sep = mx.symbol.Convolution(data=prelu6_dw, kernel=(1, 1), num_filter=lnet106_basenum*8, name="conv6_sep")
    bn6_sep = mx.sym.BatchNorm(data=conv6_sep, name='bn6_sep', fix_gamma=False,momentum=0.9)
    prelu6_sep = mx.symbol.LeakyReLU(data=bn6_sep, act_type="prelu", name="prelu6_sep")

    conv6_3 = mx.symbol.FullyConnected(data=prelu6_sep, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group


bn_mom = 0.9
#bn_mom = 0.9997

def Act(data, act_type, name):
    #ignore param act_type, set it in this function 
    body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    #body = mx.sym.Activation(data=data, act_type='relu', name=name)
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), dilate=(1,1), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, dilate=dilate, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=bn_mom)
    act = Act(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act
    
def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=bn_mom)    
    return bn

def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    return conv    

    
def DResidual(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix=''):
    conv = Conv(data=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_sep' %(name, suffix))
    conv_dw = Conv(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name='%s%s_conv_dw' %(name, suffix))
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_proj' %(name, suffix))
    return proj
    
def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix=''):
    identity=data
    for i in range(num_block):
    	shortcut=identity
    	conv=DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name='%s%s_block' %(name, suffix), suffix='%d'%i)
    	identity=conv+shortcut
    return identity
	
res_base_dim_big = 64
def L106_Net112_big(mode="train"):
#def L106_Net112(mode="train"):
    """
    #Proposal Network
    #input shape 3 x 112 x 112
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    # data = 112X112
    # conv1 = 56X56
    conv1 = Conv(data, num_filter=res_base_dim_big, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv1")
    conv2 = Residual(conv1, num_block=2, num_out= res_base_dim_big, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim_big, name="res2")
    
	#conv23 = 28X28
    conv23 = DResidual(conv2, num_out=res_base_dim_big*2, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=res_base_dim_big*2, name="dconv23")
    conv3 = Residual(conv23, num_block=6, num_out=res_base_dim_big*2, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim_big*2, name="res3")
    
	#conv34 = 14X14
    conv34 = DResidual(conv3, num_out=res_base_dim_big*4, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=res_base_dim_big*4, name="dconv34")
    conv4 = Residual(conv34, num_block=10, num_out=res_base_dim_big*4, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim_big*4, name="res4")
    
	#conv45 = 7X7
    conv45 = DResidual(conv4, num_out=res_base_dim_big*8, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=res_base_dim_big*8, name="dconv45")
    conv5 = Residual(conv45, num_block=2, num_out=res_base_dim_big*8, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim_big*8, name="res5")
    
	# conv6 = 1x1
    conv6 = Conv(conv5, num_filter=res_base_dim_big*8, kernel=(7, 7), pad=(0, 0), stride=(1, 1), name="conv6")
    fc1 = Conv(conv6, num_filter=res_base_dim_big*8, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="fc1")
    fc2 = Conv(fc1, num_filter=res_base_dim_big*8, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="fc2")	
    conv6_3 = mx.symbol.FullyConnected(data=fc2, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group
	
res_base_dim = 8
#def L106_Net112_small(mode="train"):
def L106_Net112(mode="train"):
    """
    #Proposal Network
    #input shape 3 x 112 x 112
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    # data = 112X112
    # conv1 = 56X56
    conv1 = Conv(data, num_filter=res_base_dim, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv1")
    conv2 = Residual(conv1, num_block=1, num_out= res_base_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim, name="res2")
    
	#conv23 = 28X28
    conv23 = DResidual(conv2, num_out=res_base_dim*2, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=res_base_dim*2, name="dconv23")
    conv3 = Residual(conv23, num_block=2, num_out=res_base_dim*2, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim*2, name="res3")
    
	#conv34 = 14X14
    conv34 = DResidual(conv3, num_out=res_base_dim*4, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=res_base_dim*4, name="dconv34")
    conv4 = Residual(conv34, num_block=3, num_out=res_base_dim*4, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim*4, name="res4")
    
	#conv45 = 7X7
    conv45 = DResidual(conv4, num_out=res_base_dim*8, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=res_base_dim*8, name="dconv45")
    conv5 = Residual(conv45, num_block=2, num_out=res_base_dim*8, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim*8, name="res5")
    
	# conv6 = 1x1
    conv6 = Conv(conv5, num_filter=res_base_dim*8, kernel=(7, 7), pad=(0, 0), stride=(1, 1), name="conv6")
    fc1 = Conv(conv6, num_filter=res_base_dim*16, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="fc1")
    fc2 = Conv(fc1, num_filter=res_base_dim*32, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="fc2")	
    conv6_3 = mx.symbol.FullyConnected(data=fc2, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group
	
def L106_Net80(mode="train"):
    """
    #Proposal Network
    #input shape 3 x 80 x 80
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    
    # data = 80X80
    # conv1 = 40X40
    conv1 = Conv(data, num_filter=res_base_dim, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv1")
    conv2 = Residual(conv1, num_block=1, num_out= res_base_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim, name="res2")
    
	#conv23 = 20X20
    conv23 = DResidual(conv2, num_out=res_base_dim*2, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=res_base_dim*2, name="dconv23")
    conv3 = Residual(conv23, num_block=2, num_out=res_base_dim*2, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim*2, name="res3")
    
	#conv34 = 10X10
    conv34 = DResidual(conv3, num_out=res_base_dim*4, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=res_base_dim*4, name="dconv34")
    conv4 = Residual(conv34, num_block=3, num_out=res_base_dim*4, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim*4, name="res4")
    
	#conv45 = 5X5
    conv45 = DResidual(conv4, num_out=res_base_dim*8, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=res_base_dim*8, name="dconv45")
    conv5 = Residual(conv45, num_block=2, num_out=res_base_dim*8, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=res_base_dim*8, name="res5")
    
	# conv6 = 1x1
    conv6 = Conv(conv5, num_filter=res_base_dim*8, kernel=(5, 5), pad=(0, 0), stride=(1, 1), name="conv6")
    fc1 = Conv(conv6, num_filter=res_base_dim*16, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="fc1")
    fc2 = Conv(fc1, num_filter=res_base_dim*32, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="fc2")	
    conv6_3 = mx.symbol.FullyConnected(data=fc2, num_hidden=212, name="conv6_3")	
    bn6_3 = mx.sym.BatchNorm(data=conv6_3, name='bn6_3', fix_gamma=False,momentum=0.9)
    if mode == "test":
        landmark_pred = bn6_3
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=bn6_3, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark106', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group
	
def hourglass_block(data, data_dim, suffix=''):
    conv1 = Residual(data, num_block=1, num_out= data_dim, num_group=data_dim, kernel=(5,5), pad=(2,2), stride=(1,1), name="res1", suffix=suffix)
    # conv1 = 28x28
    conv12 = DResidual(conv1, num_out=data_dim*2, num_group=data_dim*2, kernel=(5, 5), pad=(2, 2),stride=(2, 2), name="dconv12", suffix=suffix)
    conv2 = Residual(conv12, num_block=1, num_out=data_dim*2, num_group=data_dim*2, kernel=(5, 5), pad=(2, 2), stride=(1, 1), name="res2", suffix=suffix)
    # conv2 = 14x14
    conv23 = DResidual(conv2, num_out=data_dim*2, num_group=data_dim*2, kernel=(5, 5), pad=(2, 2),stride=(2, 2), name="dconv23", suffix=suffix)
    conv3 = Residual(conv23, num_block=1, num_out=data_dim*2, num_group=data_dim*2, kernel=(5, 5), pad=(2, 2), stride=(1, 1), name="res3", suffix=suffix)
    # conv3 = 7x7
    conv3_up = mx.symbol.UpSampling(conv3,scale=2, sample_type='nearest', name="conv3_up%s"%(suffix))
    feat23 = mx.symbol.Concat(*[conv2,conv3_up],name="feat23%s"%(suffix))
    feat2_sep = Conv(feat23, num_filter=data_dim*2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="feat2_sep%s"%(suffix))
    feat2 = Residual(feat2_sep, num_block=1, num_out=data_dim*2, num_group=data_dim*2, kernel=(5, 5), pad=(2, 2), stride=(1, 1), name="feat_res2", suffix=suffix)
    # feat2 = 14x14
    feat2_up = mx.symbol.UpSampling(feat2,scale=2, sample_type='nearest', name="feat2_up%s"%(suffix))
    feat12 = mx.symbol.Concat(*[conv1,feat2_up],name="feat12%s"%(suffix))
    feat1_sep = Conv(feat12, num_filter=data_dim, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="feat1_sep%s"%(suffix))
    feat1 = Residual(feat1_sep, num_block=1, num_out=data_dim, num_group=data_dim, kernel=(5, 5), pad=(2, 2), stride=(1, 1), name="feat_res1", suffix=suffix)
    # feat1 = 28x28
    
    identity = data+feat1
    return identity
    
    
def HeatMapStage(data, out_channel, stage_id, in_feat = None):
    id = int(stage_id)
    conv1 = Conv(data, num_filter=heatmap_base_dim, kernel=(5, 5), pad=(4, 4), stride=(1, 1), dilate=(2,2), name="stg%d_conv1"%id)
	# conv1 = 112X112
    conv2_dw = Conv(conv1, num_filter=heatmap_base_dim, num_group= heatmap_base_dim, kernel=(7, 7), pad=(3, 3), stride=(1, 1), name="stg%d_conv2_dw"%id)
    # conv2_dw = 112X112
    conv2_sep = Conv(conv2_dw, num_filter=heatmap_base_dim*2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="stg%d_conv2_sep"%id)
    # conv2_sep = 112X112
	
    conv3_dw = Conv(conv2_sep, num_filter=heatmap_base_dim*2, num_group=heatmap_base_dim*2, kernel=(7, 7), pad=(3, 3), stride=(2, 2), dilate=(1,1), name="stg%d_conv3_dw"%id)
    # conv3_dw = 56X56
    conv3_sep = Conv(conv3_dw, num_filter=heatmap_base_dim*2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="stg%d_conv3_sep"%id)
    # conv3_sep = 56X56
    
    conv4_dw = Conv(conv3_sep, num_filter=heatmap_base_dim*2, num_group=heatmap_base_dim*2, kernel=(9, 9), pad=(8, 8), stride=(1, 1), dilate=(2,2), name="stg%d_conv4_dw"%id)
    # conv4_dw = 56X56
    conv4_sep = Conv(conv4_dw, num_filter=heatmap_base_dim*4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="stg%d_conv4_sep"%id)
    # conv4_sep = 56X56

    conv5_dw = Conv(conv4_sep, num_filter=heatmap_base_dim*4, num_group=heatmap_base_dim*4, kernel=(7, 7), pad=(3, 3), stride=(2, 2), dilate=(1,1), name="stg%d_conv5_dw"%id)
    # conv5_dw = 28X28
    conv5_sep = Conv(conv5_dw, num_filter=heatmap_base_dim*4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="stg%d_conv5_sep"%id)
    # conv5_sep = 28X28
    
    if id == 1:
        conv6_dw = Conv(conv5_sep, num_filter=heatmap_base_dim*4, num_group=heatmap_base_dim*4, kernel=(7, 7), pad=(6, 6), stride=(1, 1), dilate=(2,2), name="stg%d_conv6_dw"%id)
        # conv6_dw = 28X28
        conv6_sep = Conv(conv6_dw, num_filter=heatmap_base_dim*4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="stg%d_conv6_sep"%id)
        # conv6_sep = 28X28
    else:
        concat = mx.symbol.Concat(*[conv5_sep,in_feat],name="stg%d_concat"%id)
        conv6_sep = Conv(concat, num_filter=heatmap_base_dim*4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="stg%d_conv6_sep"%id)
        # conv6_sep = 28X28
    
	
    conv7_dw = Conv(conv6_sep, num_filter=heatmap_base_dim*4, num_group=heatmap_base_dim*4, kernel=(7, 7), pad=(3, 3), stride=(1, 1), dilate=(1,1), name="stg%d_conv7_dw"%id)
    # conv7_dw = 28X28
    conv7_sep = Conv(conv7_dw, num_filter=heatmap_base_dim*8, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="stg%d_conv7_sep"%id)
    # conv7_sep = 28X28
	
    conv8_dw = Conv(conv7_sep, num_filter=heatmap_base_dim*8, num_group=heatmap_base_dim*8, kernel=(7, 7), pad=(3, 3), stride=(1, 1), dilate=(1,1), name="stg%d_conv8_dw"%id)
    # conv8_dw = 28X28
    conv8_sep = Conv(conv8_dw, num_filter=heatmap_base_dim*16, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="stg%d_conv8_sep"%id)
    # conv8_sep = 28X28
	
    feat = ConvOnly(conv8_sep, num_filter=out_channel, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="stg%d_feat"%id)
    feat_bn = mx.sym.BatchNorm(data=feat, name='stg%d_feat_bn'%id, fix_gamma=False,momentum=0.9)
	
    return feat_bn

heatmap_base_dim = 8
def L14_Net112_heatmap(mode="train"):
    """
    #Proposal Network
    #input shape 3 x 112 x 112
    """
    data = mx.symbol.Variable(name="data")
    landmark_target = mx.symbol.Variable(name="landmark_target")
    # data = 112X112
    
    stg1_feat = HeatMapStage(data, 15, 1)
    if config.HeatMapStage > 1:
        stg2_feat = HeatMapStage(data, 15, 2, stg1_feat)
    if config.HeatMapStage > 2:
        stg3_feat = HeatMapStage(data, 15, 3, stg2_feat)
    if config.HeatMapStage > 3:
        stg4_feat = HeatMapStage(data, 15, 4, stg3_feat)
    if config.HeatMapStage > 4:
        stg5_feat = HeatMapStage(data, 15, 5, stg4_feat)
    if config.HeatMapStage > 5:
        stg6_feat = HeatMapStage(data, 15, 6, stg5_feat)
    if config.HeatMapStage > 6:
        stg7_feat = HeatMapStage(data, 15, 7, stg6_feat)
    if config.HeatMapStage > 7:
        stg8_feat = HeatMapStage(data, 15, 8, stg7_feat)
	
    if config.HeatMapStage == 1:
        heatmap = stg1_feat
    elif config.HeatMapStage == 2:
        heatmap = mx.sym.Concat(*[stg1_feat,stg2_feat],name='feat_concat')
    elif config.HeatMapStage == 3:
        heatmap = mx.sym.Concat(*[stg1_feat,stg2_feat,stg3_feat],name='feat_concat')
    elif config.HeatMapStage == 4:
        heatmap = mx.sym.Concat(*[stg1_feat,stg2_feat,stg3_feat,stg4_feat],name='feat_concat')
    elif config.HeatMapStage == 5:
        heatmap = mx.sym.Concat(*[stg1_feat,stg2_feat,stg3_feat,stg4_feat,stg5_feat],name='feat_concat')
    elif config.HeatMapStage == 6:
        heatmap = mx.sym.Concat(*[stg1_feat,stg2_feat,stg3_feat,stg4_feat,stg5_feat,stg6_feat],name='feat_concat')
    elif config.HeatMapStage == 7:
        heatmap = mx.sym.Concat(*[stg1_feat,stg2_feat,stg3_feat,stg4_feat,stg5_feat,stg6_feat,stg7_feat],name='feat_concat')
    elif config.HeatMapStage == 8:
        heatmap = mx.sym.Concat(*[stg1_feat,stg2_feat,stg3_feat,stg4_feat,stg5_feat,stg6_feat,stg7_feat,stg8_feat],name='feat_concat')
    
    heatmap_flat = mx.sym.Flatten(heatmap)
    if mode == "test":
        landmark_pred = heatmap_flat
        group = mx.symbol.Group([landmark_pred])
    else:
        
        landmark_pred = mx.symbol.LinearRegressionOutput(data=heatmap_flat, label=landmark_target,
                                                 grad_scale=1, name="landmark_pred")
        out = mx.symbol.Custom(landmark_pred=landmark_pred, landmark_target=landmark_target, 
                            op_type='negativemining_onlylandmark14_heatmap', name="negative_mining")
        group = mx.symbol.Group([out])
        
    return group