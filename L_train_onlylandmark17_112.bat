set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_L_net_onlylandmark17_112.py --lr 0.01 --image_set cut_train2017_anno --end_epoch 60000 --lr_epoch 5000,10000,60000 --pretrained model/lnet17_112 --prefix model/lnet17_112 --batch_size 100 --thread_num 10 --frequent 10 

pause 