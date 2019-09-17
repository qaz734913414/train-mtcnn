set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_L_net_onlylandmark106_80_offline.py --gpus 1 --begin_epoch 570 --epoch 570 --resume --lr 0.1 --image_set landmark106 --pretrained model/lnet106_80 --prefix model/lnet106_80 --end_epoch 60000 --lr_epoch 200,4000,20000 --batch_size 1000 --thread_num 10 --frequent 10
pause 