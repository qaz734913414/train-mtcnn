set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_L_net_onlylandmark106_64_offline.py --gpus 0 --begin_epoch 170 --epoch 170 --resume --lr 0.1 --image_set landmark106_64 --pretrained model/lnet106_64 --prefix model/lnet106_64 --end_epoch 60000 --lr_epoch 200,4000,20000 --batch_size 1000 --thread_num 10 --frequent 10
pause 