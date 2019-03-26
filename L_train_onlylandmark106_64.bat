set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_L_net_onlylandmark106_64.py --lr 0.01 --image_set 106data_merge --end_epoch 8000 --lr_epoch 2500,5000,8000 --batch_size 100 --thread_num 10 --frequent 10 
pause 