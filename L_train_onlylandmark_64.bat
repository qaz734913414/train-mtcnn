set MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python example\train_L_net_onlylandmark_64.py --lr 0.01 --image_set img_cut_celeba_all --end_epoch 5000 --pretrained model/lnet64_v3 --prefix model/lnet64_v3 --lr_epoch 2000,4000 --batch_size 100 --thread_num 10 --frequent 100 
pause 