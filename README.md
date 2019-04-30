train mtcnn: a modified version by Zuo Qing from https://github.com/Seanlinx/mtcnn

训练环境windows 7/10, 其他环境未测试

**计算量统计规则：fma只算一条指令，均为merge_bn之后的计算量**

**十种Pnet20（原版Pnet等价计算量为45.4M）**

| 模型名称                                                        | 输入尺寸     | 计算量（不计bbox）| 训练时精度      | pooling个数 |  备注                |
| --------                                                        | ------       | ------------      | -----------     | ----------- | -------------------- |
| [Pnet20_v00](https://pan.baidu.com/s/1g7JnOxnbXIbNWPXGI-IzrQ)   | 320x240      | 10.3 M            | 0.888-0.894     |     0       | 对标libfacedetection |
| [Pnet20_v0](https://pan.baidu.com/s/1r3VcmEX1a2C5gKlGKnC4kw)    | 320x240      | 14.1 M            | 0.900-0.908     |     0       | 对标libfacedetection |
| [Pnet20_v1](https://pan.baidu.com/s/1qVU3_nporbOUzXYu7giZkA)    | 320x240      | 18.3 M            | 0.915-0.920     |     0       |                      |
| [Pnet20_v2](https://pan.baidu.com/s/1bXzdmsTgfqU_TJHsozSmrQ)    | 320x240      | 22.9 M            | 0.928-0.933     |     0       | 对标原版pnet         |
| ~~Pnet20_v3~~                                                   | ~~320x240~~  | ~~32.4 M~~        | ~~0.930-0.935~~ |     ~~1~~   | ~~性价比不如v2~~     |
| Pnet20_v4                                                       | 320x240      | 55.8 M            | 0.945-0.950     |     0       |                      |
| ~~Pnet20_v5~~                                                   | ~~320x240~~  | ~~83.9 M~~        | ~~0.945-0.950~~ |     ~~1~~   | ~~不建议使用~~       |
| Pnet20_v6                                                       | 320x240      | 92.4 M            | 0.952-0.957     |     0       |                      |
| ~~Pnet20_v7~~                                                   | ~~320x240~~  | ~~102.1 M~~       | ~~0.952-0.958~~ |     ~~2~~   | ~~性价比不如v6~~     |
| Pnet20_v8                                                       | 320x240      | 125.2 M           | 0.954-0.958     |     0       |                      |

**三种Pnet20_s2（训练用的样本与上面并不相同）**

| 模型名称       | 输入尺寸     | 计算量（不计bbox）| 训练时精度      | pooling个数 |  备注                |
| --------       | ------       | ------------      | -----------     | ----------- | -------------------- |
| Pnet20_s2v1    | 320x240      | 63.2 M            | 待测            |     0       |        stride=2      |
| Pnet20_s2v2    | 320x240      | 107.5 M           | 待测            |     0       |        stride=2      |
| Pnet20_s2v3    | 320x240      | 169.2 M           | 0.940-0.946     |     0       |        stride=2      |

**两种Pnet16（训练用的样本与上面并不相同）**

| 模型名称                                                        | 输入尺寸     | 计算量（不计bbox）| 训练时精度      | pooling个数 |  备注                |
| --------                                                        | ------       | ------------      | -----------     | ----------- | -------------------- |
| [Pnet16_v0](https://pan.baidu.com/s/1s5eZLeAKnqp1ZDTrzaOD_w)    | 256x192      | 8.6 M             | 0.857-0.865     |     0       |         stride=4     |
| [Pnet16_v1](https://pan.baidu.com/s/1Lf0z6rRq5WUKE_DMze_C7w)    | 256x192      | 12.5 M            | 0.873-0.879     |     0       |         stride=4     |


**两种Rnet（原版Rnet计算量为1.6M）**

| 模型名称                                                      | 输入尺寸   | 计算量           | 训练时精度      | pooling个数 |  备注                |
| --------                                                      | ------     | ------------     | -----------     | ----------- | -------------------- |
| [Rnet_v1](https://pan.baidu.com/s/1SEIolnvmtPvdqbHxU1vPWQ)    | 24x24      | 0.6 M            | 0.943-0.948     |     0       | 对标原版Rnet         |
| [Rnet_v2](https://pan.baidu.com/s/1APWYGcFC5MAn6Ba5vWo80w)    | 24x24      | 1.6 M            | 0.957-0.962     |     0       |                      |

**三种Onet（原版Onet计算量为12.9M）**

| 模型名称                                                      | 输入尺寸   | 计算量           | 训练时精度      | pooling个数 |  备注                |
| --------                                                      | ------     | ------------     | -----------     | ----------- | -------------------- |
| [Onet_v1](https://pan.baidu.com/s/1UTvSKErOul2wkT5EMxXgVA)    | 48x48      | 2.4 M            | 0.947-0.954     |     0       | 不含landmark         |
| [Onet_v2](https://pan.baidu.com/s/19QomSIy3Py516OEIBFDcVg)    | 48x48      | 3.6 M            | 0.961-0.967     |     0       | 不含landmark         |
| Onet_v3                                                       | 48x48      | 9.3 M            | 0.979-0.985     |     0       | 不含landmark         |

**两种Lnet（原版Onet计算量为12.9M）**

| 模型名称                                                      | 输入尺寸   | 计算量            | 训练时L2   | 训练时L1    |  备注                |
| --------                                                      | ------     | ------------      | -----------| ----------- | -------------------- |
| Lnet_v1                                                       | 48x48      |  3.8 M            | 约0.0021   | 约0.032     | lnet_basenum=16      |
| Lnet_v1                                                       | 48x48      | 11.5 M            | 约0.0016   | 约0.026     | lnet_basenum=32      |
| [Lnet_v2](https://pan.baidu.com/s/1W6bxNeD0psxwxbou_xwK-g)    | 48x48      |  3.8 M            | 约0.0014   | 约0.027     | lnet_basenum=16      |
| [Lnet_v2](https://pan.baidu.com/s/1e3tuwrR3AoU_zRKkIFK8xg)    | 48x48      | 11.5 M            | 约0.0012   | 约0.025     | lnet_basenum=32      |
| Lnet_v3                                                       | 64x64      |  7.7 M            | 约0.0011   | 约0.023     | lnet_basenum=32      |

**两种Lnet106**

| 模型名称      | 输入尺寸   | 计算量            | 训练时L2      | 训练时L1    |  备注                   |
| --------      | ------     | ------------      | -----------   | ----------- | --------------------    |
| Lnet106_v1    | 48x48      | 11.6 M            | 待测          | 待测        | lnet106_basenum=32      |
| Lnet106_v1    | 48x48      | 38.8 M            | 待测          | 待测        | lnet106_basenum=64      |
| [Lnet106_v2](https://pan.baidu.com/s/1D3G3oGzxODPw8dZahqiNIA)    | 48x48      | 11.6 M            | 0.0030-0.0036 | 0.040-0.043 | lnet106_basenum=32      |
| [Lnet106_v2](https://pan.baidu.com/s/1Ym_N07hJZqc_jFlgXByDHQ)    | 48x48      | 38.8 M            | 0.0019-0.0024 | 0.032-0.036 | lnet106_basenum=64      |
| Lnet106_v2    | 48x48      | 140.1 M           | 0.0017-0.0024 | 0.032-0.036 | lnet106_basenum=128     |

**两种Lnet106_96**

| 模型名称      | 输入尺寸   | 计算量(merge bn之后) | 训练时L2      | 训练时L1    |  备注                   |
| --------      | ------     | ------------         | -----------   | ----------- | --------------------    |
| Lnet106_96_v1 | 96x96      | 42.8 M               | 0.0014-0.0019 | 0.028-0.032 | lnet106_basenum=32      |
| Lnet106_96_v1 | 96x96      | 140.2 M              | 待测          | 待测        | lnet106_basenum=64      |
| Lnet106_96_v2 | 96x96      | 42.8 M               | 0.0013-0.0017 | 0.028-0.031 | lnet106_basenum=32      |
| Lnet106_96_v2 | 96x96      | 140.2 M              | 0.0012-0.0015 | 0.027-0.030 | lnet106_basenum=64      |

**使用数据106data_merge训练**

| 模型名称      | 输入尺寸   | 计算量(merge bn之后) | 训练时L2      | 训练时L1    |  备注                   |
| --------      | ------     | ------------         | -----------   | ----------- | --------------------    |
| Lnet106_96_v1 | 96x96      | 42.8 M               | 0.0010-0.0015 | 0.024-0.028 | lnet106_basenum=32      |
| Lnet106_96_v1 | 96x96      | 140.2 M              | 待测          | 待测        | lnet106_basenum=64      |
| Lnet106_96_v2 | 96x96      | 14.6 M               | 0.0015-0.0021 | 0.030-0.034 | lnet106_basenum=16      |
| Lnet106_96_v2 | 96x96      | 42.8 M               | 0.0011-0.0015 | 0.025-0.030 | lnet106_basenum=32      |
| Lnet106_96_v2 | 96x96      | 140.2 M              | 0.0009-0.0013 | 0.023-0.028 | lnet106_basenum=64      |
| Lnet106_96_v3 | 96x96      | 18.7 M               | 0.0014-0.0020 | 0.028-0.032 | lnet106_basenum=32      |

| 模型名称      | 输入尺寸   | 计算量(merge bn之后) | 训练时L2      | 训练时L1    |  备注                   |
| --------      | ------     | ------------         | -----------   | ----------- | --------------------    |
| Lnet106_64_v3 | 64x64      | 7.8 M                | 0.0026-0.0034 | 0.036-0.042 | lnet106_basenum=32      |

**使用数据106data_merge_migu训练**

| 模型名称      | 输入尺寸   | 计算量(merge bn之后) | 训练时L2      | 训练时L1    |  备注                   |
| --------      | ------     | ------------         | -----------   | ----------- | --------------------    |
| Lnet106_64_v3 | 64x64      | 7.8 M                | 0.0020-0.0027 | 0.032-0.037 | lnet106_basenum=32      |
| Lnet106_96_v3 | 96x96      | 18.7 M               | 待测          | 待测        | lnet106_basenum=32      |

# 基本说明

**(1)请使用[ZQCNN_MTCNN](https://github.com/zuoqing1988/ZQCNN)来进行forward**

**(2)Pnet改为Pnet20需要在你的MTCNN中更改cell_size=20, stride=4**

	1920*1080图像找20脸，第一层Pnet20_v0输入尺寸1920x1080，计算量324.6M，原版Pnet输入1152x648，计算量1278.0M

**(3)Rnet保持size=24不变，v1计算量约为原版1/3**

**(4)Onet带landmark我没有训练成功过**

**(5)Lnet是专门训练landmark的**

# 训练建议

**(1)下载[WIDER_train](https://pan.baidu.com/s/1PSR11Xs8lWmtVazCGoYR7Q)解压到data文件夹**

	解压之后目录为data/WIDER_train/images

**(2)双击gen_anno_and_train_list.bat**

	生成prepare_data/wider_annotations/anno.txt和data/mtcnn/imglists/train.txt

## 训练Pnet20 

**(3)双击P20_gen_data.bat**

	生成训练Pnet20所需样本
	
**(4)双击P20_gen_imglist.bat**

	生成训练Pnet20的list文件

**(5)双击P20_train.bat**

	训练Pnet20
	
**(6)双击P20_gen_hard_example.bat**

	利用训练得到的Pnet20模型，生成用于进一步训练Pnet20的hard样本，请用文本方式打开，酌情填写参数
		
**(7)双击P20_gen_imglist_with_hard.bat**

	生成用于进一步训练Pnet20的list文件
	
**(8)双击P20_train_with_hard.bat**
	
	进一步训练Pnet20
	
## 训练Rnet

**(9)双击R_gen_data.bat**

	生成训练Rnet所需样本
	
**(10)双击R_gen_hard_example.bat**
	
	利用训练得到的Pnet20模型，生成用于训练Rnet的hard样本，请用文本方式打开，酌情填写参数
	
**(11)双击R_gen_imglist_with_hard.bat**

	生成用于训练Rnet的list文件
	
**(12)双击R_train_with_hard.bat**

	训练Rnet
	
## 训练Onet

**(13)双击O_gen_data.bat**

	生成训练Onet所需样本
	
**(14)双击O_gen_hard_example.bat**
	
	利用训练得到的Pnet20、Rnet模型，生成用于训练Onet的hard样本，请用文本方式打开，酌情填写参数
	
**(15)双击O_gen_imglist_with_hard.bat**

	生成用于训练Onet的list文件
	
## 不带landmark
**(16)双击O_train_with_hard.bat**

	训练Onet
	
## 带landmark

下载[img_cut_celeba](https://pan.baidu.com/s/1XeGsYT_6VCP3n177oa3KGw)，解压到data/img_cut_celeba

图片位置在data/img_cut_celeba/xx.jpg

**(17)双击O_gen_landmark.bat**

	生成训练Onet所需landmark样本

**(18)双击O_gen_imglist_with_hard_landmark.bat**

	生成用于训练Onet的list文件

## 单独训练landmark

**(19)双击L_train.bat**

	训练Lnet
	
# 省硬盘的方式训练landmark

选择以下三个数据集之一：(A)是原始celeba数据，(B)(C)是我加工过的，加载速度B>C>A，（**我推荐用C，理论上用C应该和用A训练出来的结果一样**）

(A)[img_celeba](https://pan.baidu.com/s/1f6lYVNVYR7h28Vh-1nIPnQ)，解压到data/img_celeba

图片位置在data/img_celeba/xx.jpg

以文本方式编辑 L_train_onlylandmark.bat, 设置参数--image_set img_celeba_all

修改config.py中config.landmark_img_set='img_celeba'

双击 L_train_onlylandmark.bat 运行

(B)[img_align_celeba](https://pan.baidu.com/s/1rUBW8NasLZGtfQ33uA6Kdg)，解压到data/img_align_celeba

图片位置在data/img_align_celeba/xx.jpg

以文本方式编辑 L_train_onlylandmark.bat, 设置参数--image_set img_align_celeba_good

修改config.py中config.landmark_img_set='img_align_celeba'

双击 L_train_onlylandmark.bat 运行

(C)[img_cut_celeba](https://pan.baidu.com/s/1XeGsYT_6VCP3n177oa3KGw)，解压到data/img_cut_celeba

图片位置在data/img_cut_celeba/xx.jpg

以文本方式编辑 L_train_onlylandmark.bat, 设置参数--image_set img_cut_celeba_all

修改config.py中config.landmark_img_set='img_cut_celeba'

双击 L_train_onlylandmark.bat 运行

**备注：调整minibatch_onlylandmark.py里的参数得到的landmark精度不一样**

# 训练106点landmark


下载[Training_data106](https://pan.baidu.com/s/1SCdyksAWRSvhWCOJ4Syk1A)解压到data/Training_data106

解压后目录结构应为

	data/Training_data106/AFW
	data/Training_data106/HELEN
	data/Training_data106/IBUG
	data/Training_data106/LFPW
	
将 data/Training_data106/landmark106.txt拷贝到data/mtcnn/imglists/landmark106.txt， 在config.py设置

	config.landmark_img_set = 'Training_data106'
	
双击L_train_onlylandmark106.bat开始训练

**新的数据[106data_merge](https://pan.baidu.com/s/1PsOFqZeQoFK06MlPmgZxFw)**

解压后目录结构应为

	data/106data_merge/AFW
	data/106data_merge/HELEN
	data/106data_merge/IBUG
	data/106data_merge/LFPW
	data/106data_merge/clean0
	
将 data/106data_merge/106data_merge.txt拷贝到data/mtcnn/imglists/106data_merge.txt， 在config.py设置

	config.landmark_img_set = '106data_merge'
	
更改L_train_onlylandmark106.bat中参数--image_set landmark106 为 --image_set 106data_merge

双击L_train_onlylandmark106.bat开始训练

