# GAN-SFTM
# A Flexible Reference-Insensitive Spatiotemporal Fusion Model for Remote Sensing Images Using Conditional Generative Adversarial Network
[A Flexible Reference-Insensitive Spatiotemporal Fusion Model for Remote Sensing Images Using Conditional Generative Adversarial Network](https://ieeexplore.ieee.org/abstract/document/9336033)
## 致力于帮助理解代码~
## Models
The main branch is the implementation of the GAN-STFM model.

## Environment:

Tested on the following environment:

Python: >=3.6

PyTorch: >=0.4

pip install torchgan

pip install rasterio

pip install tqdm  #用来显示进度的

# DataSat
CIA:  https://data.csiro.au/collection/csiro:5846   

LGC:  https://data.csiro.au/collection/csiro:5847v3

![图像1](screenshots/Data-Directory-1.png)
![图像2](screenshots/Data-Directory-2.png)
# Console
运行时后需要的一些参数，根据实际情况来调整


tif格式图像可以使用QGIS软件打开查看


modis是高时间分辨率图像

landsat是高空间分辨率图像

你可以直接在命令行中使用：

```shell
python run.py --lr 0.0002 --num_workers 12 --batch_size 8 --epochs 300 --cuda --ngpu 2 --image_size 2720 3200 --save_dir /home/paul623/datasets/STFusion/RunLog/ --data_dir /home/paul623/datasets/STFusion/LGC/LGC_data/
```
或者在PyCharm里面指定参数
```shell
--lr
0.0002
--num_workers
12
--batch_size
4
--epochs
1
--cuda
--ngpu
2
--image_size
2720
3200
--save_dir
/home/paul623/datasets/STFusion/RunLog/
--data_dir
/home/paul623/datasets/STFusion/LGC/LGC_data/
--patch_size
80
80
```
爆显存很常见，请根据自己的测试设备实际情况来调整

patch_size要能够被长宽整除，否则在test阶段会报错按照给的实例来就好




# 测试时候的输入
```
2005_045_0214-2005_061_0302中，有两组照片（一张低分，一张高分，对应）
pred: 2005_061_0302
ref: 2005_045_0214
inputs=[modis:2005061, landsat:20050214]
所以预测的是2005061，用20050214这张landsat作为参考。
最后计算预测与实际的差值：modis_2005061和landsat_20050302
```
