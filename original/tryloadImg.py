import glob
import os.path
import shutil
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np

# # 读取并处理的图像数据
# img_dir = r"/home/zbl/datasets/STFusion/CIA/modis/MOD09GA_A2001290.sur_refl.int.tif"
# ds = gdal.Open(img_dir)
#
# if ds is not None:
#     # 读取图像数据
#     num_bands = ds.RasterCount
#     image_data = np.zeros((ds.RasterYSize, ds.RasterXSize, num_bands), dtype=np.uint8)
#     for band_num in range(num_bands):
#         band = ds.GetRasterBand(band_num + 1)
#         image_data[:, :, band_num] = band.ReadAsArray()
#
#     # 如果图像通道数为6，将其转换为RGB
#     if num_bands == 6:
#         rgb_image = image_data[:, :, [3, 2, 1]]  # 选择通道顺序
#
#         # 显示图像
#         plt.imshow(rgb_image)
#         plt.title('Your Image Title')
#         plt.axis('off')  # 不显示坐标轴
#         plt.show()
#     else:
#         print('图像通道数不是6，无法显示。')
# else:
#     print('无法打开图像文件')
data_dir = r"/home/zbl/datasets/STFusion/CIA/data_cia/val/"

tm_files = glob.glob(os.path.join(data_dir, '*TM.tif'))
mod_files = glob.glob(os.path.join(data_dir, '*refl.tif'))

