# import pytorch_msssim
# import torch
# import tifffile
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# m = pytorch_msssim.MSSSIM()
#
# img1 = tifffile.imread("/home/byt/Fusion/ganstfm-main/ganstfm-main/run_LGC_C1/test/PRED_2005_029_0129-2005_013_0113.tif")
# img2 = tifffile.imread("/home/byt/Fusion/ganstfm-main/ganstfm-main/dataset/LGC_data/refs/20050129_TM.tif")
#
# # img1 = torch.rand(1, 1, 256, 256)
# # img2 = torch.rand(1, 1, 256, 256)
#
# print(pytorch_msssim.msssim(img1, img2))
# print(m(img1, img2))

from pytorch_msssim import ms_ssim
import tifffile
import torch

prediction = tifffile.imread("/home/byt/Fusion/ganstfm-main/ganstfm-main/run_LGC_C2/test/PRED_2005_045_0214-2005_029_0129.tif")
target = tifffile.imread("/home/byt/Fusion/ganstfm-main/ganstfm-main/dataset/LGC_data/refs/20050129_TM.tif")

prediction = torch.Tensor(prediction)
target = torch.Tensor(target)

prediction = prediction.unsqueeze(0)
target = target.unsqueeze(0)

prediction = prediction.permute(0, 3, 1, 2)
target = target.permute(0, 3, 1, 2)

print(prediction.size())
print(ms_ssim(prediction, target))