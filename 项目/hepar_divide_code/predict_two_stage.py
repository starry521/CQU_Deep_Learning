import  os
import glob
import time
import torch
from network.get_model import get_model
from monai.transforms import  Compose,Resized,LoadImaged,EnsureChannelFirstd,ScaleIntensityRanged,\
                            EnsureTyped,CropForeground,SpatialCrop,CopyItemsd
from monai.data import Dataset, CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from utils.post_ways import post_ways
import SimpleITK as sitk
import numpy as np
import argparse
import warnings

warnings.filterwarnings('ignore')   # 禁止在控制台打印warnning信息

#可推理的阶段: 两阶段方法coarse + fine
def predict_two_stage(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transfrom
    test_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=args.win_level[0], a_max=args.win_level[1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CopyItemsd(keys=["image"], times=1, names=["resize_image"]),  # 复制
        Resized(keys="resize_image", spatial_size=args.Roi_size, mode="trilinear"),
        EnsureTyped(keys=["image", "resize_image"])
    ])

    # Path
    image_paths = sorted(glob.glob(os.path.join(r"./predict/origin", args.input_path, "*.nii.gz")))
    test_data_dicts = [{"image": image_name} for image_name in zip(image_paths)]
    test_dataset = Dataset(test_data_dicts, test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4)

    # 模型加载(.pth)
    coarse_model = fine_model = get_model(num_classes=args.num_classes, input_size=args.Roi_size, win_level=args.win_level).to(device)
    coarse_model.load_state_dict(torch.load(args.weight_coarse, map_location=device))
    fine_model.load_state_dict(torch.load(args.weight_fine, map_location=device))

    #ckpt加载方法
    # ckpt_coarse = torch.load(fine_model_weight_path)
    # coarse_model.load_state_dict({k.replace('_model.', ''): v for k, v in ckpt_coarse['state_dict'].items()})
    # ckpt_fine = torch.load(fine_model_weight_path)
    # fine_model.load_state_dict({k.replace('_model.', ''): v for k, v in ckpt_fine['state_dict'].items()})

    coarse_model.eval()
    fine_model.eval()

    # 后处理
    post_label, post_pred = post_ways(num_classes=args.num_classes)

    with torch.no_grad():
        
        count = 0

        for _, batch in enumerate(test_dataloader):
            
            time1 = time.time()

            origin_images, resized_image = batch["image"].to(device), batch["resize_image"].to(device)
            print("原图像:",origin_images.shape)

            # 保存路径
            suffix_img = batch["image_meta_dict"]["filename_or_obj"]
            suffix = suffix_img[0].split("\\")[-1]
            save_image_path = args.output_path
            save_image = save_image_path + suffix

            # 直接predict, crop采用滑窗
            coarse_outputs = sliding_window_inference(resized_image, args.Roi_size, 4, coarse_model)

            # sigmoid or softmax,输出都是一个通道,softmax也经过argmax了
            coarse_outputs = [post_pred(i) for i in decollate_batch(coarse_outputs)]  # 多分类，softmax
            
            # 根据一阶段结果，裁剪原图，主要是获得ROI区域的坐标
            # result[0]是裁剪出的前景图像，result[1]是坐标最小值，result[1]是坐标最大值
            result = CropForeground(select_fn=lambda x: x > 0, return_coords=True, margin=2)(coarse_outputs[0])
            
            # Resized缩放尺度
            H_factor = args.Roi_size[0] / origin_images.shape[2]
            W_factor = args.Roi_size[1] / origin_images.shape[3]
            D_factor = args.Roi_size[2] / origin_images.shape[4]

            # 根据result坐标确定裁剪box去裁剪原图
            bbox = np.array([[np.int(np.ceil(result[1][0] / H_factor)), np.int(np.ceil(result[2][0] / H_factor))],
                             [np.int(np.ceil(result[1][1] / W_factor)), np.int(np.ceil(result[2][1] / W_factor))],
                             [np.int(np.ceil(result[1][2] / D_factor)), np.int(np.ceil(result[2][2] / D_factor))],
            ])
            
            # 裁剪原图
            stage2_image = SpatialCrop(roi_start=[bbox[0][0], bbox[1][0], bbox[2][0]],
                                       roi_end = [bbox[0][1], bbox[1][1], bbox[2][1]])(origin_images[0])
            
            print("裁剪前景后:",stage2_image.shape)

            # 精分割使用滑窗推理
            fine_outputs = sliding_window_inference(stage2_image.unsqueeze(0), args.Roi_size, 4, fine_model)

            # sigmoid or softmax
            fine_outputs = [post_pred(i) for i in decollate_batch(fine_outputs)]

            # 将预测的结果填回原图大小
            outputs = torch.zeros(1, 1, origin_images.shape[2], origin_images.shape[3], origin_images.shape[4], device=device)

            outputs[0][0][bbox[0][0]:bbox[0][1],
                          bbox[1][0]:bbox[1][1],
                          bbox[2][0]:bbox[2][1]] = fine_outputs[0][0]

            # Save
            dicom = sitk.ReadImage(suffix_img[0])
            output = sitk.GetImageFromArray(outputs[0][0].detach().cpu().numpy().transpose(2, 1, 0))
            output.SetOrigin(dicom.GetOrigin())
            output.SetSpacing(dicom.GetSpacing())
            output.SetDirection(dicom.GetDirection())
            sitk.WriteImage(output, save_image)
            torch.cuda.empty_cache()

            count += 1
            time2 = time.time()  # 验证结束时间

            # 打印此轮预测结果
            print("Save: {}/{},  time:{}".format(count, len(image_paths), time2-time1))

        print("Over Predict !")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--win_level", default=[-300, 300], type=int, help="调窗范围")
    parser.add_argument("--input_path", default="image", type=str, help="预测数据集")
    parser.add_argument("--output_path", default="./predict/origin/predict_out/", type=str, help="输出路径")
    parser.add_argument("--weight_coarse", default="./weights/two_stage/best_model_coarse.pth", type=str, help="粗分割模型参数路径")
    parser.add_argument("--weight_fine", default="./weights/two_stage/best_model_fine.pth", type=str, help="细分割模型参数路径")
    parser.add_argument("--num_classes", default=1, type=int, help="分类")
    parser.add_argument("--Roi_size", default=[128, 128, 128], type=int, help="Roi_size=Resized_size=Input_size")
    args = parser.parse_args()
    predict_two_stage(args=args)


if __name__ == '__main__':
    main()

