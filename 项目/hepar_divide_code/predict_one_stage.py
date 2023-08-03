import  os
import glob
import SimpleITK as sitk
import torch
from network.get_model import get_model
from monai.data import DataLoader, CacheDataset, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from utils.post_ways import post_ways
from utils.test_transforms import transforms
import  argparse
import time
import warnings

warnings.filterwarnings('ignore')   # 禁止在控制台打印warnning信息


def predict_one_stage(args):
    
    # 定义设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 预测Transform
    test_transforms = transforms(args.win_level)

    # 加载test图像数据
    image_paths = sorted(glob.glob(os.path.join("./predict/resized", args.input_path, "*.nii.gz")))
    test_data_dicts = [{"image": image_name} for image_name in zip(image_paths)]
    test_dataset = Dataset(test_data_dicts, test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4)

    # 加载模型和参数
    model = get_model(num_classes=args.num_classes,input_size=args.Roi_size, win_level=args.win_level).to(device)
    model.load_state_dict(torch.load(args.weight_path, map_location=device))

    # ckpt加载方法
    # ckpt_coarse = torch.load(args.weight_path)
    # model.load_state_dict({k.replace('_model.', ''): v for k, v in ckpt_coarse['state_dict'].items()})

    model.eval()

    # 后处理
    post_label, post_pred = post_ways(num_classes=args.num_classes)

    with torch.no_grad():  # 不计算梯度,更加节约显存

        count = 0

        for _,batch  in enumerate(test_dataloader):

            time1 = time.time()

            # 获得test图像数据
            images = batch['image'].to(device)
            
            # 获取图像路径列表
            suffix_img = batch["image_meta_dict"]["filename_or_obj"]

            # 保存预测结果路径
            suffix = suffix_img[0].split("\\")[-1]  # 获取文件名
            save_image_path = args.output_path
            save_image = save_image_path + suffix  # 保存路径

            # Predict
            outputs = sliding_window_inference(images, args.Roi_size, 4, model)
 
            #多分类还是二分类
            # one_stage_outputs = post_pred(outputs)  #二分类，sigmoid
            one_stage_outputs = [post_pred(i) for i in decollate_batch(outputs)]  # 多分类，softmax


            # 保存预测结果
            dicom = sitk.ReadImage(suffix_img[0])
            output = sitk.GetImageFromArray(one_stage_outputs[0][0].detach().cpu().numpy().transpose(2,1,0))  # [z,y,x]->[x,y,z]
            output.SetOrigin(dicom.GetOrigin())  # 原点
            output.SetSpacing(dicom.GetSpacing())  # 体素大小
            output.SetDirection(dicom.GetDirection()) # 轴方向
            sitk.WriteImage(output, save_image)

            del images, outputs, one_stage_outputs, output
            torch.cuda.empty_cache()  # 清空缓存

            count += 1
            time2 = time.time()  # 验证结束时间

            # 打印此轮预测结果
            print("Save: {}/{},  time:{}".format(count, len(image_paths), time2-time1))

        print("Over Predict !")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--win_level", default=[-300, 300], type=int, help="调窗范围")
    parser.add_argument("--input_path", default="image", type=str, help="预测数据集路径")
    parser.add_argument("--output_path", default="./predict/resized/predict_out/", type=str, help="预测结果输出路径")
    parser.add_argument("--weight_path", default="./weights/two_stage/best_model_coarse.pth", type=str, help="模型参数加载路径")
    parser.add_argument("--num_classes", default=1, type=int, help="分类数")
    parser.add_argument("--Roi_size", default=[128, 128, 128], type=int, help="大小")
    args = parser.parse_args()
    predict_one_stage(args=args)


if __name__ == '__main__':
    main()


