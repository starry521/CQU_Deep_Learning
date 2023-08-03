import  os
import glob
import SimpleITK as sitk
from monai.transforms import  Compose, LoadImaged, EnsureChannelFirstd, CropForegroundd, Resized
from monai.data import ThreadDataLoader, Dataset
import  argparse
import sys
import warnings

warnings.filterwarnings('ignore')   # 禁止在控制台打印warnning信息


def Trans(args):

    trans_list = []
    trans_list.append(LoadImaged(keys=["image", "label"]))
    trans_list.append(EnsureChannelFirstd(keys=["image", "label"]))
    if args.transform == "resize":
        trans_list.append(Resized(keys=["image", "label"], spatial_size=args.size, mode=["trilinear", "nearest"]))

    elif args.transform =="foreground":
        trans_list.append(CropForegroundd(keys=["image", "label"], source_key="label", margin=30))

    else:
        sys.exit('No choose resize or foreground !')

    train_transforms = Compose(trans_list)

    image_paths = sorted(glob.glob(os.path.join(args.data_path, "image", "*.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(args.data_path, "label", "*.nii.gz")))

    # 获取图像路径
    train_data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(image_paths, label_paths)
    ]

    # 创建dataset和dataloader
    train_dataset = Dataset(train_data_dicts, train_transforms)
    Data_loader = ThreadDataLoader(train_dataset, num_workers=0, batch_size=1, shuffle=False)

    count = 0
    for _, batch in enumerate(Data_loader):

        # 读取图像和标签
        image, label, suffix_img, suffix_label = batch['image'][0][0], batch['label'][0][0], \
                                                 batch["image_meta_dict"]["filename_or_obj"], batch["label_meta_dict"]["filename_or_obj"]  


        suffix = suffix_img[0].split("\\")[-1]                     # 获取图像文件名
        save_image = os.path.join(args.save_path,"image/" + suffix)  # 图像保存路径
        save_label = os.path.join(args.save_path,"label/" + suffix)  # 标签保存路径


        # image
        dicom = sitk.ReadImage(suffix_img[0])
        output = sitk.GetImageFromArray(image.numpy().transpose(2, 1, 0))
        output.SetOrigin(dicom.GetOrigin())
        output.SetSpacing(dicom.GetSpacing())
        output.SetDirection(dicom.GetDirection())
        sitk.WriteImage(output, save_image)

        # label
        dicom = sitk.ReadImage(suffix_label[0])
        label[label==2] = 1
        output = sitk.GetImageFromArray(label.numpy().transpose(2, 1, 0))
        output.SetOrigin(dicom.GetOrigin())
        output.SetSpacing(dicom.GetSpacing())
        output.SetDirection(dicom.GetDirection())
        sitk.WriteImage(output, save_label)

        count += 1
        print("Save: {}/{}".format(count,len(image_paths)))

    print("Over Trans !")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default=[128, 128, 128], type=int,help="only_resized")
    parser.add_argument("--transform", default="foreground", type=str,help="resize/foreground")
    parser.add_argument("--data_path", default="data/origin", type=str, help="读取路径")
    parser.add_argument("--save_path", default="data/foreground", type=str, help="resized/foreground")
    args = parser.parse_args()
    Trans(args=args)


if __name__ == '__main__':
    main()


