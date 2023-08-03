import os
import glob
import argparse
import torch
from utils.train_one_epoch import train_one_epoch
from utils.evaluate_one_epoch import evaluate_one_epoch
from network.get_model import get_model
from monai.data import CacheDataset, DataLoader, Dataset
from utils.Metric import metircs
from utils.Loss_function import Loss_functions
from utils.train_val_transforms import transforms
from monai.optimizers import Novograd
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings

warnings.filterwarnings('ignore')   # 禁止在控制台打印warnning信息

def train(args):

    # 定义装置为GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据集路径
    image_paths = sorted(glob.glob(os.path.join(args.data_path, "image", "*.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(args.data_path, "label", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(image_paths, label_paths)
    ]

    # 划分训练集和验证集 3:1 = 78:26，总共131个样本,有27个样本没有用到，因为内存不够大
    train_data_dicts = data_dicts[:4*int(len(image_paths)/5)]
    val_data_dicts = data_dicts[int(4*len(image_paths)/5):]

    # train和val transform
    train_transforms, val_transforms = transforms(args.win_level, args.Roi_size, args.stage_ways)

    # # Dataset && DataLoader
    train_dataset = CacheDataset(train_data_dicts, train_transforms)
    train_dataloader = DataLoader(train_dataset,
                                  num_workers=4,
                                  batch_size=args.Batchsize,
                                  shuffle=True)     # shufflle:是否打乱

    val_dataset = Dataset(val_data_dicts, val_transforms)
    val_dataloader = DataLoader(val_dataset,
                                num_workers=4,
                                batch_size=args.Batchsize)

    # 模型
    model = get_model(num_classes=args.num_classes, input_size=args.Roi_size, win_level=args.win_level).to(device)


    # 优化器和学习率调度器
    optimizer = Novograd(model.parameters(), lr=args.lr)
    # 线性预热（warmup_epochs轮数内学习率线性增加，预热起始学习率设置为初始学习率的一半），余弦退火
    # scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=5,
    #                               max_epochs=args.epochs, warmup_start_lr=args.lr * 0.5)

    # 余弦退火策略，学习率分别在epoch为5,20,50时达到最大值args.lr
    scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=5, T_mult=3)

    # 损失函数loss
    loss = Loss_functions(num_classes=args.num_classes)

    # 评价指标
    metric = metircs(num_classes=args.num_classes)


    best_dice = float('-inf')
    best_loss = 1.0

    for epoch in range(args.epochs):

        # train
        epoch_mean_loss = train_one_epoch(model=model,
                                          optimizer=optimizer,
                                          data_loader=train_dataloader,
                                          device=device,
                                          epoch=epoch,
                                          loss_function=loss,
                                          scheduler=scheduler)

        # val
        if (epoch+1) % args.val_epoch == 0:

            epoch_mean_dice = evaluate_one_epoch(model=model,
                                                data_loader=val_dataloader,
                                                device=device,
                                                epoch=epoch,
                                                num_classes = args.num_classes,
                                                dice_metric=metric,
                                                roi_size=args.Roi_size,
                                                sw_batch_size=args.sw_batch_size)
            
            if epoch_mean_dice > best_dice:

                best_dice = epoch_mean_dice


        if epoch_mean_loss < best_loss:

            torch.save(model.state_dict(), args.save_path)
            best_loss = epoch_mean_loss

    print("Over Train ! Minimum loss={}, Maximum dice={}".format(best_loss, best_dice))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/foreground", type=str, help="训练数据集-两阶段训练用resized和foreground")
    parser.add_argument("--save_path", default="./weights/two_stage/best_model_fine_1.pth", type=str, help="模型参数保存路径")
    parser.add_argument("--win_level", default=[-300, 300], type=int, help="调窗范围")
    parser.add_argument("--Roi_size", default=[128,128,128], type=int, help="Roi_size=Resized_size=Input_size")
    parser.add_argument("--num_classes", default=1, type=int, help="分类数")
    parser.add_argument("--stage_ways", default="Crop", type=str, help="选择处理的策略-两阶段训练用none和Crop")
    parser.add_argument("--Batchsize", default=1, type=int, help="批次大小")
    parser.add_argument("--lr", default=0.004, type=int, help="学习率为0.004")
    parser.add_argument("--epochs", default=100, type=int, help="训练轮次")
    parser.add_argument("--val_epoch", default=10, type=int, help="验证间隔轮次")
    parser.add_argument("--sw_batch_size", default=4, type=int, help="滑动推理片数")
    args = parser.parse_args()
    train(args=args)
    

if __name__ == '__main__':
    main()