import torch
import time
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from utils.post_ways import post_ways

def evaluate_one_epoch(model, data_loader, device, epoch,num_classes, dice_metric, roi_size, sw_batch_size):
    
    model.eval()
    total_dice = 0
    number = 0

    # 定义后处理compose
    post_label, post_pred = post_ways(num_classes)

    with torch.no_grad():   #不计算梯度

        for _, batch in enumerate(data_loader):

            # 获取图像和标签
            images, labels = batch['image'].to(device), batch['label'].to(device)

            # forward
            outputs = sliding_window_inference(images, roi_size, sw_batch_size, model)

            # 计算评价指标
            outputs = [post_pred(i) for i in decollate_batch(outputs)]
            labels = [post_label(i) for i in decollate_batch(labels)]
            dice = dice_metric(y_pred=outputs, y=labels)

            # 计算总dice系数和总batch数
            total_dice += dice.item()
            number = number + 1

            # 清空占用空间
            del images, labels
            torch.cuda.empty_cache()

        # 计算平均dice系数
        epoch_mean_dice = total_dice / number

        # 打印此轮验证结果
        print(f"epoch {epoch + 1}, average dice: {epoch_mean_dice:.4f}")
        
        return  epoch_mean_dice