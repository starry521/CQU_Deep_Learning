from monai.metrics import DiceMetric

def metircs(num_classes):
    #二分类
    if num_classes==1:
        # include_background:是否跳过第一个通道计算，二分类就一个输出通道，不能跳过，而多分类可以跳过
        # reduction设置计算多个图像的Dice系数之间的平均值，每个批次有多个滑动窗口
        # get_not_nans表示返回包含NaN值的Dice系数，如果有NaN值
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    # 多分类
    else:
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    return  dice_metric