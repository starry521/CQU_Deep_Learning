from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss

def Loss_functions(num_classes):

    # 二分类
    if num_classes==1:
        loss_function = DiceLoss(to_onehot_y=False, sigmoid=True, include_background=False) # 无需转换为独热码，使用sigmoid

    # 多分类
    else:
        loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=True) # 转换为独热码，使用softmax

    return  loss_function
