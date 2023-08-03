from monai.transforms import  Compose,Resized,LoadImaged,EnsureChannelFirstd,ScaleIntensityRanged,\
                            EnsureTyped,SpatialPadd,RandCropByPosNegLabeld

def transforms(win_level, roi_size, stage_ways):

    #-----------Train---------------#
    """LoadImaged + AddChanneld + ScaleIntensityRanged + [Crop | Resize] + EnsureTyped"""
    Tr_Lists = []
    
    #加载图片
    Tr_Lists.append(LoadImaged(keys=["image", "label"]))
    Tr_Lists.append(EnsureChannelFirstd(keys=["image", "label"]))
    
    # 强度缩放(0-1之间)
    Tr_Lists.append(ScaleIntensityRanged(keys=["image"], a_min=win_level[0], a_max=win_level[1],
            b_min=0.0, b_max=1.0, clip=True))

    #分块/resized方法  二选一
    if stage_ways == "Crop":

        # 空间填充（使图像与指定图像大小相匹配）
        Tr_Lists.append(SpatialPadd(keys=["image","label"], spatial_size=roi_size))

        # 随机裁剪,数据增强
        Tr_Lists.append(RandCropByPosNegLabeld(keys=["image","label"], pos=1, neg=1, num_samples=4,\
                                               label_key="label", spatial_size=roi_size))

    if stage_ways == "Resize":
        Tr_Lists.append(Resized(keys=["image", "label"], spatial_size=roi_size, mode=["trilinear", "nearest"]))

    Tr_Lists.append(EnsureTyped(keys=["image", "label"]))  # 确保数据是正确的，例如都为张量
    train_transforms = Compose(Tr_Lists)



    # -----------Val---------------#
    """LoadImaged + AddChanneld + ScaleIntensityRanged + [Resized] + EnsureTyped"""
    Val_Lists = []
    Val_Lists.append(LoadImaged(keys=["image", "label"]))
    Val_Lists.append(EnsureChannelFirstd(keys=["image", "label"]))

    # 强度缩放
    Val_Lists.append(ScaleIntensityRanged(keys=["image"], a_min=win_level[0], a_max=win_level[1],
                                      b_min=0.0, b_max=1.0, clip=True))
    
    #只有一阶段方法才用到
    if stage_ways == "Resize":
        Val_Lists.append(Resized(keys=["image", "label"], spatial_size=roi_size, mode=["trilinear", "nearest"]))

    Val_Lists.append(EnsureTyped(keys=["image", "label"]))
    val_transforms = Compose(Val_Lists)


    return train_transforms,val_transforms

