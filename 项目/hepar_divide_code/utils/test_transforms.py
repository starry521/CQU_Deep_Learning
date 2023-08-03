from monai.transforms import  Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, EnsureTyped


"""LoadImaged + AddChanneld + ScaleIntensityRanged + EnsureTyped"""
def transforms(win_level):
    Te_Lists = []
    Te_Lists.append(LoadImaged(keys=["image"]))
    Te_Lists.append(EnsureChannelFirstd(keys=["image"]))
    Te_Lists.append(ScaleIntensityRanged(
        keys=["image"], a_min=win_level[0], a_max=win_level[1],
        b_min=0.0, b_max=1.0, clip=True     # clip表示裁剪超出范围的值
    ))
    Te_Lists.append(EnsureTyped(keys=["image"]))
    test_transform = Compose(Te_Lists)

    return test_transform