from monai.transforms import  Compose, AsDiscrete, EnsureType, Activations

def post_ways(num_classes):
    
    # 标签数据后处理
    post_label = Compose([EnsureType()])

    # 预测数据后处理
    if num_classes == 1:
        # 二分类
        post_pred = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    else:
        # 多分类
        post_pred = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(argmax=True)])

    return post_label, post_pred