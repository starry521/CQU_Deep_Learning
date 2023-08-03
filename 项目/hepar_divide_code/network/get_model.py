from network.unet import UNet


def get_model(num_classes, input_size, win_level):
    
    model_cfg = {'NUM_CLASSES': num_classes,                # 分类数
                 'NUM_CHANNELS': [16, 32, 64, 128, 256],    # 每层通道数
                # 'NUM_CHANNELS': [32, 64, 128, 256, 512],    # 每层通道数
                 'NUM_DEPTH': 4,                            # 网络深度
                #  'NUM_BLOCKS': [2, 2, 2, 2],                # 编码模块数
                #  'DECODER_NUM_BLOCK': 2,                    # 解码模块数
                #  'AUXILIARY_TASK': False,                   # 是否进行辅助任务
                #  'AUXILIARY_CLASS': 1,                      # 辅助任务分类数   
                 'ENCODER_CONV_BLOCK': 'ResTwoLayerConvBlock',   # 编码器模块
                 'DECODER_CONV_BLOCK': 'ResTwoLayerConvBlock',   # 解码器模块
                 'IS_PREPROCESS': False,                    # 预处理
                 'IS_POSTPROCESS': False,                   # 后处理
                #  'INPUT_SIZE':input_size,                   # 输入图像大小
                #  "WINDOW_LEVEL":win_level,                  # 调窗范围
                 'IS_DYNAMIC_EMPTY_CACHE': True}            # 动态清空GPU缓存

    model = UNet(model_cfg)

    return model