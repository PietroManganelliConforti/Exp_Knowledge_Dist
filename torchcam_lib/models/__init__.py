from .resnet import resnet38, resnet110, resnet116, resnet14x2, resnet38x2, resnet110x2, resnet56, resnet18, resnet32, resnet44
from .resnet import resnet8x4, resnet14x4, resnet32x4, resnet38x4, resnet8
from .vgg import vgg8_bn, vgg13_bn, vgg19_bn, repvgg_a2
from .mobilenetv2 import mobile_half, mobile_half_double
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2, ShuffleV2_1_5

from .resnet_imagenet import resnet18, resnet34, resnet50, wide_resnet50_2, resnext50_32x4d
from .resnet_imagenet import wide_resnet10_2, wide_resnet18_2, wide_resnet34_2, wide_resnet101_2
from .mobilenetv2_imagenet import mobilenet_v2
from .shuffleNetv2_imagenet import shufflenet_v2_x1_0

model_dict = {
    # CIFAR100
    'resnet38': resnet38,
    'resnet110': resnet110,
    'resnet116': resnet116,
    'resnet14x2': resnet14x2,
    'resnet38x2': resnet38x2,
    'resnet110x2': resnet110x2,
    'resnet8x4': resnet8x4,
    'resnet14x4': resnet14x4,
    'resnet32x4': resnet32x4,
    'resnet38x4': resnet38x4,
    'vgg8': vgg8_bn,
    'vgg13': vgg13_bn,
    'MobileNetV2': mobile_half,
    'MobileNetV2_1_0': mobile_half_double,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'ShuffleV2_1_5': ShuffleV2_1_5,
    
    'resnet8' : resnet8, # Added by US
    'resnet56': resnet56, # Added by US
    'resnet44': resnet44, # Added by US
    'resnet32': resnet32, # Added by US
    'resnet18': resnet18, # Added by US
    'vgg19bn': vgg19_bn, # Added by US
    'repvgg_a2': repvgg_a2, # Added by US
    
    #Imagenet    
    'ResNet18': resnet18,
    'ResNet34': resnet34,
    'ResNet50': resnet50,
    'resnext50_32x4d': resnext50_32x4d,
    'wide_resnet10_2': wide_resnet10_2,
    'wide_resnet18_2': wide_resnet18_2,
    'wide_resnet34_2': wide_resnet34_2,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2,
    
    'mobilenet_v2': mobilenet_v2,
    'shufflenet_v2_x1_0': shufflenet_v2_x1_0,
}
