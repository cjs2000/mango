#pip install ptflops
from ptflops import get_model_complexity_info
from torchvision.models import resnet50
from models.inceptionv3 import *
from models.resnet import *
from models.eca_resnet import *
from models.Vit_transformer import *
from models.mobilenet import *
from models.mobilenetv2 import *
from models.shufflenet import *
from models.shufflenetv2 import *
from models.squeezenet import *
from models.SwinTransformer import *
from models.densenet import *
from models.shufflenet_eca import *
from shufflenet_IM.shufflenet_SE import *
from shufflenet_IM.shufflenet_BAM import *
from shufflenet_IM.shufflenet_EMA import *
from shufflenet_IM.shufflenet_MobileViTAttention import *
from shufflenet_IM.shufflenet_GAM import *
from shufflenet_IM.shufflenet_ECA_GC import *
from models.LeNet5 import *
from models.AlexNet import *
from models.EfficientNet import *
from models.googlenet import *
from shufflenet_IM.shuffleNet_GC import *
from models.xception import *

from ptflops import get_model_complexity_info
model = xception()
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print('Flops:  ' + flops)
print('Params: ' + params)