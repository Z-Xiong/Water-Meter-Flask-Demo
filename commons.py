import io
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms


def get_detection_model():
    model = torchvision.models.detection.__dict__['maskrcnn_resnet50_fpn'](num_classes=91,
                                                              pretrained=True)
    model.to('cpu')
    checkpoint = torch.load('./models/water_meter_detection.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def get_recognition_model():
    from model import model
    model = model.WaterMeterModel()
    checkpoint = torch.load('./models/water_meter_recognition.pth', map_location='cpu')
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    model.eval()
    return model


def transform_image(image_path):
    my_transforms = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path)
    return my_transforms(image).unsqueeze(0)


def transform_roi(roi):
    my_transforms = transforms.Compose([
        transforms.Resize((48, 160)),
        transforms.ToTensor()
    ])
    return my_transforms(roi).unsqueeze(0)


import torch
import torch.nn.functional as F

C = {0: 0,
     1: 1,
     2: 2,
     3: 3,
     4: 4,
     5: 5,
     6: 6,
     7: 7,
     8: 8,
     9: 9,
     10: 10,
     11: 11,
     12: 12,
     13: 13,
     14: 14,
     15: 15,
     16: 16,
     17: 17,
     18: 18,
     19: 19,
     20: 'b'}


# 定义函数delet_char, 去重序列，并删除空白字符b
def get_preTarget(a):
    for i in range(len(a) - 1, 0, -1):
        if a[i] == a[i - 1]:
            del a[i]
    while 'b' in a:
        a.remove('b')
    return a


# 将得到的预测序列转换为实际读数
def middle_char(y):
    for i in range(len(y)):
        if y[i] > 9 and y[i] != 'b':
            if i == len(y):
                y[i] = y[i] - 9.5
            else:
                y[i] = y[i] - 10
    return y


def feature_to_y(x):
    x = F.softmax(x, dim=1)  # 把特征序列转换为概率
    x, index = torch.max(x, dim=1)  # 选出概率最大的那个
    index = torch.squeeze(index)  # 压缩index维度
    if torch.cuda.is_available():
        index = index.cpu()
    index = index.numpy().tolist()  # 把index转换为list，方便遍历

    a = []
    for i in index:
        a.append(C[i])  # 把概率转换为字符序列

    y = get_preTarget(a)  # 字符序列去重，保留原有顺序,并删除空白字符b,这时得到的结果与target一致

    return y
