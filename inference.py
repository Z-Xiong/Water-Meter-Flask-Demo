import os
import cv2
from PIL import Image
from commons import get_detection_model, get_recognition_model, transform_image, transform_roi, middle_char, feature_to_y

detection_model = get_detection_model()
recognition_model = get_recognition_model()

basedir = os.path.abspath(os.path.dirname(__file__))


def get_detection(image_path, filename):
    try:
        tensor = transform_image(image_path)
        outputs = detection_model.forward(tensor)
    except Exception:
        return 0, 'error'
    image = tensor.squeeze(0).mul(255).permute(1, 2, 0).byte().numpy()
    x1, y1, x2, y2 = int(outputs[0]['boxes'][0, 0]), int(outputs[0]['boxes'][0, 1]), int(
        outputs[0]['boxes'][0, 2]), int(outputs[0]['boxes'][0, 3])

    roi = image[y1:y2, x1:x2, :]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    Image.fromarray(image).save('static/result/'+filename)  # 用PIL保存檢測后標註出的图像
    Image.fromarray(roi).save('static/roi/'+filename)  # 保存感興趣區域
    return Image.fromarray(roi)


def get_recognition(roi):
    try:
        tensor = transform_roi(roi)
        outputs = recognition_model.forward(tensor)
    except Exception:
        return 0, 'error'

    y = feature_to_y(outputs)
    water_number = middle_char(y)
    # print(water_number)

    water_reading = 0
    length = len(water_number)
    for l in range(length):
        water_reading = water_reading + pow(10, length-l-1)*water_number[l]
    print(water_reading)
    return water_reading


# test recongnizition
img = Image.open('./static/roi/1.jpg')
water_reading = get_recognition(img)
# test
