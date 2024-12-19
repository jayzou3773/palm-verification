import os
import math
import threading
import cv2
import numpy as np
import torch
import gc
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from models.resnet_model import PalmprintResNet
from models.simplescript import SiameseNetwork
from PIL import Image
from ultralytics import YOLO
import torch.nn.functional as F
import matplotlib.pyplot as plt


app = Flask(__name__)
CORS(app)

# Ensure the uploads directory exists
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# The process of images
temp_images = {"left": [None, None, None], "right": [None, None, None], "val": None}
image_process_status = {"left": [None, None, None], "right": [None, None, None], "val": None}

# YOLO model loading (only once)
MODEL_PATH = "./models/best.onnx"  # 替换为实际路径
YOLO_MODEL = YOLO(MODEL_PATH, task='detect')

"""
The status of data：
1. None: no data
2. processing
3. processed
4. failed
"""

# Yolo detection
def getYOLOOutput(img, CONFIDENCE=0.5):
    resultlist = YOLO_MODEL.predict(source=img, imgsz=512)
    results = resultlist[0]
    print(
        f"[INFO] YOLO took {results.speed['preprocess'] + results.speed['inference'] + results.speed['postprocess']} ms "
    )
    boxes = []
    dfg = []
    pc = []
    for box in results.boxes:
        if box.conf > CONFIDENCE:
            boxes.append(box)
    for box in boxes:
        if box.cls == 0:
            dfg.append([box.xywh[0][0], box.xywh[0][1], box.conf])
        else:
            pc.append([box.xywh[0][0], box.xywh[0][1], box.conf])

    return dfg, pc


def onePoint(x, y, angle):
    X = x * math.cos(angle) + y * math.sin(angle)
    Y = y * math.cos(angle) - x * math.sin(angle)
    return [int(X), int(Y)]


def extractROI(img, dfg, pc):
    (H, W) = img.shape[:2]
    if W > H:
        im = np.zeros((W, W, 3), np.uint8)
        im[...] = 255
        im[1:H, 1:W, :] = img[1:H, 1:W, :]
        edge = W
    else:
        im = np.zeros((H, H, 3), np.uint8)
        im[...] = 255
        im[1:H, 1:W, :] = img[1:H, 1:W, :]
        edge = H

    center = (edge / 2, edge / 2)
    x1, y1 = float(dfg[0][0]), float(dfg[0][1])
    x2, y2 = float(dfg[1][0]), float(dfg[1][1])
    x3, y3 = float(pc[0][0]), float(pc[0][1])

    x0 = (x1 + x2) / 2
    y0 = (y1 + y2) / 2
    unitLen = math.sqrt(np.square(x2 - x1) + np.square(y2 - y1))

    k1 = (y1 - y2) / (x1 - x2)
    b1 = y1 - k1 * x1 

    k2 = (-1) / k1 
    b2 = y3 - k2 * x3

    tmpX = (b2 - b1) / (k1 - k2)
    tmpY = k1 * tmpX + b1

    vec = [x3 - tmpX, y3 - tmpY]
    sidLen = math.sqrt(np.square(vec[0]) + np.square(vec[1]))
    vec = [vec[0] / sidLen, vec[1] / sidLen]
    print(vec)

    if vec[1] < 0 and vec[0] > 0:
        angle = math.pi / 2 - math.acos(vec[0])
    elif vec[1] < 0 and vec[0] < 0:
        angle = math.acos(-vec[0]) - math.pi / 2
    elif vec[1] >= 0 and vec[0] > 0:
        angle = math.acos(vec[0]) - math.pi / 2
    else:
        angle = math.pi / 2 - math.acos(-vec[0])

    x0, y0 = onePoint(x0 - edge / 2, y0 - edge / 2, angle)
    x0 += edge / 2
    y0 += edge / 2

    M = cv2.getRotationMatrix2D(center, angle / math.pi * 180, 1.0)
    tmp = cv2.warpAffine(im, M, (edge, edge))
    ROI = tmp[int(y0 + unitLen / 2):int(y0 + unitLen * 3), int(x0 - unitLen * 5 / 4):int(x0 + unitLen * 5 / 4), :]
    ROI = cv2.resize(ROI, (224, 224), interpolation=cv2.INTER_CUBIC)
    return ROI

# Get the ROI area in image
def get_imageROI(image, image_id, hand_type, CONFIDENCE=0.5):
    dfg, pc = getYOLOOutput(image, CONFIDENCE)
    gc.collect()

    if len(dfg) < 2:
        print('Detect fail. Please re-take photo and input it.')
        image_process_status[hand_type][image_id] = 'failed'
        return False
    else:
        if len(dfg) > 2:
            tmpdfg = []
            maxD = 0
            for i in range(len(dfg) - 1):
                for j in range(i + 1, len(dfg)):
                    d = math.sqrt(pow(dfg[i][0] - dfg[j][0], 2) + pow(dfg[i][1] - dfg[j][1], 2))
                    if d > maxD:
                        tmpdfg = [dfg[i], dfg[j]]
                        maxD = d
            dfg = tmpdfg

        pc = sorted(pc, key=lambda x: x[-1], reverse=True)

    ROI = extractROI(image, dfg, pc)

    # update
    if hand_type == 'val':
        temp_images['val'] = ROI
        image_process_status['val'] = 'processed'
    else:
        temp_images[hand_type][image_id] = ROI
        image_process_status[hand_type][image_id] = 'processed'
    return ROI

def get_valROI(image, CONFIDENCE=0.5):
    dfg, pc = getYOLOOutput(image, CONFIDENCE)
    gc.collect()
    if len(dfg) < 2:
        print('Detect fail. Please re-take photo and input it.')
        return False
    else:
        if len(dfg) > 2:
            tmpdfg = []
            maxD = 0
            for i in range(len(dfg) - 1):
                for j in range(i + 1, len(dfg)):
                    d = math.sqrt(pow(dfg[i][0] - dfg[j][0], 2) + pow(dfg[i][1] - dfg[j][1], 2))
                    if d > maxD:
                        tmpdfg = [dfg[i], dfg[j]]
                        maxD = d
            dfg = tmpdfg

        pc = sorted(pc, key=lambda x: x[-1], reverse=True)

    ROI = extractROI(image, dfg, pc)
    return ROI

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload/<image_id>', methods=['POST'])
def uploadImg(image_id):
    # 如果 image_id 是 'validation'，跳过转换为整数的步骤
    if image_id == 'validation':
        return jsonify({'message': 'Validation image upload received.'}), 200

    # 其他 image_id 的处理
    try:
        image_id = int(image_id)
    except ValueError:
        return jsonify({'message': 'Invalid image ID'}), 400

    left_hand = request.files.get('left_hand')
    right_hand = request.files.get('right_hand')

    if not left_hand or not right_hand:
        return jsonify({'message': 'Both left and right hand images are required'}), 400

    if left_hand.filename == '' or right_hand.filename == '':
        return jsonify({'message': 'One or more selected files are missing'}), 400

    if image_id > 2 or image_id < 0:
        return jsonify({'message': 'The id of hands is not right'}), 400

    image_process_status['left'][image_id] = 'processing'
    image_process_status['right'][image_id] = 'processing'

    # Save images to the uploads folder
    left_path = os.path.join(UPLOAD_FOLDER, f'left_{image_id}.jpg')
    right_path = os.path.join(UPLOAD_FOLDER, f'right_{image_id}.jpg')

    left_hand.save(left_path)
    right_hand.save(right_path)
    
    # # Load images using OpenCV
    # img_left = cv2.imread(left_path)
    # img_right = cv2.imread(right_path)

    # # Start threads for processing images
    # thread_left = threading.Thread(target=get_imageROI, args=(img_left, image_id, 'left'))
    # thread_right = threading.Thread(target=get_imageROI, args=(img_right, image_id, 'right'))

    # thread_left.start()
    # thread_right.start()
    
    return jsonify({'message': 'File uploaded successfully'}), 200



@app.route('/check', methods=['GET'])
def check_process_status():
    if any(status == 'processing' for status in image_process_status['left']) or \
       any(status == 'processing' for status in image_process_status['right']):
        return jsonify({'status': 'processing'}), 200

    if any(status == 'failed' for status in image_process_status['left']) or \
       any(status == 'failed' for status in image_process_status['right']):
        return jsonify({'status': 'failed'}), 200

    return jsonify({'status': 'completed'}), 200


@app.route('/compare', methods=['POST'])
def compare():
    # 获取请求中的图片
    val_image = request.files.get('val_image')
    left_image = request.files.get('left_image')
    right_image = request.files.get('right_image')

    if val_image is None or left_image is None or right_image is None:
        return jsonify({'message': 'Validation, left, and right images are required.'}), 400

    # 将图片保存到本地
    val_image_path = os.path.join(UPLOAD_FOLDER, 'val_image.jpg')
    left_image_path = os.path.join(UPLOAD_FOLDER, 'left_image.jpg')
    right_image_path = os.path.join(UPLOAD_FOLDER, 'right_image.jpg')

    val_image.save(val_image_path)
    left_image.save(left_image_path)
    right_image.save(right_image_path)

    # 加载图像并提取特征
    val_image = cv2.imread(val_image_path)
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)
    
    
    # 提取ROI区域
    val_image = get_valROI(val_image)
    left_ROI = get_imageROI(left_image, 0, 'left')
    right_ROI = get_imageROI(right_image, 0, 'right')

    model = PalmprintResNet()
    model.load_state_dict(torch.load('./models/best_resnet50_model_epoch_0.pth', map_location=torch.device('cpu')))
    
    # model = SiameseNetwork()
    # model.load_state_dict(torch.load('./models/best_siamese.pth', map_location=torch.device('cpu')))

    RGB_MEAN = [0.5, 0.5, 0.5]  # for normalize inputs to [-1, 1]
    RGB_STD = [0.5, 0.5, 0.5]
    test_transform = transforms.Compose([
        transforms.Resize([224, 224], interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ])

    # convert val_image
    val_image = Image.fromarray(val_image)
    val_image = test_transform(val_image)
    val_image = val_image.unsqueeze(0)
    similarities = {'left': [], 'right': []}

    # 计算左手和右手的相似度
    for hand, ROI in [('left', left_ROI), ('right', right_ROI)]:
        ROI = Image.fromarray(ROI)
        ROI = test_transform(ROI)
        ROI = ROI.unsqueeze(0)
        output1, output2 = model(val_image, ROI)

        # 修正: 使用 detach() 分离梯度
        output1_np = output1.detach().cpu().numpy()
        output2_np = output2.detach().cpu().numpy()

        # 计算相似度
        similarity = np.dot(output1_np, output2_np.T) / (
                np.linalg.norm(output1_np) * np.linalg.norm(output2_np))
        similarities[hand].append(similarity)
        print(similarity)

    similarity_left = sum(similarities['left']) / len(similarities['left'])
    similarity_right = sum(similarities['right']) / len(similarities['right'])
    print(similarity_left, similarity_right)
    matched = False
    if similarity_left > 0.9999 or similarity_right > 0.9999:
        matched = True
    if matched:
        return jsonify({'message': '配对成功'}), 200
    else:
        return jsonify({'message': '配对失败'}), 200

    


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
