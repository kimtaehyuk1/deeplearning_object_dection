'''
    - 1. 커스텀 데이터를 이용한 파인튜닝 학습후 모델 적용 (추후 노트 제공, v3, v5 학습예시)
    - 2. 웹 브라우저 상에서 동일하게 적용되도록 확장(javascript) 
    - 3. Mobile(안드로이드/IOS 앱-native or hybrid(+웹앱포함)), IOT(라즈베리파이포팅)
    - 4. 스마트팩토리 -> 장비(고해상 카메라 사용), 컨베이어벨트 -> 불량 제품 감지(GAN 활용)
    - 5. 카메라 환경을 그대로 사용 -> 모션인식, 안면인식, 분야를 확장 -> CCTV 확장 + 컨셉
'''
import numpy as np
import cv2 as cv
import sys

USE_CAMERA_LIVE = True # 라이브 모드

def init_yolo_v3():
    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layers_names = model.getLayerNames()
    out_layers = [layers_names[index-1]
                  for index in model.getUnconnectedOutLayers()]
    with open('coco_labels.txt') as f:
        labels = [label.strip() for label in f.readlines()]
    return model, out_layers, labels

def predict(img,  model, out_layers):
    model.setInput(img)
    return model.forward(out_layers)

def parse_predict(outputs, labels, img_h, img_w):
    boxs, confs, label_ids = list(), list(), list()
    for object in outputs:
        for info in object:
            confi_condidates = info[5:]
            id = np.argmax(confi_condidates)
            max_confidence = confi_condidates[id]
            if max_confidence > 0.5:
                confs.append(max_confidence)
                label_ids.append(id)
                c_x, c_y = int(info[0] * img_w), int(info[1] * img_h)
                w, h = int(info[2] * img_w), int(info[3] * img_h)
                x, y = int(c_x-w/2), int(c_y-h/2),
                boxs.append([x, y, x+w, y+h])
    indexs = cv.dnn.NMSBoxes(boxs, confs, 0.5, 0.4)
    final_infos = [boxs[i]+[confs[i]]+[label_ids[i]]
                   for i in range(len(confs)) if i in indexs]
    return final_infos

def detect_img(model, img_src, out_layers, labels):
    img_h, img_w, _ = img_src.shape
    predict_img = cv.dnn.blobFromImage(
        img_src, 1/256, (224, 224), (0, 0, 0), swapRB=True)
    outputs = predict(predict_img, model, out_layers)
    final_infos = parse_predict(outputs, labels, img_h, img_w)
    colors = np.random.uniform(0, 255, size=(len(labels), 3))
    for info in final_infos:
        x1, y1, x2, y2, confidence, id = info
        cv.rectangle(img_src, (x1, y1), (x2, y2), colors[id], 2)
        text = f'{labels[id]}-{confidence}'
        cv.putText(img_src, text, (x1, y1+20),
                   cv.FONT_HERSHEY_PLAIN, 1.0, colors[id], 2)

    cv.imshow('Yolo v3를 활용한 객체 탐지', img_src)
    cv.waitKey()
    cv.destroyWindow()

def detect_live():   # 실시간 영상 데이터로부터 객체 탐지
    pass

def main():
    model, out_layers, labels = init_yolo_v3()

    img_src = None
    if USE_CAMERA_LIVE:
        capture = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not capture.isOpened():
            sys.exit('카메라 연결 오류')
        else:
            while True:
                res, frame = capture.read()
                if not res:
                    sys.exit('프레임 정보 획득 오류, 여기서는 종료처리')

                # yolo 예측 추가
                detect_img(model, frame, out_layers, labels)

                cv.imshow('camera live image', frame)

                # 특정 키를 입력하면 종료
                key = cv.waitKey(1)
                if key == ord('z'):break
            
            # 카메라와 연결된 자원 해제
            capture.release()
            cv.destroyAllWindows()
        pass
    else:
        img_src = cv.imread('dog.jpg')    
        if img_src is None:
            sys.exit('이미지 파일 누락 진행 불가')
    
        detect_img(model, img_src, out_layers, labels)

if __name__ == '__main__':
    main()