'''
    yolo v3를 이용한 제로샷 러닝방식으로 객체 탐지 구현
    목표는 yolo를 opencv와 연계하여 사용하는 방법 익히는것
        - 정지 이미지, 카메라를 통한 실시간 영상
            - 차후 필요하면 mediapipe 연계 가능
        - 탐지 대상은 coco_names.txt에 있는 대상한 한정
            - 차후, 커스텀 데이터를 학습하면 이를 추가로 탐지할수 있다
'''
import numpy as np
import cv2 as cv
import sys

def init_yolo_v3(): # yolo v3 모델 로드
    '''
        예측 수행에 필요한 모든 리소스 로드
        모델, 예측시 사용하는 레이어명, 정답표
    '''
    # 1. 모델 로드 (cv로부터 로드), 입력(가중치, 모델구조정보)    
    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    # 2. 예측시 사용하는 레이어명 획득
    # 모델의 각층 이름 획득 -> 예측시 사용될 레이어의 이름 획득 (yolo v3 사용방식)
    layers_names = model.getLayerNames()
    # print( layers_names )
    # 예측 수행시 사용되는 레이어 이름의  인덱스 반환 => [200 227 254] , model.getUnconnectedOutLayers()
    # => 이 번호 해당되는 레이어 이름 추출해서 리스트에 담는다, 
    # 단, model.getUnconnectedOutLayers() -> 1부터 카운트함
    out_layers = [ layers_names[index-1] for index in model.getUnconnectedOutLayers() ]
    #print( out_layers )

    # 3. 정답표 획득
    with open('coco_labels.txt') as f:
        labels = [label.strip() for label in f.readlines()]
    
    # yolov3모델, 예측시사용되는레이어이름, 분류표(클레스명, 순서대로)
    return model, out_layers, labels

def detect_img( model, out_layers ):   # 1장의 이미지 데이터로부터 객체 탐지
    # 1. 객체 탐지에 대상이 되는 이미지 로드
    img_src = cv.imread('dog.jpg') # 이미지 경로 => 이미지가 로드됨
    if img_src is None:
        sys.exit('이미지 파일 누락 진행 불가')

    # 2. 이미지상에서 정보를 추출하여, yolo v3 모델에 입력이 가능하도록 변환 처리
    # 블롭 (blob) 형태 변환(옵션 의미는 지금은 생략)
    # (224,224) => (448,448)
    predict_img = cv.dnn.blobFromImage(img_src, 1/256, (224,224), (0,0,0), swapRB=True)
    
    # 3. 모델에 입력
    model.setInput( predict_img )
    
    # 4. 예측 수행
    outputs = model.forward(out_layers)

    # 5. 예측 결과 정보 추출 -> 바운딩박스 정보(좌표), 정답번호, 신뢰도값(이 값을 기준으로 걸러낸다)

    # 6. 화면 처리 -> 박스 드로잉, 분류이름 드로잉, 정확도(신뢰도) 드로잉

    pass

def detect_live():   # 실시간 영상 데이터로부터 객체 탐지
    pass

def main(): # 모든 업무를 절차적으로 진행 메인코드
    # 1.  yolo v3 로드
    model, out_layers, labels = init_yolo_v3()
    # 2.  1장의 이미지 내에서 객체 탐지
    detect_img( model, out_layers )
    pass

if __name__ == '__main__':
    main()