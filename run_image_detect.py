'''
    yolo v3를 이용한 제로샷 러닝방식으로 객체 탐지 구현
    목표는 yolo를 opencv와 연계하여 사용하는 방법 익히는것
        - 정지 이미지, 카메라를 통한 실시간 영상
            - 차후 필요하면 mediapipe 연계 가능
        - 탐지 대상은 coco_labels.txt에 있는 대상한 한정 => 80개
            - 차후, 커스텀 데이터를 학습하면 이를 추가로 탐지할수 있다

    예측 정보 해석 => tuple 형식으로 나오고, 개수가 탐지한 객체의 수
    (
     탐지한 객체 정보 1, 
     탐지한 객체 정보 2, 
     ...
    )
    탐지한 객체 정보 1 => ndarray
    ( 탐지한정보개수(박스개수), 피처(좌표4개, 더미1개, 학습시분류한레이블총개수(80개)) )
    ex) ( 147(탐지하기 위해 박스가 그려진총개수), 85(4 + 1 + 80) )

    output[0] => (좌표0,좌표1,좌표2,좌표3,더미,5번값~(총80개의 정답에 대한 신뢰값:confidence)>0.5)
    0번값*이미지너비 => 바운딩박스 중심점 X
    1번값*이미지높이 => 바운딩박스 중심점 Y
    2번값*이미지너비 => 바운딩박스 너비
    3번값*이미지높이 => 바운딩박스 높이
    (중심점 X- 너비)/2 = x(왼쪽상단좌표)
    (중심점 Y- 너비)/2 = y(왼쪽상단좌표)
    5:~ : 이중에 가장높은값이 분류값 -> 0.5 이상이면 이미지에 그려라

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
    # 예측 수행시 사용되는 레이어 이름의  인덱스 반환 => [200 227 254] 
    # => 이 번호 해당되는 레이어 이름 추출해서 리스트에 담는다, 
    # 단, model.getUnconnectedOutLayers() -> 1부터 카운트함
    # yolo v3 구조에서 출력을 담당하는 3개의 layer가 존재 => 이를 찾아서 예측시 사용
    out_layers = [ layers_names[index-1] for index in model.getUnconnectedOutLayers() ]
    # 출력 담당 3개의 층 : ['yolo_82', 'yolo_94', 'yolo_106']
    print( out_layers )

    # 3. 정답표 획득 -> 80개 -> yolo v3에서는 80개의 객체를 탐지할수 있다
    with open('coco_labels.txt') as f:
        labels = [label.strip() for label in f.readlines()]
    
    # yolov3모델, 예측시사용되는레이어이름, 분류표(클레스명, 순서대로)
    return model, out_layers, labels

def predict( img,  model, out_layers):
    # 모델에 데이터 주입
    model.setInput( img )
    # 모델에 출력층 정보를 세팅해서 예측 수행
    return model.forward(out_layers)

def parse_predict( outputs, labels, img_h, img_w ): # 예측 결과를 파싱해서 바인딩박스좌표, 신뢰도, 분류번호(값)
    boxs, confs, label_ids = list(), list(), list()

    for object in outputs:  # 탐지된 객체별로 반복 -> 3회(여기서는) 반복
        for info in object: # 객체별 바운딩 박스별로 반복 -> 객체별로 박스수가 다름
            # 여기서 최적의 신뢰도를 가진 박스 정보, 신뢰도값, 레이블번호(분류아이디) 추출
            # 85(4 + 1 + 80)
            # 1. info[5:] 이 구간에서 최대값을 가진 인덱스 추출 -> 인덱스를 넣어서-> 신뢰도 획득
            #    -> 0.5(설정한 신뢰도 임계값) 보다 클결우 -> 바운딩박스 좌표 계산
            confi_condidates = info[5:]                    # 80개의 분류 예측 정보만 추출
            id               = np.argmax(confi_condidates) # 최대값을 가진 인덱스 추출
            max_confidence   = confi_condidates[id]        # id를 이용하여 최대 신뢰도값 추출
            if max_confidence > 0.5:
                # 바운딩 박스 좌표 계산
                # 해당 정보를 모두 boxs, confs, label_ids에 담는다
                #print( id, labels[id], max_confidence )
                '''
                    출력 결과를 모니터링 한결과 객체별로 후보군이 뽑혔다 => 차후 선별과정 필요
                    1 bicycle 0.9915968
                    7 truck 0.5699423 
                    7 truck 0.50926095
                    7 truck 0.5933573 
                    16 dog 0.95524824 
                    16 dog 0.9613244
                    16 dog 0.98602635
                    16 dog 0.99229467
                    16 dog 0.5655263
                '''
                confs.append( max_confidence ) # 신뢰도 담기
                label_ids.append( id )         # 분류 번호 담기
                # 박스 좌표 계산후 담기
                '''
                    0번값*이미지너비 => 바운딩박스 중심점 X
                    1번값*이미지높이 => 바운딩박스 중심점 Y
                    2번값*이미지너비 => 바운딩박스 너비
                    3번값*이미지높이 => 바운딩박스 높이
                    (중심점 X- 너비)/2 = x(왼쪽상단좌표)
                    (중심점 Y- 너비)/2 = y(왼쪽상단좌표)
                '''
                c_x, c_y = int(info[0] * img_w), int(info[1] * img_h)
                w, h     = int(info[2] * img_w), int(info[3] * img_h)
                x, y     = int(c_x-w/2),int(c_y-h/2),
                boxs.append( [x, y, x+w, y+h ])
                pass
            pass
        pass
    print( boxs[0], confs[0], label_ids[0])
    # 이 후보들 중에서 가장 최대값을 추출해서 최종 박스 정보만 추출
    # 노이즈 제거, 비최대억제(NMS) 알고리즘을 적용하여 바운딩 박스중 최대값만 선택하게 계산
    # (예시, 엣지 디텍팅)
    # (박스좌표후보들정보, 신뢰도정보, 신뢰도임계값, NMS임계값)
    indexs = cv.dnn.NMSBoxes(boxs, confs, 0.5, 0.4)
    # indexs => [7 0 3] : 7번 정보, 0번 정보, 3번 정보가 최종 선택
    print( indexs )
    # [ [ x, y, w, h, 신뢰도, 분류아이디], ...  ]
    final_infos = [ boxs[i]+[confs[i]]+[label_ids[i]] for i in range( len(confs) ) if i in indexs ]
    print( final_infos )
    
    # 최종 정보 리턴
    return final_infos
    pass

def detect_img( model, out_layers, labels ):   # 1장의 이미지 데이터로부터 객체 탐지
    # 1. 객체 탐지에 대상이 되는 이미지 로드
    img_src         = cv.imread('dog.jpg') # 이미지 경로 => 이미지가 로드됨 => BGR 형식
    if img_src is None:                    # 이미지가 메모리로 로드가 않되면 종료 -> 나중에는 메시지 처리
        sys.exit('이미지 파일 누락 진행 불가')
    img_h, img_w, _ = img_src.shape        # 이미지의 높이, 너비, 채널(사용않함)
    print( img_h, img_w )   

    # 2. 이미지상에서 정보를 추출하여, yolo v3 모델에 입력이 가능하도록 변환 처리
    # 블롭 (blob) 형태 변환(옵션 의미는 지금은 생략)
    # yolov3 규격에 맞게 수정 (224, 224) or (448, 448)
    # swapRB=True => BGR -> RGB로 변형
    # 사전에 정사각형 이미지라면 정보손실이 없을듯!!
    predict_img = cv.dnn.blobFromImage(img_src, 1/256, (224, 224), (0,0,0), swapRB=True)

    # 4. 예측 수행
    outputs = predict(predict_img, model, out_layers)  

    # 5. 예측 결과 정보 추출 -> 바운딩박스 정보(좌표), 정답번호, 신뢰도값(이 값을 기준으로 걸러낸다)
    final_infos = parse_predict( outputs, labels, img_h, img_w )

    # 박스를 그릴때 객체별로 서로 다른 색상을 부여한다 => 클레스의 숫자만큼 랜덤하게 색상 준비
    # 같은 객체는 같은색상으로 박스 드로잉, 다른 객체간에는 다른색상으로 구분
    # 0 ~ 255 로 (80, 3채널) 랜덤하게 칼라 준비
    colors = np.random.uniform( 0, 255, size=(len(labels), 3))

    # 6. 화면 처리 -> 박스 드로잉, 분류이름 드로잉, 정확도(신뢰도) 드로잉    
    for info in final_infos: # 최종 선별된 박스 정보수 만큰 드로잉
        # info에서 분해하여 6개의 변수에 각각 값을 담는다
        x1, y1, x2, y2, confidence, id = info
        # 박스그리기
        # colors[id] => id와 일치하는 색상을 사용 => 객체(dog,..)와 매칭된 색상
        cv.rectangle(img_src, (x1,y1), (x2,y2), colors[id], 2) # 3:두께
        # 신뢰도 및 분류 클레스값(dog,...) 그리기
        text = f'{labels[id]}-{confidence}' # ex) dog-0.94355
        # 2.0 : 스케일, 3:두께
        cv.putText(img_src, text, (x1,y1+20), cv.FONT_HERSHEY_PLAIN, 1.0, colors[id], 2)

    # 7. 이미지 출력
    cv.imshow('Yolo v3를 활용한 객체 탐지', img_src)

    # 8. 쓰레드 대기
    cv.waitKey()
    # 9. 윈도우 종료
    cv.destroyWindow()
    pass

def detect_live():   # 실시간 영상 데이터로부터 객체 탐지
    pass

def main(): # 모든 업무를 절차적으로 진행 메인코드
    # 1.  yolo v3 로드
    model, out_layers, labels = init_yolo_v3()
    # 2.  1장의 이미지 내에서 객체 탐지
    detect_img( model, out_layers, labels )
    pass

if __name__ == '__main__':
    main()