<img src="https://user-images.githubusercontent.com/67897827/221734971-004f61b3-3c9a-4cc2-84b1-87b78fc8f656.PNG" width="600" height="500"/>

# Yolo V3 keras 기반 모델을 활용한 객체 탐지 직접 구현
    - 이미지를 입력해서 객체 탐지
    - 카메라를 통한 실시간 영상을 통해서 객체 탐지
        - 프로그램 수행으로 구동
    - 웹상에 카메라를 통한 실시간 영상을 통해서 객체 탐지
        - 웹브라우저에서 구동

# 설치
    - opencv 가 설치되어 있어야함
    - pip install opencv-python

# yolo 모델 다운로드
    - 가중치 파일  (yolov3.weights)
        - https://pjreddie.com/media/files/yolov3.weights
    - 모델구성정보 (yolov3.cfg)
        - https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
    - 초기모델의 학습시 사용한  class 값 ()
        - https://github.com/pjreddie/darknet/blob/master/data/coco.names
        - coco_labels.txt 저장

# 모델 생성에 입력으로 들어간 yolov3.weights는 파일관계상 깃에는 올리지 못했음
