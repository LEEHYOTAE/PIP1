import tensorflow as tf
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 모델 로드
model = tf.keras.models.load_model('keras_model.h5')

# 이미지 전처리 함수
def preprocess_image(img):
    # 이미지를 224x224 크기로 조정하고 픽셀 값을 [0, 1]로 정규화
    img = cv2.resize(img, (224, 224)) / 255.0
    # 이미지를 배치 형태로 변환
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit 애플리케이션
def main():
    # 웹캠 열기
    cap = cv2.VideoCapture(0)

    # 이전 프레임에서 처리된 키 이벤트의 상태를 저장할 변수
    key_event = -1

    while True:
        # 웹캠에서 이미지를 읽어옴
        ret, frame = cap.read()
        if ret:
            # 이미지를 전처리하여 모델에 입력
            img = preprocess_image(frame)
            pred = model.predict(img)

            # 예측 결과 출력
            if np.argmax(pred) == 0:
                st.write("Prediction: Class 1")
            elif np.argmax(pred) == 1:
                st.write("Prediction: Class 2")
            else:
                st.write("Prediction: Class 3")

            # 이미지 출력
            st.image(frame, channels="BGR")

            # 이전 프레임에서 처리된 키 이벤트를 확인하여 'q'가 입력되었으면 종료
            if key_event == ord('q'):
                break

            # waitKeyEx() 함수를 사용하여 초당 한 번의 검증만 수행
            key_event = cv2.waitKeyEx(1)

    # 웹캠 종료
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()