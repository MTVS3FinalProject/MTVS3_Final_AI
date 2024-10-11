import logging
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, HTTPException, UploadFile, File
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from io import BytesIO

app = FastAPI()

# 로그 설정
logging.basicConfig(
    filename="app_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# MediaPipe Face Detection 모델 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

TARGET_SIZE = (224, 224)

# 얼굴 검출 및 얼굴 영역 추출 함수 (MediaPipe 사용)
def detect_face(image):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = face_detection.process(img)

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)
            face_img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            return face_img
        else:
            return None

# 얼굴 임베딩 추출 함수 (모델명 반환)
def get_face_embedding(image, img_name):
    model_name = "Facenet"  # 모델 이름 지정
    try:
        face_img = detect_face(image)
        if face_img is None:
            logging.error(f"얼굴을 찾을 수 없습니다 ({img_name}).")
            return None, model_name

        img_resized = cv2.resize(face_img, TARGET_SIZE)

        # 얼굴 임베딩 추출
        embedding = DeepFace.represent(
            img_path=img_resized,
            model_name=model_name,  # 모델 이름 사용
            enforce_detection=False,
            align=True
        )

        return embedding[0]['embedding'], model_name  # 임베딩과 모델명 반환
    except Exception as e:
        logging.error(f"임베딩 추출 실패 ({img_name}): {str(e)}")
        return None, model_name

# 코사인 유사도 계산 함수
def calculate_similarity(embedding1, embedding2):
    if embedding1 is not None and embedding2 is not None:
        similarity = cosine_similarity([embedding1], [embedding2])
        return similarity[0][0]
    else:
        logging.error("유사도 계산 실패")
        return None

@app.post("/verification")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        # 이미지 로드
        img1 = Image.open(BytesIO(await file1.read()))
        img2 = Image.open(BytesIO(await file2.read()))

        # 임베딩 추출 및 모델명 반환
        embedding1, model_name1 = get_face_embedding(img1, "originImg")
        embedding2, model_name2 = get_face_embedding(img2, "currentImg")

        if embedding1 is None or embedding2 is None:
            logging.error("얼굴을 찾을 수 없습니다.")
            raise HTTPException(status_code=400, detail="얼굴을 찾을 수 없습니다.")

        # 유사도 계산
        similarity_score = calculate_similarity(embedding1, embedding2)

        if similarity_score is not None:
            # 소수점 두 자리까지만 표시
            similarity_score_rounded = round(similarity_score * 100, 3)

            match_result = 1 if similarity_score_rounded >= 80 else 0

            # 유사도, 결과, 모델명을 한 줄에 로그로 출력 (응답에는 포함하지 않음)
            logging.info(f"유사도: {similarity_score_rounded}, 결과: {'일치' if match_result == 1 else '불일치'}, 모델: {model_name1}")

            # 응답에는 유사도와 결과만 반환
            return {
                "similarity_score": similarity_score_rounded,
                "match_result": match_result
            }
        else:
            logging.error("유사도 계산 불가.")
            raise HTTPException(status_code=400, detail="유사도 계산 불가.")
    except Exception as e:
        logging.error(f"서버 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn main_byte:app --reload --port 8090
# ngrok http 8090 --domain adapted-charmed-panda.ngrok-free.app
