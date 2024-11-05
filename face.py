import logging
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, HTTPException
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from io import BytesIO
import requests

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
mp_face_mesh = mp.solutions.face_mesh

TARGET_SIZE = (224, 224)

# 얼굴 검출 및 얼굴 영역 추출 함수 (MediaPipe 사용 + 그레이스케일 변환 추가)
def detect_face(image):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
        # 이미지를 그레이스케일로 변환
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # DeepFace는 BGR 이미지를 사용하므로 다시 BGR로 변환
        results = face_detection.process(img_bgr)

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = img_bgr.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)
            face_img = img_bgr[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            return face_img
        else:
            return None

def anti_spoofing_check(image):
    try:
        # Laplacian 변수를 계산하여 이미지의 텍스처 분석 (선명도 확인)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

        # 조명 분석: 평균 밝기와 분산을 사용하여 반사광 패턴 분석
        brightness_mean = np.mean(gray_image)
        brightness_std = np.std(gray_image)

        # RGB 채널 간 비율을 확인하여 사진의 품질 및 위조 여부 판단
        mean_r = np.mean(image[:, :, 2])  # Red 채널
        mean_g = np.mean(image[:, :, 1])  # Green 채널
        mean_b = np.mean(image[:, :, 0])  # Blue 채널

        # 일반적으로 자연스러운 얼굴 사진에서 RGB 비율이 일정하게 유지됨
        # 특정 비율이 크게 벗어나면 비정상적인 조명이나 재촬영된 이미지일 가능성이 있음
        if abs(mean_r - mean_g) > 15 or abs(mean_g - mean_b) > 15:
            logging.warning("RGB 채널 비율이 비정상적입니다. 스푸핑이 의심됩니다.")
            return False

        # 조명 분석 및 텍스처 분석 기준
        if laplacian_var < 100 or brightness_std < 20:
            logging.warning("안티 스푸핑 경고: 스푸핑 이미지로 의심됨.")
            return False  # 스푸핑 이미지로 간주
        return True
    except Exception as e:
        logging.error(f"안티 스푸핑 검사 실패: {str(e)}")
        return False

# 얼굴 임베딩 추출 함수 (모델명 반환)
def get_face_embedding(image, img_name):
    model_name = "Facenet"  # 모델 이름 지정
    try:
        face_img = detect_face(image)
        if face_img is None:
            logging.error(f"얼굴을 찾을 수 없습니다 ({img_name}).")
            return None, model_name

        # 안티 스푸핑 검사
        if not anti_spoofing_check(face_img):
            logging.error(f"스푸핑이 의심되는 이미지입니다 ({img_name}).")
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

# 마스크 착용 여부 감지 함수 (FaceMesh 사용)
def mask_detection(image):
    try:
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7) as face_mesh:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 코와 입의 랜드마크 (코: 1, 입: 13, 14, 15 등의 인덱스)
                    nose_tip = face_landmarks.landmark[1]  # 코 끝 부분
                    mouth_upper = face_landmarks.landmark[13]  # 입 윗부분
                    mouth_lower = face_landmarks.landmark[14]  # 입 아랫부분

                    # 코와 입의 랜드마크가 제대로 보이는지 확인
                    if abs(nose_tip.y - mouth_upper.y) < 0.05:  # 코와 입 사이의 거리가 너무 짧다면 마스크가 있는 것
                        logging.warning("마스크가 착용된 것으로 의심됩니다.")
                        return False  # 마스크 착용 의심

                logging.info("마스크 착용 감지되지 않음.")
                return True  # 마스크 미착용 (정상 얼굴)
            else:
                logging.error("FaceMesh 랜드마크를 찾을 수 없습니다.")
                return False
    except Exception as e:
        logging.error(f"마스크 감지 중 오류 발생: {str(e)}")
        return False

# URL에서 이미지를 불러오는 함수
def load_image_from_url(url):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        logging.error(f"이미지 로드 실패: {str(e)}")
        return None

@app.post("/verification")
async def verify_faces(request: dict):
    try:
        # JSON 형식으로 URL 받기
        origin_img_url = request.get("originImg")
        current_img_url = request.get("currentImg")
        
        # URL에서 이미지 로드
        img1 = load_image_from_url(origin_img_url)
        img2 = load_image_from_url(current_img_url)

        if img1 is None or img2 is None:
            logging.error("이미지를 불러오지 못했습니다.")
            raise HTTPException(status_code=400, detail="이미지를 불러오지 못했습니다.")

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

            match_result = 1 if similarity_score_rounded >= 20 else 0

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

# 회원가입 시 제출된 이미지가 사람인지 확인하는 엔드포인트
@app.post("/recognition")
async def verify_image(request: dict):
    try:
        # JSON 형식으로 URL 받기
        face_img_url = request.get("faceImg")
        if not face_img_url:
            raise HTTPException(status_code=400, detail="유효한 이미지 URL이 제공되지 않았습니다.")

        # URL에서 이미지 로드
        img = load_image_from_url(face_img_url)
        if img is None:
            raise HTTPException(status_code=400, detail="이미지를 불러오지 못했습니다.")

        img = np.array(img)

        # 얼굴 검출
        face_img = detect_face(img)
        if face_img is None:
            logging.error("얼굴을 찾을 수 없습니다.")
            return {"message": "얼굴을 찾을 수 없습니다. 사람이 아닐 가능성이 높습니다.", "result": 0}

        # 마스크 착용 여부 감지
        if not mask_detection(face_img):
            logging.error("마스크가 감지되었습니다. 검증 실패.")
            return {"message": "마스크가 감지되었습니다. 얼굴을 확인할 수 없습니다.", "result": 0}

        # 안티 스푸핑 검사
        if not anti_spoofing_check(face_img):
            logging.error("스푸핑이 의심되는 이미지입니다.")
            return {"message": "스푸핑이 의심됩니다. 사진이 사람인지 확인하세요.", "result": 0}

        logging.info("사람 얼굴이 검증되었습니다.")
        return {"message": "사람 얼굴이 확인되었습니다.", "result": 1}

    except Exception as e:
        logging.error(f"서버 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"서버 오류가 발생했습니다: {str(e)}")


# ---------------------------------- 수동 ----------------------------------
# gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 127.0.0.1:8090
# sudo systemctl start nginx
# ngrok http 8080 --domain adapted-charmed-panda.ngrok-free.app
# ---------------------------------- 수동 ----------------------------------

# ---------------------------------- docker ----------------------------------
# docker-compose up --build
# sudo systemctl start nginx
# ngrok http 8080 --domain adapted-charmed-panda.ngrok-free.app
# ---------------------------------- docker ----------------------------------

# ---------------------------------- nginx ----------------------------------
# sudo systemctl start nginx <- 시작
# sudo systemctl stop nginx <- 종료
# sudo systemctl restart nginx <- 재시작
# sudo systemctl status nginx <- nginx 상태 확인
# ps aux | grep nginx <- nginx 프로세스 확인
# ---------------------------------- nginx ----------------------------------

# Nginx + Gunicorn + FastAPI + ngrok