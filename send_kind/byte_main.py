from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

# 얼굴 임베딩 추출 함수
def get_face_embedding(image):
    try:
        # OpenCV 형식으로 변환
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 얼굴 임베딩 추출
        embedding = DeepFace.represent(img_path=img, model_name="VGG-Face", enforce_detection=False)
        return embedding[0]['embedding']  # 임베딩 반환
    except Exception as e:
        return None

# 코사인 유사도 계산 함수
def calculate_similarity(embedding1, embedding2):
    if embedding1 is not None and embedding2 is not None:
        similarity = cosine_similarity([embedding1], [embedding2])
        return similarity[0][0]
    else:
        return None

# 이미지 두 장을 받는 엔드포인트
@app.post("/verification")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        # 이미지 파일을 로드
        img1 = Image.open(BytesIO(await file1.read()))
        img2 = Image.open(BytesIO(await file2.read()))

        # 얼굴 임베딩 추출
        embedding1 = get_face_embedding(img1)
        embedding2 = get_face_embedding(img2)

        if embedding1 is None or embedding2 is None:
            return JSONResponse(content={"error": "얼굴을 찾을 수 없습니다."}, status_code=400)

        # 유사도 계산
        similarity_score = calculate_similarity(embedding1, embedding2)

        if similarity_score is not None:
            # 유사도가 80% 이상이면 0, 그렇지 않으면 1 반환
            match_result = 0 if similarity_score >= 0.8 else 1

            # 유사도와 일치 여부 반환
            return {
                "similarity_score": similarity_score * 100,  # 퍼센트로 변환
                "match_result": match_result
            }
        else:
            return JSONResponse(content={"error": "유사도 계산 불가."}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# uvicorn byte_main:app --reload --port 8090
# ngrok http 8090 --domain adapted-charmed-panda.ngrok-free.app
