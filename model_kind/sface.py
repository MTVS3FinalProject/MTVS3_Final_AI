from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

app = FastAPI()

# 얼굴 임베딩 추출 함수 (SFace 사용)
def get_face_embedding(image):
    try:
        # OpenCV 형식으로 변환
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 얼굴 임베딩 추출 (SFace 모델 사용)
        embedding = DeepFace.represent(img_path=img, model_name="SFace", enforce_detection=False)
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

# 데이터 전송 형식 정의 (Pydantic)
class CalculateFaceRequestDTO(BaseModel):
    originImg: str  # 원본 이미지 URL
    currentImg: str  # 비교할 이미지 URL

# 이미지 URL 두 개를 받는 엔드포인트
@app.post("/verification")
async def verify_faces(request: CalculateFaceRequestDTO):
    try:
        # URL에서 이미지 다운로드
        response1 = requests.get(request.originImg)
        response2 = requests.get(request.currentImg)

        if response1.status_code != 200 or response2.status_code != 200:
            raise HTTPException(status_code=400, detail="이미지를 다운로드할 수 없습니다.")

        # 이미지를 로드
        img1 = Image.open(BytesIO(response1.content))
        img2 = Image.open(BytesIO(response2.content))

        # 얼굴 임베딩 추출
        embedding1 = get_face_embedding(img1)
        embedding2 = get_face_embedding(img2)

        if embedding1 is None or embedding2 is None:
            raise HTTPException(status_code=400, detail="얼굴을 찾을 수 없습니다.")

        # 유사도 계산
        similarity_score = calculate_similarity(embedding1, embedding2)

        if similarity_score is not None:
            # 유사도가 80% 이상이면 1, 그렇지 않으면 0 반환
            match_result = 1 if similarity_score >= 0.8 else 0

            # 유사도와 일치 여부 반환
            return {
                "similarity_score": similarity_score * 100,  # 퍼센트로 변환
                "match_result": match_result
            }
        else:
            raise HTTPException(status_code=400, detail="유사도 계산 불가.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn MTVS3_Final_AI.model_kind.sface:app --reload --port 8090
# ngrok http 8090 --domain adapted-charmed-panda.ngrok-free.app
