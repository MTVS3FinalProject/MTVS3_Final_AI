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

TARGET_SIZE = (224, 224)

# CLAHE 적용 함수
def apply_clahe(image):
    # 이미지를 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # CLAHE 적용 (클립 제한 값은 2.0, 타일 그리드 크기는 8x8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(gray_image)

    # CLAHE 결과를 BGR 형식으로 변환
    return cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

# 얼굴 임베딩 추출 함수 (FaceNet 사용 + 얼굴 정렬 + 조명 보정 및 크기 고정)
def get_face_embedding(image, img_name):
    try:
        # OpenCV 형식으로 변환
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # 이미지 크기를 224x224로 조정
        img_resized = cv2.resize(img, TARGET_SIZE)

        # CLAHE 적용을 통해 조명 보정
        img_clahe = apply_clahe(img_resized)

        
        embedding = DeepFace.represent(
            img_path=img_clahe, 
            model_name="Facenet", 
            enforce_detection=False,  # 얼굴 감지 실패해도 계속 진행
            align=True  # 얼굴 정렬 활성화
        )

        print(f"임베딩 벡터 ({img_name}): {embedding[0]['embedding']}")
        return embedding[0]['embedding']
    except Exception as e:
        print(f"임베딩 추출 실패 ({img_name}): {str(e)}")
        return None

# 코사인 유사도 계산 함수
def calculate_similarity(embedding1, embedding2):
    if embedding1 is not None and embedding2 is not None:
        print(f"임베딩 벡터 크기 1: {len(embedding1)}, 임베딩 벡터 크기 2: {len(embedding2)}")
        similarity = cosine_similarity([embedding1], [embedding2])
        print(f"유사도 계산 결과: {similarity[0][0]}")
        return similarity[0][0]
    else:
        print("유사도 계산 실패")
        return None

class CalculateFaceRequestDTO(BaseModel):
    originImg: str
    currentImg: str

@app.post("/verification")
async def verify_faces(request: CalculateFaceRequestDTO):
    try:
        
        response1 = requests.get(request.originImg)
        response2 = requests.get(request.currentImg)

        if response1.status_code != 200 or response2.status_code != 200:
            raise HTTPException(status_code=400, detail="이미지를 다운로드할 수 없습니다.")

        
        img1 = Image.open(BytesIO(response1.content))
        img2 = Image.open(BytesIO(response2.content))

        
        embedding1 = get_face_embedding(img1, "originImg")
        embedding2 = get_face_embedding(img2, "currentImg")

        if embedding1 is None or embedding2 is None:
            raise HTTPException(status_code=400, detail="얼굴을 찾을 수 없습니다.")

        
        similarity_score = calculate_similarity(embedding1, embedding2)

        if similarity_score is not None:
            
            match_result = 1 if similarity_score >= 0.8 else 0

            
            return {
                "similarity_score": similarity_score * 100,
                "match_result": match_result
            }
        else:
            raise HTTPException(status_code=400, detail="유사도 계산 불가.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn main:app --reload --port 8090
# ngrok http 8090 --domain adapted-charmed-panda.ngrok-free.app