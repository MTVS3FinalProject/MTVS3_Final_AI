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
import json
from urllib import request, parse
import random
import uuid
import websocket
import urllib
import io
import base64
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, Request
from langchain_openai import ChatOpenAI
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일 로드
openai_api_key = os.getenv("OPENAI_API_KEY")


app = FastAPI()

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = openai_api_key

# GPT 기반의 Chat 모델 설정
gpt = ChatOpenAI(model="gpt-4o-mini")

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

BASE_DIR = "./BASE_DIR"  # BASE_DIR을 명시적으로 정의
os.makedirs(BASE_DIR, exist_ok=True)

# 127.0.0.1:8188
# comfyui_service
server_address = "comfyui_service:8188"  # 서비스 이름으로 수정
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images

async def generate_prompt(question: str) -> str:
    """
    이미지 생성용 프롬프트를 생성하는 함수.
    
    Parameters:
        question (str): 사용자로부터 받은 질문

    Returns:
        str: GPT가 생성한 이미지 프롬프트
    """
    prompt_question = f'현재 나는 "{question}" 이러한 이미지 생성을 하려고 하는데 영어로 프롬프트를 짜줘.'
    return gpt(prompt_question)

async def load_image_chat(filename, question):
    # JSON 파일 로드 및 설정
    with open("./final_test.json", "r", encoding="utf-8") as f:
        workflow_data = f.read()
    workflow = json.loads(workflow_data)

    # 랜덤 시드 값 설정
    seed = random.randint(1, 1000000000)
    workflow["3"]["inputs"]["seed"] = seed

    # 프롬프트 생성
    receive = await generate_prompt(question)
    workflow["6"]["inputs"]["text"] = f"{receive}"

    # WebSocket 연결
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
    images = get_images(ws, workflow)

    # 이미지 저장
    file_path = None
    for node_id in images:
        for image_data in images[node_id]:
            img = Image.open(io.BytesIO(image_data))
            file_path = os.path.join(BASE_DIR, f'{filename}.png')
            img.save(file_path, format="PNG")
        
    return file_path

def load_image(filename):
    # 가능한 color와 style 목록
    colors1 = ["blue", "purple", "aqua", "teal"]  # 바다의 깊이와 맑음을 표현하는 색상들
    colors2 = ["purple","orange", "pink", "red"]  # 일몰이나 일출을 표현하는 따뜻한 색상들

    # 바다와 하늘을 표현하는 스타일들
    style = [
        "modern",         # 깔끔하고 세련된 현대적 느낌
        "tropical",       # 열대 바다의 이국적 느낌
        "minimalistic",   # 간결한 디자인으로 바다와 하늘의 분위기를 강조
        "surreal",        # 비현실적인 색감과 형태로 독특한 느낌을 주는 초현실적 스타일
        "abstract",       # 색감과 형태를 강조하는 추상적인 바다 표현
        "vintage",        # 고풍스러운 색감과 질감으로 고요한 느낌의 바다
        "impressionistic" # 명암과 색감이 강한 인상파 느낌의 바다
    ]

    # 프롬프트 템플릿 리스트
    prompt_templates = [
        "{colors1} waves with {colors2} reflections",
        "{colors1} ocean with hints of {colors2} light",
        "{colors1} sea stretching into the horizon, blending with {colors2} hues",
        "{colors1} ocean meeting a {colors2} sky at the horizon",
        "{colors1} water beneath a soft {colors2} sunset",
        "{colors2} dawn sky mirrored on a {colors1} sea",
        "{style} style, capturing the {colors1} tides and {colors2} sky",
        "{style} seascape with a {colors1} ocean against a {colors2} sky",
        "{style} art of the {colors1} sea with a contrasting {colors2} sky",
        "{colors1} waves under a {colors2} twilight sky",
        "{style} interpretation of a {colors2} dawn over {colors1} waters",
        "{colors2} sunset casting a glow on the {colors1} ocean in a {style} design",
        "{style} design showcasing {colors1} waters and a soft {colors2} sky",
        "{colors1} waves rolling beneath a {colors2} sky in a {style} scene",
        "{style} portrayal of {colors1} ocean depths under a {colors2} sky"
    ]

    chosen_template = random.choice(prompt_templates)
    colors1 = random.choice(colors1)
    colors2 = random.choice(colors2)
    style = random.choice(style)

    # 최종 프롬프트 생성
    prompt = chosen_template.format(colors1=colors1, colors2=colors2, style=style)

    # JSON 파일 로드 및 설정
    with open("./final_test.json", "r", encoding="utf-8") as f:
        workflow_data = f.read()
    workflow = json.loads(workflow_data)

    # 랜덤 시드 값 설정
    seed = random.randint(1, 1000000000)
    workflow["3"]["inputs"]["seed"] = seed

    # prompt 설정
    workflow["6"]["inputs"]["text"] = prompt

    # WebSocket 연결
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
    images = get_images(ws, workflow)

    # 이미지 저장
    file_path = None
    for node_id in images:
        for image_data in images[node_id]:
            img = Image.open(io.BytesIO(image_data))
            file_path = os.path.join(BASE_DIR, f'{filename}.png')
            img.save(file_path, format="PNG")
        
    return file_path

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

@app.post("/verification", tags=["Face"])
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
@app.post("/recognition", tags=["Face"])
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

@app.post("/img_random", tags=["Image"])
async def image_maker_random():
    # 고유한 파일명 생성 (예: UUID 기반)
    unique_filename = str(uuid.uuid4())
    file_path = load_image(unique_filename)
    
    # 파일이 제대로 생성되었는지 확인
    if file_path and os.path.exists(file_path):
        return FileResponse(file_path)

    # 파일이 없으면 404 에러 반환
    raise HTTPException(status_code=404, detail="File not found")

# 이미지를 생성하는 엔드포인트
@app.post("/img_chat", tags=["Image"])
async def image_maker_chat(question: str):
    # 고유한 파일명 생성 (예: UUID 기반)
    unique_filename = str(uuid.uuid4())
    file_path = await load_image_chat(unique_filename, question)
    
    # 파일이 제대로 생성되었는지 확인
    if file_path and os.path.exists(file_path):
        return FileResponse(file_path)

    # 파일이 없으면 404 에러 반환
    raise HTTPException(status_code=404, detail="File not found")

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

# docker system prune -a --volumes -f

# docker exec -it comfyui_service /bin/bash
