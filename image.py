import json
from urllib import request, parse
import random
import uuid
import websocket
import urllib
import os
from PIL import Image
import io
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse

# ngrok http 8090 --domain adapted-charmed-panda.ngrok-free.app
# uvicorn image:app --reload --port 8090

app = FastAPI()

BASE_DIR = "./BASE_DIR"  # BASE_DIR을 명시적으로 정의
os.makedirs(BASE_DIR, exist_ok=True)

server_address = "127.0.0.1:8188"
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

def load_image(color, style, filename):
    with open("./final_back2_X.json", "r", encoding="utf-8") as f:
        workflow_data = f.read()

    workflow = json.loads(workflow_data)
    seed = random.randint(1, 1000000000)
    workflow["3"]["inputs"]["seed"] = seed
    
    workflow["6"]["inputs"]["text"] = f"""An image of a {style} cityscape with a focus on {color}.
    The scene features a mix of modern and traditional architecture, illuminated by city lights and set against a {color} sky.
    The city streets are bustling with activity, reflecting the vibrancy of the {style} design.
    The overall mood highlights a dynamic urban atmosphere, emphasizing the {color} tone in the environment."""

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = get_images(ws, workflow)  # workflow를 prompt로 전달

    file_path = None
    for node_id in images:
        for image_data in images[node_id]:
            img = Image.open(io.BytesIO(image_data))
            file_path = os.path.join(BASE_DIR, f'{filename}.png')
            img.save(file_path, format="PNG")
        
    return file_path

@app.post("/img_generator")
async def image_maker(color: str, style: str, filename: str):
    load_image(color, style, filename)
    return FileResponse(f'{BASE_DIR}/{filename}.png')

@app.get("/get-image")
async def api_get_image(filename: str):
    # 이미지 파일 경로 생성
    file_path = os.path.join(BASE_DIR, filename + '.png')

    # 이미지 파일이 실제로 존재하는지 확인
    if os.path.exists(file_path):
        return FileResponse(file_path)

    # 파일이 없으면 404 에러 반환
    raise HTTPException(status_code=404, detail="File not found")