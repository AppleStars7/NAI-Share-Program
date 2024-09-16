from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
import requests
import random
from datetime import datetime
from pathlib import Path
import io, base64, zipfile
from queue import Queue
from threading import Thread, Lock
import time
import os
from flask_session import Session

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

BASE_URL = "https://image.novelai.net"
ACCESS_TOKEN = "NAI TOKEN" # NAI 토큰 입력

task_queue = Queue()
task_status = {}
user_tasks = {}

image_delay = 8 # 이미지 생성 딜레이(기본값은 8)
last_request_time = 0
time_lock = Lock()

@app.route('/')
def index():
    return send_file('index.html')

def validate_image_size(width, height):
    if width < 64:
        width = 64
    if height < 64:
        height = 64
    
    if width * height > 1048576:
        return None, None, "Image size is too large. Maximum allowed size is 1048576 pixels."
    
    return width, height, None

@app.route('/generate', methods=['POST'])
def generate_image():
    global last_request_time

    data = request.json
    client_id = data.get('client_id')
    
    if not client_id:
        return jsonify({"error": "Client ID is required"}), 400

    if client_id in user_tasks:
        return jsonify({"error": "You already have an active request"}), 429

    data = request.json
    width = int(data.get('width', 1024))
    height = int(data.get('height', 1024))
    model = data.get('model', 'nai-diffusion-3')

    width, height, error_message = validate_image_size(width, height)
    if error_message:
        return jsonify({"error": error_message}), 400

    with time_lock:
        current_time = time.time()
        time_since_last_request = current_time - last_request_time
        if time_since_last_request < image_delay:
            wait_time = image_delay - time_since_last_request
            time.sleep(wait_time)
        last_request_time = time.time()

    task_id = str(time.time())
    task_status[task_id] = 'queued'
    user_tasks[client_id] = task_id
    
    task_queue.put((task_id, data, client_id, width, height, model))
    
    return jsonify({"task_id": task_id}), 202

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    status = task_status.get(task_id, 'not_found')
    if status == 'completed' or status == 'error':
        client_id = request.args.get('client_id')
        if client_id in user_tasks:
            del user_tasks[client_id]
    return jsonify({"status": status})

def worker():
    while True:
        task_id, data, user_id, width, height, model = task_queue.get()
        task_status[task_id] = 'processing'
        
        #기본 프롬 설정
        prompt = data.get('prompt', '') + ", amazing quality, very aesthetic, incredibly absurdres"
        negative_prompt = data.get('negative_prompt', '') + ", worst quality, bad quality, very displeasing, displeasing, lowres, bad anatomy, bad perspective, bad proportions, bad aspect ratio, bad face, bad teeth, bad neck, bad arm, bad hands, bad ass, bad leg, bad feet, bad reflection, bad shadow, bad link, bad source, wrong hand, wrong feet, missing limb, missing eye, missing tooth, missing ear, missing finger, missing ear, extra faces, extra eyes, extra eyebrows, extra mouth, extra tongue, extra teeth, extra ears, extra breasts, extra arms, extra hands, extra legs, extra digits, fewer digits, cropped head, cropped torso, cropped shoulders, cropped arms, cropped legs, mutation, deformed, disfigured, unfinished, chromatic aberration, error, jpeg artifacts, watermark, unfinished, scan, scan artifacts, signature, artist name, artist logo, abstract, abstract background"

        #기본 세팅 설정
        try:
            image_data = generate_image_NAI(ACCESS_TOKEN, prompt, model, "generate", {
                "width": width,
                "height": height,
                "scale": 5,
                "sampler": "k_dpmpp_sde",
                "steps": 24,
                "seed": random.randint(0, 9999999999),
                "n_samples": 1,
                "ucPreset": 0,
                "qualityToggle": False,
                "negative_prompt": negative_prompt,
                "sm": True,
                "sm_dyn": False,
                "noise": 0,
                "noise_schedule": "native",
            })
            
            current_date = datetime.now().strftime("%Y%m%d")
            save_folder = Path(f"nai_generated_images/{current_date}")
            save_folder.mkdir(parents=True, exist_ok=True)
            
            zipped = zipfile.ZipFile(io.BytesIO(image_data))
            for idx, file_info in enumerate(zipped.infolist()):
                image_bytes = zipped.read(file_info)
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = save_folder / f"{current_time}_{task_id}_{idx}.png"
                filename.write_bytes(image_bytes)
            
            task_status[task_id] = 'completed'
        except Exception as e:
            print(f"Error processing task {task_id}: {str(e)}")
            task_status[task_id] = 'error'
        
        task_queue.task_done()

def generate_image_NAI(access_token, prompt, model, action, parameters):
    data = {
        "input": prompt,
        "model": model,
        "action": action,
        "parameters": parameters,
    }

    response = requests.post(f"{BASE_URL}/ai/generate-image", json=data, headers={"Authorization": f"Bearer {access_token}"})
    return response.content

worker_thread = Thread(target=worker, daemon=True)
worker_thread.start()

@app.route('/image/<task_id>', methods=['GET'])
def get_image(task_id):
    current_date = datetime.now().strftime("%Y%m%d")
    image_folder = Path(f"nai_generated_images/{current_date}")
    image_files = list(image_folder.glob(f"*_{task_id}_*.png"))
    
    if image_files:
        return send_file(str(image_files[0]), mimetype='image/png')
    else:
        return jsonify({"error": "Image not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)