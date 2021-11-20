# Buddy 챗봇 모델을 API 형태로 발행할 때 필요한 py 파일

import sys
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bson
from tqdm import tqdm
from flask import Flask, render_template, request, jsonify, make_response
from flask_mobility import Mobility

import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig
import cognitive_face as CF

from transformers import AdamW
from torch.utils.data import dataloader
from dataloader.wellness import WellnessTextClassificationDataset
from model.kobert import KoBERTforSequenceClassfication
from model.kobert import KoBERTforSequenceClassfication, kobert_input
from kobert_transformers import get_tokenizer

# 로컬 폴더 path 추가 & root path 설정
sys.path.append('C:')
sys.path.append('C:/buddy')
root_path = "C:/buddy"

# 해당 파일을 처음 실행시켰을 때, 한 번만 실행되어야 할 것들
# Flask 설정
app = Flask(__name__)
#app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'

# Azure Speech 서비스 key와 지역 할당
speech_key, service_region = "<YOUR KEY>", "koreacentral"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

# STT/TTS 사전 설정. 한국어와 여성의 음성을 선택
speech_config.speech_synthesis_language = "ko-KR"
speech_config.speech_synthesis_voice_name = "ko-KR-SunHiNeural"

# Face Recogntion
KEY = "<YOUR KEY>"
CF.Key.set(KEY)
BASE_URL = 'https://koreacentral.api.cognitive.microsoft.com/face/v1.0/'
CF.BaseUrl.set(BASE_URL)
happiness, anger, contempt, disgust, fear, sadness = [0, 0, 0, 0, 0, 0]
img_url = './emotions/frame.png'

# category와 answer 파일 열어서 답변 리스트를 생성하는 함수
def load_wellness_answer():
  category_path = f"{root_path}/data/wellness_dialog_category.txt"
  answer_path = f"{root_path}/data/wellness_dialog_answer.txt"

  c_f = open(category_path, 'r', encoding='UTF-8')
  a_f = open(answer_path, 'r', encoding='UTF-8')

  category_lines = c_f.readlines()
  answer_lines = a_f.readlines()

  category = {}
  answer = {}
  for line_num, line_data in enumerate(category_lines):
    data = line_data.split('    ')
    category[data[1][:-1]]=data[0]

  for line_num, line_data in enumerate(answer_lines):
    data = line_data.split('    ')
    keys = answer.keys()
    if(data[0] in keys):
      answer[data[0]] += [data[1][:-1]]
    else:
      answer[data[0]] =[data[1][:-1]]

  return category, answer

# 챗봇 Model 불러오기 및 실행
# 지난 번 실행이 느렸던건 해당 부분이 chatbotResponse()에 포함되어 반복해서 불러오는 작업을 수행했기 때문
checkpoint_path = f"{root_path}/checkpoint"
save_ckpt_path = f"{checkpoint_path}/kobert-wellnesee-text-classification.pth"

category, answer = load_wellness_answer()

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

checkpoint = torch.load(save_ckpt_path, map_location=device) # 저장한 Checkpoint 불러오기

model = KoBERTforSequenceClassfication()
model.load_state_dict(checkpoint['model_state_dict'])

model.to(ctx)
model.eval()

tokenizer = get_tokenizer()

# 기본 home 출력하는 탬플릿
@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())

# 대기 시 수행하는 것. 먼저 말을 걸거나 표정이 안좋아보이기 전엔 대기
@app.route('/standby', methods=["GET", "POST"])
def standby():
    if request.method == 'POST': #post 방식으로 전송될 경우
        # files과 json 형식으로 데이터를 받음
        binary_data = {}
        the_files = request.files
        the_data = request.get_json()
        for file in the_files:
            binary_data[file] = the_files[file].read()

        # 받은 png와 wav 파일 저장
        f1 = open('./emotions/frame.png', 'wb')
        f1.write(binary_data['images'])
        f2 = open('./audio/user_voice.wav', 'wb')
        f2.write(binary_data['wave'])
        f1.close()
        f2.close()
        state = float(binary_data['state'])

        # 표정에서 감정 추출
        faces = CF.face.detect(img_url, True, False, "age, gender, emotion")
        for i in faces:
            happiness = i['faceAttributes']['emotion']['happiness']
            anger = i['faceAttributes']['emotion']['anger']
            contempt = i['faceAttributes']['emotion']['contempt']
            disgust = i['faceAttributes']['emotion']['disgust']
            fear = i['faceAttributes']['emotion']['fear']
            sadness = i['faceAttributes']['emotion']['sadness']
        bad_expression = anger + contempt + disgust + fear + sadness

        if happiness < bad_expression:  # 우울해 보일 때
            # 무슨 일이 있으신가요? 기분이 안좋아보여요. 제가 도울 수 있는게 있으면 말해주세요.
            # play_audio("./audio/
            state = state + 0.1 # 잘못 체크 할 수도 있으니 약간만 증가
            return jsonify({"response": "무슨 일이 있으신가요? 기분이 안좋아보여요. 제가 도울 수 있는게 있으면 말해주세요.",
                            "run_check": "1", "state": str(state)})
        else:  # 우울해 보이지 않을 땐 버디 이름을 부를때까지 대기
            audio_input = speechsdk.AudioConfig(filename="./audio/user_voice.wav")
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config,
                                                           language="ko-KR", audio_config=audio_input)
            the_question = speech_recognizer.recognize_once_async().get()
            if '버디' in the_question.text:  # 이름을 불렀을 때
                # 네, 부르셨나요?
                # play_audio("./audio/call_name.wav")
                return jsonify({"response": "네, 부르셨나요?", "run_check": "1", "state": str(state)})

        # 마찬가지로 json 형식으로 return
        return jsonify({"response": "대기", "run_check": "0", "state": str(state)})

# 본격적인 대화 부분
@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    if request.method == 'POST': #post 방식으로 전송될 경우
        # files 형식으로 데이터를 받음
        binary_data = {}
        the_files = request.files
        for file in the_files:
            binary_data[file] = the_files[file].read()

        # 받은 png와 wav 파일 저장
        f1 = open('./emotions/frame.png', 'wb')
        f1.write(binary_data['images'])
        f2 = open('./audio/user_voice.wav', 'wb')
        f2.write(binary_data['wave'])
        f1.close()
        f2.close()
        state = float(binary_data['state'])
        silence = int(binary_data['silence'])

        faces = CF.face.detect(img_url, True, False, "age, gender, emotion")
        for i in faces:
            happiness = i['faceAttributes']['emotion']['happiness']
            anger = i['faceAttributes']['emotion']['anger']
            contempt = i['faceAttributes']['emotion']['contempt']
            disgust = i['faceAttributes']['emotion']['disgust']
            fear = i['faceAttributes']['emotion']['fear']
            sadness = i['faceAttributes']['emotion']['sadness']
        bad_expression = anger + contempt + disgust + fear + sadness

        if happiness < bad_expression:  # 우울해 보일 때
            state = state + bad_expression * 0.5

        # STT API로 사용자의 발화를 문자로 변환
        audio_input = speechsdk.AudioConfig(filename="./audio/user_voice.wav")
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config,
                                                           language="ko-KR", audio_config=audio_input)
        the_question = speech_recognizer.recognize_once_async().get()
        print(the_question.text)

        if '종료' in the_question.text:  # 종료를 원할 때
            # 네. 종료하겠습니다.
            return jsonify({"response": "네. 종료하겠습니다.", "run_check": "0", "state": str(0),
                            "silence": str(silence), "music": str(0)})

        elif '음악' in the_question.text:  # 음악 재생을 바랄 때
            # 음악을 재생합니다.
            return jsonify({"response": "음악을 재생합니다.", "run_check": "0", "state": str(0),
                            "silence": str(silence), "music": str(1)})

        elif state > 10:  # 기준 수치 초과
            # 마음이 소란 스러울 때, 듣기 좋은 음악을 들어보시는 건 어떤가요? 원하신다면 음악 틀어줘, 라고 말해주세요.
            return jsonify({"response": "마음이 소란 스러울 때, 듣기 좋은 음악을 들어보시는 건 어떤가요? 원하신다면 음악 틀어줘, 라고 말해주세요",
                            "run_check": "1", "state": str(state), "silence": str(silence), "music": str(0)})

        elif len(str(the_question.text)) > 0: # 음성이 녹음되어 있을 때
            data = kobert_input(tokenizer, str(the_question.text), device, 512)

            output = model(**data)

            logit = output[0]
            softmax_logit = torch.softmax(logit, dim=-1)
            softmax_logit = softmax_logit.squeeze()

            max_index = torch.argmax(softmax_logit).item()
            max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

            answer_list = answer[category[str(max_index)]]
            answer_len = len(answer_list) - 1
            answer_index = random.randint(0, answer_len)

            response = answer_list[answer_index]
            state = state + 1

            return jsonify({"response": response, "run_check": "1", "state": str(state),
                            "silence": str(silence), "music": str(0)})

        else:
            silence = silence + 1
            return jsonify({"response": "대기합니다.", "run_check": "1", "state": str(state),
                            "silence": str(silence), "music": str(0)})

# wav 받고 wav 전송하기
@app.route('/test5', methods=["GET", "POST"])
def test5():
    if request.method == 'POST': #post 방식으로 전송될 경우
        binary_data = {}
        the_files = request.files
        for file in the_files:
            binary_data[file] = the_files[file].read()

        f1 = open('C:/buddy/test.wav', 'wb')
        f1.write(binary_data['wave'])
        f1.close()

        f1 = open('./audio/check_voice.wav', 'rb')
        wav_data = f1.read()
        f1.close()

        # make_response 사용
        response = make_response(wav_data, 200)  # 파일 데이터 저장
        #response.mimetype = 'audio/wav'

        return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
