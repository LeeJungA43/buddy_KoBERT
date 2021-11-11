import os
import sys
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from flask import Flask, render_template, request, jsonify

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


# Post와 GET 작업시 반복해서 수행되는 부분
@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    if request.method == 'POST': #post 방식으로 전송될 경우
        # data를 json 형식으로 POST 받음
        the_question = request.get_json()
        # print(the_question)

        # post 받은 문장을 사용해서 model 카테고리 및 답변 추출
        data = kobert_input(tokenizer, str(the_question['question']), device, 512)

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

        # 마찬가지로 json 형식으로 return
        return jsonify({"response": str(response)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)