import sys
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from tqdm import tqdm

import torch
from transformers import AdamW
from torch.utils.data import dataloader
from dataloader.wellness import WellnessTextClassificationDataset
from model.kobert import KoBERTforSequenceClassfication

import torch
import torch.nn as nn
import random

from model.kobert import KoBERTforSequenceClassfication, kobert_input
from kobert_transformers import get_tokenizer

sys.path.append('E:')
sys.path.append('E:/buddy')


def load_wellness_answer():
  root_path = "E:/buddy"
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'
model = pickle.load(open('E:/buddy/checkpoint/kobert-wellnesee-text-classification.pkl', 'rb'))

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())

@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    if request.method == 'POST':
        the_question = request.form['question']

    category, answer = load_wellness_answer()

    ctx = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(ctx)

    model.to(ctx)
    model.eval()

    tokenizer = get_tokenizer()

    data = kobert_input(tokenizer, the_question, device, 512)

    output= model(**data)

    logit = output[0]
    softmax_logit = torch.softmax(logit, dim=-1)
    softmax_logit = softmax_logit.squeeze()

    max_index = torch.argmax(softmax_logit).item()
    max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

    answer_list = answer[category[str(max_index)]]
    answer_len = len(answer_list) - 1
    answer_index = random.randint(0, answer_len)

    response = answer_list[answer_index]

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888', debug=True)