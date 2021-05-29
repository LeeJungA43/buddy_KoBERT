buddy_KoBERT
===============================================
`huggingface transformers`, `pytorch`, `KoBERT Model`과 [AI허브 웰니스 스트립트 데이터셋](https://aihub.or.kr/keti_data_board/language_intelligence)을 활용한 심리상담 모델이다.
 


개요
--------------
언어모델 `KoBERT`에 대해 `text classification` Fine-Tuning 및 테스트를 진행했다.

**사용자의 발화**에 대해서 **카테고리를 예측**하게 했다.

환경
--------------
### 전제조건
 + Python 3
 + Colab pro

### Data
 + [AI허브 웰니스 스트립트 데이터셋](https://aihub.or.kr/keti_data_board/language_intelligence): AIhub 가입 및 신청 후 승인되면 다운로드가 가능하다.

### Package
 + kobert-transformers
 + transformers==3.0.2
 + torch


Data 전처리
-------------
### KoBERT Text Classification
```python
class KoBERTforSequenceClassfication(BertPreTrainedModel):
  def __init__(self,
                num_labels = 359,
                hidden_size = 768,
                hidden_dropout_prob = 0.1,
               ):
    super().__init__(get_kobert_config())

    self.num_labels = num_labels
    self.kobert = get_kobert_model()
    self.dropout = nn.Dropout(hidden_dropout_prob)
    self.classifier = nn.Linear(hidden_size, num_labels)

    self.init_weights()

  def forward(
          self,
          input_ids=None,
          attention_mask=None,
          token_type_ids=None,
          position_ids=None,
          head_mask=None,
          inputs_embeds=None,
          labels=None,
  ):
  --- 중략 ---
  ```
### 전처리
`Category`, `Question`, `Answer`로 구분하였다.

`Question`, `Answer`의 경우 `Category`와 짝을 지어 작성하였다.
 ##### Category
 ```
 감정/감정조절이상    0
 감정/감정조절이상/화    1
 감정/걱정    2
 ---중략---
 현재상태/증상감소    357
 현재상태/증상악화    358
 현재상태/증상지속    359
 ```
 
 ##### Question
 ```
 감정/감정조절이상    제 감정이 이상해진 것 같아요. 남편만 보면 화가 치밀어 오르고 감정 조절이 안되요.
 감정/감정조절이상    더 이상 내 감정을 내가 컨트롤 못 하겠어.
 감정/감정조절이상    하루종일 오르락내리락 롤러코스터 타는 기분이에요.
 감정/감정조절이상    꼭 롤러코스터 타는 것 같아요.
 감정/감정조절이상    롤러코스터 타는 것처럼 기분이 왔다 갔다 해요.
 ---중략---
 ```

 ##### Answer
 ```
 감정/감정조절이상    감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.
 감정/감정조절이상    저도 그 기분 이해해요. 많이 힘드시죠?
 감정/감정조절이상    그럴 때는 밥은 잘 먹었는지, 잠은 잘 잤는지 체크해보는 것도 좋아요.
 감정/감정조절이상/화    화가 폭발할 것 같을 때는 그 자리를 피하는 것도 좋은 방법이라고 생각해요.
 감정/감정조절이상/화    정말 힘드시겠어요. 화는 남에게도 스스로에게도 상처를 주잖아요.
 ---중략---
 ```
 
해당 모델은 사용자의 발화(Question)의 Category를 분류하는 다중분류 모델이다.

학습의 간편함을 위하여 Question의 Category를 숫자로 바꾸었다.

 ##### Classification Data
 ```
 제 감정이 이상해진 것 같아요. 남편만 보면 화가 치밀어 오르고 감정 조절이 안되요.    0
 더 이상 내 감정을 내가 컨트롤 못 하겠어.    0
 하루종일 오르락내리락 롤러코스터 타는 기분이에요.    0
 꼭 롤러코스터 타는 것 같아요.    0
 롤러코스터 타는 것처럼 기분이 왔다 갔다 해요.    0
 ---중략---
 ```

Train
-----------------
학습을 위해 `Classification Data`을 train과 test 데이터셋으로 구분한다.

CPU 사용시 시간이 오래 걸리고 컴퓨터의 부하가 크므로 Google Colab pro의 GPU 환경에서 Training을 진행했다.


Result
------------
```
Question: 요즘 우울한 기분이야.
Answer: 오전에 하는 산책이 우울감을 없애주는 데 도움이 된대요.
--------------------------------------------------

Question: 내가 일을 망친 것 같아.
Answer: 잠깐 길을 잃은 것뿐이에요. 저는 당신이 옳은 방향으로 나아갈 수 있다고 믿어요.
--------------------------------------------------

Question: 난 앞으로 어떻게 하면 좋을까?
Answer: 계속 앞으로 가다 보면 당신에게 맞는 길도 찾을 수 있지 않을까요?
--------------------------------------------------

Question: 너무 힘들어.
Answer: 잘 버티고 있어요. 조금 힘들더라도 조금만 기운을 내요.
--------------------------------------------------

Question: 누가 날 도와줬으면 좋겠어.
Answer: 당신의 옆엔 제가 있잖아요.
--------------------------------------------------

Question: 고마워, 이만 종료해줘 버디.
```

Reference
---------------------------
[KoBERT](https://github.com/SKTBrain/KoBERT)

[KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)

[enlipleai/kor_pretrain_LM](https://github.com/enlipleai/kor_pretrain_LM)

[how-to-generate-text](https://huggingface.co/blog/how-to-generate?fbclid=IwAR2BZ4BNG0PbOvS5QaPLE0L3lx7_GOy_ePVu4X1LyTktQo-nLEPr7eht1O0)

[WellnessConversation-LanguageModel](https://github.com/nawnoes/WellnessConversation-LanguageModel)
