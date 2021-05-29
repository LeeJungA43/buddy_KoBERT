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
  ```



















