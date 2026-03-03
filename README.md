# 📊 국비교육 감정분석 프로젝트 분석 리포트

본 문서는 `국비교육_감정분석.py` 파일을 기반으로, 프로젝트의 시스템 아키텍처 및 정량적 평가 결과를 분석하여 정리한 리포트입니다.

---

## 1. 🏗️ 시스템 아키텍처 (System Architecture)

본 프로젝트는 데이터 수집부터 모델 배포 직전까지의 **End-to-End 자연어 처리(NLP) 딥러닝 파이프라인**을 구축하였습니다. 전체 아키텍처는 크게 **4단계(수집 → 전처리 → 모델 체인 → 예측)**로 구성되어 있습니다.

### 1-1. 데이터 파이프라인 (Data Pipeline)
*   **크롤링 (Crawling):** `google_play_scraper` 라이브러리를 활용하여 배달 앱(요기요, 쿠팡이츠 등)의 구글 플레이스토어 리뷰 데이터를 직접 수집했습니다.
*   **전처리 (Preprocessing):**
    *   한국어 정규표현식(`[^ㄱ-ㅎㅏ-ㅣ가-힣 ]`)을 통한 노이즈 제거
    *   중복 데이터 및 결측치(NaN) 제거
*   **데이터 증강 (Data Augmentation - EDA 적용):**
    *   긍정/부정 데이터의 불균형 해소 및 모델 파인튜닝 성능 향상을 위해 EDA(Easy Data Augmentation) 기법 도입.
    *   **동의어 치환(Synonym Replacement):** 직접 구축한 형태소 기반 커스텀 동의어 사전을 활용해 임의의 단어를 동의어로 변경.
    *   **랜덤 삭제(Random Deletion):** 일정 확률(p=0.2)로 문장 내 형태소를 랜덤 삭제하여 모델의 견고성(Robustness) 강화.

### 1-2. 자연어 처리 및 피처 엔지니어링 (NLP & Feature Engineering)
*   **형태소 분석:** `KoNLPy`의 `Okt` 분석기를 사용하여 명사/동사 추출 및 토큰화 진행.
*   **불용어 제거:** '의', '가', '이' 등 의미 분석에 불필요한 조사/어미 리스트(Stopwords)를 정의하여 노이즈 필터링.
*   **토큰화 및 패딩 (Tokenization & Padding):**
    *   Keras `Tokenizer`를 통해 텍스트를 정수 인덱스 시퀀스로 변환.
    *   입력 시퀀스의 최대 길이(maxlen)를 100으로 설정하여 `pad_sequences` 적용.

### 1-3. 딥러닝 모델 아키텍처 (Model Architecture)
한국어 텍스트 문맥 내러티브를 분석하기 위해 **연속된 시계열 데이터 처리에 강력한 LSTM 신경망 구조**를 채택했습니다.
*   **Input Layer:** `Embedding` 레이어 (임베딩 차원: 100)
    *   텍스트 데이터를 밀집 벡터(Dense Vector) 공간으로 투영.
*   **Hidden Layer:** `LSTM` (장단기 메모리, 유닛 수: 128)
    *   문장 내 단어들의 순서와 문맥(Context) 상태를 기억하고 학습.
*   **Output Layer:** `Dense` (활성화 함수: `sigmoid`)
    *   최종적으로 0(부정)과 1(긍정) 사이의 확률값을 반환하는 이진 분류기(Binary Classifier).

---

## 2. 📈 정량적 평가 (Quantitative Evaluation)

딥러닝 모델의 성능 극대화를 위한 하이퍼파라미터 튜닝 및 최적화 전략이 적용되었으며, 검증 데이터를 통해 모델의 안정성과 정확도를 평가했습니다.

### 2-1. 학습 환경 및 파라미터 (Training Setup)
*   **손실 함수 (Loss Function):** `binary_crossentropy` (이진 감성 분류에 최적화)
*   **옵티마이저 (Optimizer):** `rmsprop` (순환 신경망-RNN 계열에서 기울기 소실 문제를 효과적으로 제어)
*   **평가 지표 (Metrics):** `Accuracy` (정확도)
*   **Epoch & Batch:** 최대 15 Epochs 학습 / `batch_size=64`
*   **데이터 분할:** Train Data의 20%를 검증 데이터(Validation Set)로 활용 (`validation_split=0.2`).

### 2-2. 콜백 최적화 전략 (Callbacks Optimization)
과적합(Overfitting) 방지 및 최고의 가중치(Weights) 저장을 위해 두 가지 콜백 함수를 도입했습니다.
*   **Early Stopping (조기 종료):**
    *   모니터링 지표: 검증 손실률(`val_loss`)
    *   조기 종료 조건: 손실률이 4번 연속(`patience=4`) 개선되지 않을 경우 즉시 학습 중단하여 과적합 방지.
*   **Model Checkpoint (모델 체크포인트):**
    *   모니터링 지표: 검증 정확도(`val_acc`)
    *   기능: 에포크마다 검증 정확도를 평가하여 가장 높은 정확도를 기록한 순간의 모델 가중치만 `best_model.h5`로 저장(`save_best_only=True`).

### 2-3. 최종 추론 엔진 성능 (Inference Performance)
*   저장된 `best_model.h5`와 `tokenizer.pickle`을 로드하여 독립적인 `sentiment_predict()` 추론 함수를 구성했습니다.
*   **추론 방식:** 새로운 문자열 입력 -> 정규화 -> 형태소 시퀀스 변환 -> 패딩 -> 최종 확률 추론.
*   **Threshold (임계값):** Sigmoid 출력 결과 `Score`가 `0.5` 이상이면 긍정, 미만이면 부정으로 최종 분류하는 스코어링 시스템 구현.




프로젝트 기간 : 2023/10/5 ~ 2023/10/24

추가연구기간 : 2025/07/21 ~ 2025/08/05

인원및목표 : 5명/배달 앱 리뷰를 활용한 텍스트 감정분석

주요역할 : 데이터증강,워드클라우드(쿠팡이츠,요기요)

사용한 데이터:쿠팡이츠,요기요 배달앱 리뷰데이터

결과: 긍,부정분류정확도(accuracy) 약 93%

## 감정분석 프로젝트 정리
https://data8968.tistory.com/13
 
### 데이터 증강 참고
https://fish-tank.tistory.com/95


### 태블로 활용
https://public.tableau.com/app/profile/juoh.hong/viz/__17608347454560/2
