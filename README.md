# [iM DiGital Banker] ML 컴퍼티션

## Churn(고객 이탈) Classifiaction 인사이트 분석

본 프로젝트는 은행 고객 데이터에서 **고객 이탈(Churn)**에 영향을 미치는 주요 요인을 분석하고, **AutoML**과 **앙상블 기법(Stacking)**을 활용하여 예측 모델의 성능을 극대화하는 것을 목표로 합니다.

### 기간

2026년 4월 10일 (금)

### 데이터

Bank Customer Churn Dataset (row: 10000, col:12)  
출처: [Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data)

### 라이브러리 리스트

#### Data Manipulation

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

#### Visualization

![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![SHAP](https://img.shields.io/badge/SHAP-20222F?style=for-the-badge&logoColor=white)

#### AutoML & Optimization

![PyCaret](https://img.shields.io/badge/PyCaret-630075?style=for-the-badge&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-20222F?style=for-the-badge&logoColor=white)

#### Machine Learning

![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-0080FF?style=for-the-badge&logoColor=white)

### 수행 과정

1. PyCaret 기반 AutoML 모델 탐색
   1. compare_models() 사용하여 F1-score 기준 최적의 후보군 선정
   2. Top 5 Models: **AdaBoost, CatBoost, LightGBM, Gradient Boosting, Random Forest**
      > CatBoost 제외, 네 가지 사용

<br />

1. 데이터 전처리 및 EDA 수행
   1. Label Encoding: 범주형 변수(country, gender) 수치화
   2. Feature Selection: 상관관계 및 시각화 분석을 통해 핵심 변수(**age, products_number, country, balance**) 선정
   3. 인사이트:  
      **독일 거주 고객 및 40~60대 중장년층**의 이탈률이 상대적으로 높음
      **보유 상품 수**가 적을수록 이탈 위험이 큼 → Cross-selling 전략의 유효성 확인

<br />

1. 하이퍼 파라미터 튜닝
   1. **Optuna** 프레임워크를 도입하여 개별 모델의 파라미터 최적화
   2. n_estimators, learning_rate, max_depth 등 주요 하이퍼 파라미터의 범위를 설정하여 모델별 20회 수행

<br />

4. SHAP Value를 활용한 변수 중요도 시각화
   1. Random Forest 모델을 대상으로 TreeExplainer 수행
   2. 분석 결과:  
      <img width="754" height="312" alt="SHAP value" src="https://github.com/user-attachments/assets/cdf7a506-3455-417f-984c-97523d362a13" />  
      **age**가 이탈 예측의 가장 강력한 지표임을 확인 (양의 상관관계)  
      고액 자산가(**balance**)의 이탈 경향 파악  
      → **40-60대 VIP 타겟팅** 필요성 도출

<br />

5. **앙상블(Stacking) 모델** 구축 및 최종 F1-score 산출
   1. Base Models: AdaBoost, LightGBM, GradientBoost, RandomForest
   2. Meta Learner: Logistic Regression (cross validation = 5)
   3. 최종 결과:  
      F1-score 0.5523  
      개별 모델 성능보다 향상되지 않았음

### 데이터 분석 인사이트 및 마케팅 전략 제안

1. 시니어 케어 서비스 강화 (age 대응):  
   40~60세 중장년층 고객을 위한 전담 자산 관리 서비스나 상속/연금 특화 상품을 출시

2. 크로스 셀링(Cross-selling) 유도 (products_number 대응):  
   상품을 1개만 이용 중인 고객에게 두 번째 상품(적금, 보험, 신용카드 등) 가입 유도 - 금리 혜택을 제공하여  
   락인 효과 노림

3. 우량 고객 및 독일 집중 마케팅 (balance & country 대응):  
   '독일의 고액 자산가'가 이탈하지 않게 VIP 우대 또는 지역 특화 금융 관리 도입

### 모델 성능

분류 모델용 지표인 F1-score 이용하여 측정

| 모델명             |   F1-score   |
| :----------------- | :----------: |
| AdaBoost           |   `0.5748`   |
| CatBoost           |   `0.5732`   |
| LightGBM           |   `0.5726`   |
| GradientBoost      |   `0.5708`   |
| RandomForest       |   `0.4800`   |
| **Stacking Model** | **`0.5523`** |

### 개선 방안

- 최적 모델에서 제외했던 CatBoost 사용
- 최적 모델 중 성능이 가장 낮았던 RF 대신 성능 더 높은 모델로 시도
- 스태킹 후방 모델 변경 또는 cv 조정

### 레퍼런스

reference.ipynb
