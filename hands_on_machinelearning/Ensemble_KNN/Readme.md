# Ensemble
<br>

### 목차

1. Ensemble?
2. voting
3. bagging
4. Random Forest
5. Boosting
6. Stacking

<br>

#### 1. Ensmble? : 약분류기를 모아서 하나의 강한 분류기를 만드는 supervised Learning 기법

과적합을 방지하며 분산 or 편향을 줄이며 정형데이터에는 최고의 기법

<br>

#### 2. voting : 투표로 최종예측값을 뽑는다!!
1. Hard voting : 예측한 결과중 다수의 분류기가 결정한 값으로 최종예측
2. soft voting : 클래스의 결정확률을 평균하여 최종 예측

<br>

```{.python}

from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators(분류기), *, voting='hard' or 'soft', weights=None, n_jobs=None, flatten_transform=True, verbose=False)
   
   
```

#### 3. bagging(bootstrap aggregating) : 중복을 허용하여 train sample에서 여러번 sampling 하는 기법
#### pasting(페이스팅) : 중복을 허용하지 않고 샘플링하는 방식

<br>
방법 : bagging sample 이용해서 voting

![image](https://user-images.githubusercontent.com/63804074/128627454-8132f8dc-c03f-4851-9a0c-4b1c0bb9a82d.png)

<br>

- 범주형 변수는 다수결

- 연속형 변수는 평균으로 집계


**장점** 
분산을 줄이는 효과 : 원래 모델이 unstable하면(DT)면 급격한 분산 감소

**단점** : 해석하기 힘듦/ 계산 복잡

```{.python}
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor

clf = BaggingClassifier(base_estimator=SVC(), bootstrap=TRUE(배깅) or False(페이스팅), oob_score=True(oob)검증 사용), n_estimators=10, random_state=0).fit(X, y)

```


<br>

**voting과 다른점:** 매번 다른 샘플 랜덤 추출, 같은 종류의 모델 여러번 사용

<br>
특성 샘플링 : max_feature, bootstrap_features 활용하여 feature의 일부만 활용(image에 굿), 편향을 늘리고 분산을 낮춤

<br>

**OOB 평가:** 배깅은 36.8%의 부트스트랩이 생략되어있어, 생략된 샘플을 교차검증이 활용

<br>

#### 4. Random Forest : 트리로 bagging 구성 (randomness를 강조)

**장점:** 오버피팅과 DT의 불안정성 해결, input outlier에 강함, 특성 중요도를 확인 할 수 있음 <br>

~~~{.python}
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestClassifier(max_depth=2, random_state=0)
~~~

#### 5. boosting : 약한 학습기를 순차적으로 학습(예측 반복하여 잘못 예측한 데이터에 가중치 부여하여 오류 개선)

![image](https://user-images.githubusercontent.com/63804074/128627918-f71d33c0-a61f-41c2-8ca1-3d955062ff17.png)

<br>

**배깅과의 공통점** : 약분류기 여러개 사용
**배깅과의 차이점** : 
- weight를 부여, 학습하며 조절
- 배깅은 독립적으로 모델 학습시키지만, 부스팅은 이전 모델의 학습을 가중치를 통해 다음 모델 학습에 반영
- 전체 데이터를 모두 사용한다
- training error를 최대한 빨리 줄이고 배깅보다 성능이 좋다.

<br>
연속된 학습을 하기에 병렬화 불가능하다는 단점.

<br>

#### - Adaboost 학습 방법
<br>

**step : 가중치 초기화 -> 가중치를 사용해서 모델 생성 -> 모델에 대한 err 계산 -> err를 이용해서 모델에 대한 가중치 계산 -> 데이터 가중치 업데이트 (모델 갯수만큼 반복) -> 최종모델**

**단점** : weight가 낮은 데이터 주위에 높은 weight를 가진 데이터가 있으면 잘못분류되어 성능이 떨어지는 단점

~~~{.python}
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
ada_clf = AdaBoostClassifier(base_estimator= model]입력, *, n_estimators=50 (최대 estimator), learning_rate=1.0(학습률), algorithm='SAMME.R'(다중 클래스(클래스 확률에 기반)))
~~~

학습률이 낮으면 트리가 많이 필요하지만 예측 성능은 좋아짐

<br>

#### - gradient 부스트 학습방법 : 경사하강법을 이용하여 오류값을 최소화 하는 방식으로 학습
<br>

**Adaboost와 차이점**: 오차를 최소화하기 위해 residual을 최소화 하는 방법으로 학습 진행
<br>

~~~{.python}
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
~~~

<br>

#### - Xgboost : gradient 부스트 기반하고 Tree를 만들때 병렬처리를 가능케 하여  gradient 부스트 속도 개선.

- 오버피팅에 강하고, 다른 알고리즘과 연계하기 좋다.

~~~{.python}
import xgboost as xgb
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

param = {'max_depth': 4, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic' }
num_round = 100
bst = xgb.train(param, dtrain, num_round)

preds = bst.predict(dtest)
~~~

- 일반 파라미터
booster : 어떤 booster를 쓸지(gbtreem gblinear 등등)
nthread : 몇개의 thread를 동시에 처리할지
num_feature : feature를 ㅁㅊ개 사용할지

- 부스팅 파라미터
eta : 학습률
gamma : 커지면 트리 깊이가 줄어들어 보수적인 모델이됨
max_depth : 트리당 깊이 키울수록 과적합 위험
lambda : L2 weight 숫자가 클수록 규제 높음 
alpha : L1 weights 숫자가 클수록 규제 높음


- 학습과정 파라미터
object : 목적함수 (reg:squarederror, reg:logistic 등등)
eval_metric : 평가함수 조정 (RMSE, log loss 등등)

num_rounds : epoch와 같다

#### - LGMB : XGboost보다 학습에 걸리는 시간이 적고 메모리 사용량 적다. 

~~~{.python}
import lightgbm as lgb 
train_ds = lgb.Dataset(X_train, label = y_train) 
test_ds = lgb.Dataset(X_val, label = y_val) 

params = {'learning_rate': 0.01, 'max_depth': 16, 'boosting': 'gbdt', 
'objective': 'regression', 'metric': 'mse', 'is_training_metric': True, 'num_leaves': 144, 'feature_fraction': 0.9, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'seed':2020} model = lgb.train(params, train_ds, 1000, test_ds, verbose_eval=100, early_stopping_rounds=100) 

y_pred=model.predict(X_val)
~~~

![image](https://user-images.githubusercontent.com/63804074/128628597-432e20f8-9dd7-47a4-99ac-3399b864b83f.png)

#### -catboost :  Catergorical feature를 처리하는데 중점을 둔 알고리즘


#### Stacking : 두개 이상의 다른 모델을 조합해서 최고의 성능을 내는 모델을 만듦

1. 데이터 분리
2. train set을 n개의 머신러닝 모델이 학습
3. 모델마다 x_val을 넣어서 y_val의 predict을 얻는다
4. 모델에서 얻은 predict 값을 학습데이터로 사용
5. predict 데이터 값과 y_val 값으로 학습
6. 새로운 학습 데이터를 얻을때처럼 x_test로 새로운 test 데이터 획득 그리고 y_test로 최종평가

![image](https://user-images.githubusercontent.com/63804074/128628735-7729f5c6-48a6-4e2b-9bef-68b3618e141e.png)

기본 stacking은 오버피팅 문제로 사용하지 않는다.

#### cross validation 기반 stacking

1. data Fold로 나눔
2. 각 모델 별로 Fold로 나누어진 데이터를 기반으로 훈련
    1) 각 fold 마다 뽑아진 훈련 데이터로 모델을 훈련(검증 데이터 활용해서 Val 예측)
    2) 각 fold 마다 나온 model을 기반으로 원본 X_test 데이터를 훈련하여 저장
3. 2까지 진행해서 나온 각 모델별 예측값을 stacking 하여 최종 모델 훈력 데이터로 사용

4. 2-2에서 나온 데이터로 예측을 수행하여 predict값 뽑아냄
5. 4에서 나온 predict과 y_test값을 비교해서 최종모델 평가


![image](https://user-images.githubusercontent.com/63804074/128628937-0f7ff05f-3bd0-40b9-b0d2-d7a44104475a.png)


~~~{.python}
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

X, y = load_iris(return_X_y=True)

estimators = [
     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))]
     
     
clf = StackingClassifier(estimators=estimators, cv=None (cross validation), final_estimator=LogisticRegression())
~~~


# KNN

### 목차
1. 분류? 군집화?
2. KNN이란?
3. DIstance 측정법
4. KNN 고려사항
5. KNN 장단점

#### 1. 분류? 군집화?
![image](https://user-images.githubusercontent.com/63804074/128629327-d84b7df9-5068-4360-be26-98a0c45eec63.png)

**분류** : 지도학습
**군집화**: 비지도학습

#### 2. KNN이란? : K개의 가까운 이웃을 찾아 분류하는 방법
<br>

![image](https://user-images.githubusercontent.com/63804074/128629461-308e4f45-e016-46e6-83c0-9eebf4b8a826.png)

<br>

**- classfication step**

1. 새로운 데이터가 들어오면
2. 모든 데이터들과의 거리를 구해서
3. 가장 가까운 K개의 데이터를 선택하고
4. 이 K개 데이터의 클래스를 확인해 다수의 데이터가 속한 클래스를 찾는다.
5. 이 클래스를 새로운 데이터의 클래스로 할당


![image](https://user-images.githubusercontent.com/63804074/128629505-dad05d92-3414-4441-88ad-dbd7537daae4.png)

**- Regression step**

1. 새로운 데이터가 들어오면
2. 모든 데이터들과의 거리를 구해서
3. 가장 가까운 K개의 데이터를 선택하고
4. 이 K개 데이터의 평균값 계산한다.
5. 이 값을 새로운 데이터의 예측값으로 사용한다.

~~~{.python}
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsClassifier(n_neighbors=5, *, weights='uniform' or 'distance, algorithm='auto', p=2, metric='minkowski' or 'chebyshev' or 'euclidean' 등등)

neigh.fit(X, y)
print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))
~~~

#### 3. Distance 측정법

- **유클라디안, 맨하탄**
![image](https://user-images.githubusercontent.com/63804074/128629712-2b40f248-6be7-44c8-b7f5-c2712994ebb4.png)

- **마할라 노비스 거리** : 데이터의 평균과 표준편차를 고려했을때 얼마나 중심에서 멀리 떨어져있는지(밀도를 고려), 변수들 간의 상관관계가 존재하는 경우 좋다.

![image](https://user-images.githubusercontent.com/63804074/128629740-a98a0b29-ea7f-429e-a9b9-4637153106cb.png)

#### 4. KNN 고려사항

- k 갯수 선택법 : k 값이 작을수록 동작을 잘한다, 1~20 사이의 값으로 설정, 홀수로 설정 (grid-search or k-fold cross validation)으로 성능이 좋은 k 값 및 거리 설정

- feature scale : 변수 단위에 민감하기때문에 scale 해주어야함(min-max, standard). 범주형일 경우 one-hot 인코딩

- weighted knn : 거리에 따라 영향력을 달리 주고 싶을때(유사도가 높은 이웃에 가중치를 줌)

#### 5. KNN 장단점

**장점** 
- 예측 모델을 따로 학습시킬 필요가 없다
- 데이터가 많으면 좋은 성능을 보임
- 간단한 모델


**단점**

- 거리 계산을 해야하기때문에 속도가 다른 알고리즘에 비해 느림
- 높은 차원의 데이터를 다루는 경우에는 성능이 낮다.
