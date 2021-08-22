## Dimensionality_Reduction

### <목차>

1. 차원축소??
2. 차원축소 기법들
    - 변수 선택법(lasso) : 불필요한 변수 제거
    - 변수 추출 : 변수를 조합하여 새로운 특징 생성
        - PCA(기본, 점진적, 랜덤, 커널)
        - LDA
        - t-sne
        - Isomap
        - LLE
        
<br>

### 1. 차원 축소?
<br>

변수가 늘어나고 차원이 커지면서 발생하는 차원의 저주를 해결하기 위해 변수를 줄이는것을 **차원축소**라 한다.

* 차원의 저주
![차원의 저주](https://user-images.githubusercontent.com/63804074/130346364-94b768dc-7b25-4c54-947f-649ad4541d9e.png)
<br>

변수(차원)이가 늘어날 수록 데이터(정보)들간의 **밀도가 감소**된다. 공간을 설명하기 위한 데이터의 부족으로 **과적합**문제가 발생하거나 **모델 성능**이 감소되는 문제가 발생

또, 늘어난 변수(차원)을 설명하기 위해 요구되는 데이터 수가 지수함수적 증식을 하게되어 **연산량이 급증**하게 된다.

<br>

### 2. 차원축소 기법들
따라서, 차원축소문제를 해결하기 위한 기법으로 **변수 선택법, 변수 추출법** 등이 있다.
<br>

- 변수 선택법: 원본 특성들 중에서 필요 없는 특징 제거
EX) lasso : 아래 사진 처럼 필요없는 변수에 가중치를 0을 줌으로써 원본 특성 중에서 필요없는 특징 제거
![image](https://user-images.githubusercontent.com/63804074/130346580-ee6dfc3d-abb1-4311-8d38-03e70d959d7d.png)
<br>

- 변수 추출법: 변수를 조합하여 새로운 특징 생성

#### 1) PCA : 가장많이 사용하는 차원 축소 알고리즘, 변수들의 전체 분산 대부분을 소수의 주성분을 통해 설명
 <br>
 
 **1-1) 기본 PCA - 분산이 최대인 축을 찾아서 차원을 축소**
 <br>
 
 **step** : 데이터 표준화 -> 공분산 행렬 계산 -> 고유값과 고유벡터를 구함 -> 적절한 벡터를 정해서 차원축소 수행
 
 **가정** : 데이터가 선형성을 띄어야함 / 찾은 축들은 서로 직교함 / 큰 분산을 갖는 방향이 중요한 정보를 담고 있다.
 
 ~~~{python}
 from sklearn.decomposition import PCA
 
 sklearn.decomposition.PCA(n_components=None, *)
 ~~~
 <br>
 
 n_components : 주성분 갯수 정하기
 <br>
 
 **최적 주성분 갯수는?**
 <br>
 
 1. elbow point : 고유값과 주성분 갯수 그래프에서 곡선의 기울기가 급격히 감소하는 지점
 2. Kaiser's Rule : 고유값이 1이상인 주성분들만 선택
 3. 누적 설명률이 70~80%인 지점
 <br>
 
 **1-2) 점진적 PCA(IPCA) - 훈련 세트를 미니 배치로 나눠서 PCA적용해서 메모리 절약**
  
  
 ~~~{python}
 from sklearn.decomposition import IncrementalPCA
 
 sklearn.decomposition.IncrementalPCA(n_components=None, *, batch_size=None)
 ~~~
<br>

n_components : 주성분 갯수 정하기 <br>
batch_size = NONE(default) 5*변수갯수

<br>

  **1-3) 랜덤 PCA- 확률적 알고리즘으로 주성분에 대한 근삿값을 빨리 찾음(계산 복잡도가 낮아 연산 속도가 빠름)**
<br>

 ~~~{python}
 from sklearn.decomposition import PCA
 
 sklearn.decomposition.PCA(n_components=None, *, svd_solver='"randomized")
 ~~~
<br>

  **1-4) 커널 PCA - 데이터가 비선형일 경우 사용**

 ~~~{python}
from sklearn.decomposition import KernelPCA
 
 sklearn.decomposition.KernelPCA(n_components=None, *, kernel='linear', gamma=None, degree=3, coef0=1, alpha=1.0, fit_inverse_transform=False)
 ~~~
 <br>

n_components : 주성분 갯수 정하기 <br>
kernel : {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed'} <br>
gamma : default(1/피쳐의 갯수) / 결정 경계의 곡률을 결정한다. <br>
degree : poly일때만 사용 <br>
coef0 : poly와 sigmoid일때만 사용 / 모델이 높은 차수와 낮은 차수에 얼마나 영향을 받을지 정함 <br>
fit_inverse_transform: True이면 원본 새믈과 제곱거리를 측정(재구성의 원상) <br>
<br>

최적의 커널을 구할때 kernel과 gamma정도만 고려해서 gridsearch사용하면 된다.
<br>

#### 2) LDA : 데이터의 분포를 학습하여 분리를 최적화 하는 결정경계를 만들어 데이터 분류
클래스를 분리 시키므로 SVM 분류기 등 다른 분류기 사용전에 차원축소용으로 사용
<br>

 ~~~{python}
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components = None)
 ~~~
 <br>

n_components : 주성분 갯수 정하기 <br>
<br>

#### 3) t-sne : 비슷한 샘플은 가까이 비슷하지 않은 샘플은 멀리 떨어지도록, 시각화에 사용되며 고차원 샘플을 군집화 할때 사용

 ~~~{python}
from sklearn.manifold import TSNE

sklearn.manifold.TSNE(n_components=2, *, perplexity=30.0, learning_rate=200.0)
~~~
<br>
 
n_components : 주성분 갯수 정하기 <br>
perplexity=30.0 최근접 이웃 수 데이터세트가 클수록 높여야한다. <br>
learning_rate=200.0 /  학습률이 낮으면 밀집됨 <br>

#### 4) Isomap : 각 샘플을 가장 가까운 이웃과 연결하는 식으로 그래프를 만든다. 샘플간의 지오데식 거리를 유지하면서 차원축소

 ~~~{python}
from sklearn.manifold import Isomap

sklearn.manifold.Isomap(*, n_neighbors=5, n_components=2, neighbors_algorithm='auto', metric='minkowski', p=2, metric_params=None)
~~~
<br>

n_components : 주성분 갯수 정하기 <br>
n_neighbors : 각점의 이웃의 갯수 <br>
neighbors_algorithm : {‘auto’, ‘brute’, ‘kd_tree’, ‘ball_tree’}, 가장 가까운 이웃 찾는 알고리즘 <br>
metric : 거리계산 <br>
p=2 : 'minkowski'일때 사용 <br>

<스위스 롤 축소>
![image](https://user-images.githubusercontent.com/63804074/130348947-8ce35fe3-0b1f-4e22-a694-3766e34cdc27.png)

#### 5) LLE(지역 선형 임베딩) : 비선형 차원 축소 기법, 시간 복잡도가 크기 때문에 대량의 데이터 셋에 적용하기 어렵다

 ~~~{python}
from sklearn.manifold import LocallyLinearEmbedding

sklearn.manifold.LocallyLinearEmbedding(*, n_neighbors=5, n_components=2, neighbors_algorithm='auto')
~~~
<br>

n_components : 주성분 갯수 정하기 <br>
n_neighbors : 각점의 이웃의 갯수 <br>
neighbors_algorithm : {‘auto’, ‘brute’, ‘kd_tree’, ‘ball_tree’}, 가장 가까운 이웃 찾는 알고리즘 <br>


<스위스 롤 축소>
![image](https://user-images.githubusercontent.com/63804074/130349160-6a77d409-8bea-4676-bc7c-fcff8e2d7d45.png)


