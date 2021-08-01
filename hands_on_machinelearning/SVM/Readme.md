## SVM

### <목차>
1. SVM??

2. SVM 종류
    - hard
    - soft
    - non-linear
3. code


## 1. SVM?
선형, 비선형, 회귀, 이상치 탐색에도 사용할 수 있는 다목적 머신러닝 모델이다. 그러나 연산이 오래 걸린다는 단점 존재!

## SVM 종류


- hard : 모든 샘플이 도로 바깥쪽에 올바르게 분류되어 있다면 이를 하드 마진분류라고한다.
데이터가 선형적으로 구분될수 있어야하며, 이상치에 민감하다는 단점을 가진다.

- soft : hard 분류시 발생하는 에러를 허용하되 이 에러들에게 제약을 부여하는 방법을 soft margin svm이라 한다. <br>
제약을 부여하는 방법은 0-1 loss, hinge loss가 있다.
    <br>
    
   **0-1 loss** : error가 발생한 개수만큼 패널티 부여 
   <br> 
#### min|w| + C#error
   **hinge loss** : 오분류의 정도에 따라 error의 크기를 다르게 하는 것 
<br>
#### argmin|w|+c#E
<br>
여기서 c는 하이퍼 파라미터!!!
<br>

C가 크면 에러를 줄이는것이 중요하고 overfitting의 가능성이 있다.
<br>

C가 작으면 에러가 있어도 큰 영향이 없다. w를 줄이는게 중요하다. underfitting의 가능성
<br>

- non-linear : 선형으로 분류 불가 -> 커널트릭 사용
<br>

**kernel**: 저차원의 데이터를 고차원으로 매핑하는 것. 사용하는 이유??? 고차원 매핑 및 연산도 간단히 할 수 있다.

- kernel의 종류-
<br>

![image](https://user-images.githubusercontent.com/63804074/127771434-5fdb06ac-edeb-463a-8e61-6703c0cfcac2.png)
<br>

1. linear
2. polynomial
3. sigmoid
4. gaussian :여기서 감마와 분산은 반비례 관계이다. 즉, 감마가 줄어들면 표준편차가 증가함!!!
<br>

감마가 클수록 인접한 것들만 같은 영역으로 본다! 인접하지 않으면 먼곳으로 인식하고 overfitting 가능성이 높다!!!

### 최적의 값을 찾는 방법? grid search!!! but 오래걸림

## code
<br>

![image](https://user-images.githubusercontent.com/63804074/127771560-8aee81c8-cbc7-45ab-8a71-bd6ed38ba31a.png)

<br>

C: 하이퍼 파라메타로 C는 데이터 샘플의 오류를 허용하는 정도
<br>

kernel : linear, poly, rbf, sigmoid, precomputed 가 있고 rbf를 많이 씀
<br>
gamma : scale, auto 가 있고 rbf, sigmoid, poly에 사용. scale = 1/(feature 갯수*x분산) auto = 1/ feature의 갯수
결정 경계의 곡률을 결정한다.

<br>
coef0: float이고 모델이 높은 차수와 낮은 차수에 얼마나 영향을 받을지 정함
<br>