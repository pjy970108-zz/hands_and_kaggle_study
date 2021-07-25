## 최적화

### <목차>
1. 최적화?
    1) 최적화의 정의
    2) constrained optimization/unconstrained optimization
    3) 최소화 문제/ 최대화 문제

2. 최적화 원리

3. 최적화 기법들
    1) 경사하강법(배치 경사하강법, 확률적 경사하강법, 미니배치 경사하강법)
    2) Momentum
    3) Adagrad
    4) Adam

4. 조기 종료


**1-1)** 최적화의 정의 : objective function의 함수값을 최적화하는 파라미터를 찾는 문제
<br>
EX) 목적 함수가 y=2x+1로 표현되면 일변수 함수 최적화
<br>
    y=ax+bx_2...으로 구성되면 선형 최적화 문제
<br>
    f(x, y) = y+x^2으로 표현되면 비선형 최적화 문제라 한다.

<br>
그러므로 선형회귀를 최적화 하기위해서는 선형최적화 문제로 접근해야한다.
<br>
<br>

**1-2)** 목적함수 외에 파라미터가 만족해야할 제약조건이 있는 경우 constrained optimization/ 없는 경우 unconstrained optimization라 한다.

EX) 라쏘, 릿지등등 제약조건이 있는 경우 constrained optimization이다.

**1-3)** 최적화 문제는 최대화 문제(이윤, 점수 등)일 경우, 최소화 문제(loss, cost, error 등)

<br>

**2. 최적화 원리** : 함수 값이 감소하는 방향(최대화면 증가)하는 방향으로 파라미터를 이동해내가는 방법


**3. 최적화 기법들**

 ![image](https://user-images.githubusercontent.com/63804074/126898885-4215f75a-371c-4e41-acb2-2cd4f9b9348d.png)
 <br>

 **1) 경사하강법**

 <br>

 ![image](https://user-images.githubusercontent.com/63804074/126898911-ed55efae-10e8-4e3f-bbed-81bf41a75bf3.png)
 <br>

 첫번째) 초기값 설정
 <br>
 두번째) 위의 식 반복
 <br>
 세번째) 적당한 횟수로 반복해서 최적의 X_t 찾는다.
 <br>
 
    - 배치 경사하강법 : 학습 한번에 모든 데이터 셋의 기울기를 구하는 방식
    안정적인 수렴, 병렬처리가능하지만 메모리 문제, 지역 최적화 문제 발생
    
    - 확률적 경사하강법 : 학습 한법에 임의의 데이터에 대해서 기울기 구하는 방식
    빠른 최적화, 지역 최적화 회피할 수 있지만 병렬처리가 불가하고 안정적으로 수렴할 수 없다.
    
    - 미니 배치 경사하강법 : 학습 한번에 데이터 셋 일부에 대해서 기울기 구하는 방식
    배치 경사하강법과 확률적 경사하강법의 장점을 합친 경사하강법

<br>

![image](https://user-images.githubusercontent.com/63804074/126899052-fbbe7f7f-8e39-45ca-a9c4-95b8ae3f99df.png)
 
 <br>

 EX) 라쏘, 릿지, 엘라스틱넷, 로지스틱 경사하강법 적용 
 
 <br>

 **라쏘**: L1-Norm을 사용한 회귀이다. feature가 영향력이 낮다면 0으로 수렴한다. 학습시 feature의 제거 가능
 <br>

 ![image](https://user-images.githubusercontent.com/63804074/126899203-34dc90ef-30c8-4373-8598-5afe75d0e22a.png)

 <br>
 
 **릿지** : L2-norm을 사용한 회귀방법. feature가 영향력이 낮다면 0에 가까운 가중치를 준다. 학습시 feature의 영향력을 최소화
 <br>

 ![image](https://user-images.githubusercontent.com/63804074/126899213-c314cdcd-484b-45fe-b827-3bc82cb812bf.png)
 
 <br>

 **엘라스틱 넷** : 라쏘와 릿지회귀 방법을 합쳐서 조절함
 <br>

![image](https://user-images.githubusercontent.com/63804074/126899230-41f5a7e3-7ae0-46c0-9234-a39cf0dc792c.png) 


 **<최적해 찾기>**
 ![image](https://user-images.githubusercontent.com/63804074/126899253-2452e033-2336-455a-bcca-d11d05ec9a94.png)
 
 <br>

 **로지스틱**
 ![image](https://user-images.githubusercontent.com/63804074/126899351-ebeddf93-4933-4d18-95f5-98d2f7f4fdd9.png)
 <br>

 **로지스틱 경사하강법**
 ![image](https://user-images.githubusercontent.com/63804074/126899364-c0ba2c69-f3a1-45a0-8e89-57e11f156268.png)
 <br>

 **2) momentum** : 관성을 통해 가속하여 수렴
 ![image](https://user-images.githubusercontent.com/63804074/126899524-1db92c49-5a7a-49fd-bba0-c49380d7c1bd.png)
 <br>

 **3) Adagrad** : 개별 매개변수에 적응적으로 학습률을 조정하면서 학습
 ![image](https://user-images.githubusercontent.com/63804074/126899545-b5ec0124-afca-4548-ae94-90634fbde1c2.png)
 <br>

 **4) Adam** : momentum과 RMSprop을 합친 형태. 딥러닝 문제에 최적화에 좋다.
 ![image](https://user-images.githubusercontent.com/63804074/126899559-fca8ff51-1506-4716-9d6e-04d85e05c21c.png)
 <br>

**4. 조기종료** : 검증셋의 에러가 올라가고 훈련셋의 에러가 줄어들어 overfitting이 발생하기 전에 조기에 학습을 종료 시키는것.

![image](https://user-images.githubusercontent.com/63804074/126899616-07795a99-d866-437a-a80e-a2cf2c43ee58.png)
 
[reference] : https://darkpgmr.tistory.com/149

