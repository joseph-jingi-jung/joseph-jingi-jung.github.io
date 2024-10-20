---
layout: post
title: AI604 - CH2 Linear classification & regression
subtitle: Linear classification & regression
date: 2024-10-20 22:58:00 +0900
category: content
tags:
  - vision
use_math: true
---
AI604 수업을 수강 후 정리한 내용이다. Stanford의 CS231n과 맞닿아 있다.

### Introduction

#### Image classification task는 2가지 기본적인 data-driven 방식 있다.
- K-nearest neighbor
- linear classifier

#### Image classification의 Challenges
- 조명의 변화(illumination changes)
- 배경(Background): 오브젝트가 작거나 배경이 유사한 경우 구분이 어려움
- 일부만 보임(Occlusion)
- Deformation: 물체의 모양이 고정되어있지 않음
- Intraclass variation: 같은 클래스 내에서도 다양한 모양을 보임
- Context: 상황에 따라 물체가 달리 보일 수 있음 ex) 철장 그림자로 인해 호랑이로 분류된 강아지

#### Traditional Approaches
- hand crafted features로 부터 시작
	- 코너, 엣지 등을 보고 연구자들이 복잡한 알고리즘 디자인

#### Machine learning approach
- Training dataset을 사용함
- 3단계 기법
	- 이미지와 레이블 데이터셋을 모으기
	- 머신러닝 알고리즘을 이용하여 분류기(Classifier)를 학습하기 
	- 새로운 이미지를 이용하여 평가하기

### Nearest Neighbor
- 특정 알고리즘 대신 큰 메모리를 이용해보자
- Train
	- 모든 데이터와 레이블을 기억하기
- Predict
	- 트레이닝 이미지와 가장 유사한 레이블을 통해 예측하기

#### 거리 기반 이미지 비교
- L1 distance: $d_1(I_1, I_2) = \sum_P \vert I_1^p - I_2^p \vert$
- Pixel-wise 절대값 비교 (Pixel-wise absolute value differences)
	- 각 픽셀별로 차를 구하고 절대값 후 전체를 합산
- 가장 거리차가 적은 이미지를 구하기

- Q. N 개의 예제에 대하여, 학습과 예측에 소요되는 시간은?
	- Train: O(1)
	- preidct: O(N)
	- 학습은 빠른데, 예측은 느려서 문제가 됨. 반대로 학습은 느려도 예측이 빨라야 함.

### K-Nearest Neighbors
- 가장 가까운 이웃의 레이블을 이용하는 것 대신, K개의 가까운 점에서 가장 많이 나오는 지점을 선택(Majority vote)

#### K-Nearest Neighbors: Distance Metric
- L1 distance: $d_1(I_1, I_2) = \sum_P \vert I_1^p - I_2^p \vert$
- L2 distance: $d_2(I_1, I_2) = \sqrt{\sum_p (I_1^P - I_2^P)^2}$
	- 데이터들이 Spares 하다고 생각되면 L1이 주로 사용됨.
	- Photometric(광학적 특성, 밝기 또는 색상 데이터) 를 다룰 때는 이미지의 많은 부분이 동일하거나 매우 유사한 색상/밝기를 가질 수 있어서 sparse하다고 표현. 따라서 이때에 L1을 사용

#### Hyperparameters
- 가장 좋은 k 는 무엇인가?
- 어떤 distance metric을 쓰는 것이 좋을까?
- Train, validation, test로 나누어서 validation set 으로 hyperparameters를 선택하고, test set에 평가
- Cross-validation
	- training 데이터를 여러 fold로 나누고, 각 fold를 한번씩 validation으로 두어 테스트하고, 그 평균을 구함.
	- 작은 데이터셋에 유용하고, deep learning 때 부터는 자주 쓰이진 않음

#### K-Nearest Neighbor: Universal Approximation
- 학습 샘플의 수가 무한하다면, nearest neighbor는 어떠한 함수도 표현이 가능하다.

#### Problem: Curse of Dimensionality
- 차원의 저주: 공간의 균일한 커버리지를 위해서는, 차원이 증가할수록 필요한 학습 데이터의 수가 기하급수적으로 증가한다.
- 따라서 32x32 흑백 이미지라 하더라도 $2^{32 \times 32} \approx 10^{308}$ 으로 경우의 수가 매우 많음

#### Problem: 픽셀의 거리 정보는 도움이 되지 않음
- 픽셀의 거리 정보(Distance metrics on pixels) 는 도움이 되지 않는다(not informative)
- Original 이미지와 3가지 변화된 이미지(Occluded, shifted, Tinted) 에 대하여 각각 pixel distance를 측정 하였을 때, 3가지의 거리가 모두 동일 할 수 있다.
	- Shifted 이미지는 semantic 정보에 아무런 변화가 없는데, occlueded 와 Tinted와 거리차가 같을 수 있는 문제가 있다.

### Linear classifier

#### Parametric Approach
- 입력: Image. Array of 32x32x3 numbers. 
- f(x, W) 를 통과
	- W: 파라미터(Parameters) 또는 웨이트(weights)
- 함수의 결과로 class scores를 나타내는 10개의 수를 나타냄

$$
f(x,W) = W \cdot x + b
$$

#### Visual viewpoint
- 각각의 weight는 class 당 template image로 볼 수 있다. (major concept of class)
- 여기서 score는 match score라 할 수 있음

#### Geometric viewpoint
- 데이터를 하이퍼 플레인으로 구분
- Weight 가 하이퍼플레인(Hyper plane) 을 결정

#### 정리
- Algebraic Viewpoint : $f(x,W) = Wx$
- Visual Viewpoint : One template for class
- Geometric Viewpoint : Hyperplanes cutting up space

#### Choose a good W
- Linear classifier는 좋은 Weight를 찾는 과정
1. Loss function을 정의하여, training data와의 score의 틀린정도(unhappiness) 를 수치화
2. Loss function을 최소화 할 수 있는 파라미터를 효율적으로 찾기 => Optimization

#### Loss function
- Loss function은 현재 classifier의 성능을 말해줌
- 주어진 데이터셋에 대하여 $\{(x_i, y_i)\}^N_{i=1}$ ,전체 데이터셋에 대한 Loss 평균은 아래와 같다

$$
L = \frac 1 N \sum_i L_i(f(x_i, W), y_i))
$$

### Multiclass SVM loss
- Diffrence in scores between correct and incorrect class

$$
L_i = \sum_{j \neq y_i} \text{max} (0, s_j - s_{y_i} + 1)
$$

- Q2. SVM loss의 최소/최대는
	- min : 0, max : 무한대
- Q3, 초기 W값이 작아서 S가 대부분 0에 가깝다고 하자.  N개의 examples 와 C개의 Class가 있을 때, loss $L_i$ 는 무엇인가?
	- 답: C-1
	- $L_i = \sum_{j \neq y_i} \text{max} (0, 0 - 0 + 1) = 1 *(C-1)$
		- $j \neq y_i$ 개 만큼을 합산하므로 C-1개
- Q4. 만약 모든 클래스에 대하여 합산 하면?
	- 불필요한 손실을 추가하여 최적화 문제를 왜곡.
- Q5. 합 대신 평균을 취한다면?
	- 수치만 줄어듬
- Q6. max에 대한 제곱을 취한다면?
	- 알 수 없는 결과를 초래

### Softmax classifier
- 분류 점수를 확률로 해석하기

$$
\begin{gather}
s = f(x_i; W)
\\ P(Y=k \vert X = x_i) = \frac{e^{s_k}}{\sum_j e^{s_j}} \quad \text{: Softmax function}
\end{gather}
$$

- 여기서 exp 연산은 s 값을 양수화 하고, 이를 합산한 값으로 나누어 normalize
- 이렇게 연산 된 확률 값을 실제 label과 비교함
	- Kullback-Leibler divergence 
		- $D_{KL}(P \Vert Q) = \sum_y P(y) log\,\frac{P(y)}{Q(y)}$
	- Cross Entropy
		- $H(P,Q) = H(P) + D_{KL}(P\Vert Q)$
- 목표: 정답 클래스 확률의 최대화
	- $L_i = -log\,P(Y=y_i \vert X=x_i)$

$$
\therefore L_i = -log \left( \frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right)
$$

- Q1. softmax Loss  $L_i$ 값의 min, max 는?
	- 확률 값이 0~1 사이 이므로, - log(확률) -> min:0, max:무한대
- Q2. 모든 $S_j$ 의 값이 근사적으로 동일하다면, C개의 클래스의 softmax loss $L_i$ 값은?
	- $-log \frac 1 C$

### Regularization

$$
L(W) = \underbrace{\frac 1 N \sum^N_{i=1} L_i(f(x_i, W), y_i)}_{\text{Data loss}} + \underbrace{\lambda R(W)}_{\text{Regularization}}
$$

- Data loss : 모델의 예측이 학습 데이터와 동일해야 함
- Regularization : 모델이 학습 데이터에 너무 잘 동작하는 것을 방지
- 즉 overfitting을 방지

