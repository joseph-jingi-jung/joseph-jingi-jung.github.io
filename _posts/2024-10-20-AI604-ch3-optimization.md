---
layout: post
title: AI604 - CH3 Optimization
subtitle: optimization
date: 2024-10-20 22:58:01 +0900
category: content
tags:
  - vision
use_math: true
---

AI604 수업을 수강 후 정리한 내용이다. Stanford의 CS231n과 맞닿아 있다.

### 전략 1 : Random search
- 랜덤하게 Weight를 초기화
- Bad

### 전략 2 : Follow the slope
- 다차원에서, 그라디언트(gradient)는 각 차원의 편미분 값(paritial derivatives) 으로 구성된 벡터이다.
- 어느 방향에서든 기울기(Slope) 는 그 방향과 그래디언트의 내적이다.
- 가장 가파르게 하강하는 방향은 그래디언트의 음수 방향이다.
- Loss가 W에 대한 함수이므로, 우리는 $\nabla_W L$ 을 구하길 원한다.
	- Analytic gradient를 구함 (수학적 미분을 통해 얻은 값)
### Gradient Descent
- Hyper-parameter
	- Weight 초기 값
	- 스탭 수
	- Learning rate (step-size)
#### Stochastic Gradient Descent(SGD)
- 전체 데이터 셋에 대한 Gradient를 구하기 어려움
- Mini-batch 를 이용한 근사적인 합산을 통해 Loss와 Loss에 대한 Gradient를 구함
- Hyper-parameter
	- Weight 초기 값, 스탭 수, Learing rate
	- Batch size, Data sampling -> 둘 다 중요한 하이퍼 파라미터
- SGD 문제점 1
	- 한쪽 방향으로는 급격히 변화하고, 다른 방향으로 천천히 변화한다면?
		- 완만한 차원에서는 매우 느린 진행을 보이고, 가파른 방향에서는 흔들림이 발생함
- SGD 문제점 2
	- Local minia 혹은 saddle point 가 있다면?
		- 해당 위치에서 그라디언트가 0가 되어, 멈춰버림
		- 고차원에서 saddle point가 흔함
- SGD 문제점 3
	- 미니배치를 이용하여 노이즈가 발생할 수 있음
		- 미니 배치를 사용하여 진정한 기울기와 다를 수 있으므로, 샘플에 포함된 데이터에 따라 기울기가 불안정하거나 편향 될 수 있음. 이로 인한 노이즈가 발생.
#### SGD + Momentum
- 이전 반복과 같은 일반적인 방향으로 계속 이동하기.
	- $v_{t+1} = \rho v_t + \nabla f(x_t)$
	- $x_{t+1} = x_t - \alpha v_{t+1}$
- 진행 중인 그라디언트의 평균을 velocity로 구성
- $\rho$ 는 마찰(friction)을 의미. 이전 기울기 방향을 얼마나 유지할지 조정. 보통 0.9 또는 0.99 이용.
- 여기서 추가된 Velocity 는 local minima 혹은 saddple point 탈출에 도움을 줌.
#### RMSProp
- 각 차원에서 Historical 제곱합(decay 를 포함)을 이용하여, 그레디언트에 요소별 스케일링을 추가
	- $g_t = \gamma\,g_{t-1} + (1 - \gamma)(\nabla f(x_{t-1}))^2$
	- $x_t = x_{t-1} - \frac{\alpha}{\sqrt{g_t + \epsilon}}\cdot\nabla f(x_{t-1})$
- $g_t$ 는 기울기 누적 크기
- $\gamma$ 지수 이동 평균의 업데이트 계수
- $\epsilon$ 분모가 0이 되는 것을 방지 하기 위한 아주 작은 값 $\approx 10^{-7}$
- $\alpha$ 학습률
- Per-parameter learning rates 혹은 adaptive learning rates 라고 부름
- RMSProp에서 무슨 일이 일어나는가?
	- 가파른 방향에서의 억제, 평탄한 방향에서의 가속
		- 제곱합을 기준으로 기울기를 나누어 스케일링 하기 때문
#### ADAM (almost) as RMSProp + Momentum
- RMSProp 과 Momentum 같이 사용. 초기 값이 작을 때를 보정하기 위한 bias correction 추가
	- $m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla f(x_{t-1})$ : Momentum
	- $g_t = \beta_2 g_{t-1} + (1-\beta_2)(\nabla f(x_{t-1}))^2$ : RMSProp의 제곱합 연산
	- $\hat{m_t} = \frac{m_t}{(1-\beta_1^t)}$, $\hat{g_t} = \frac{g_t}{(1-\beta_2^t)}$ : Bias correction(초기의 $m_t, g_t$ 가 거의 0 이므로, 값을 키워줌)
	- $x_t=x_{t-1} - \frac{\alpha}{\sqrt{\hat{g_t} +\epsilon}} \cdot \hat{m_t}$ : RSMProp

#### Second-order optimization
- First-order optimization 과 달리 이차 미분 행렬을 사용함에 따라 더 많은 정보를 활용
	- 곡률 정보 반영
		- 이차 미분 행렬(Hassian matrix) 는 곡률 정보를 제공하고, 이는 방향 뿐 아니라 그 방향에 얼마나 빠르게 변하는지 까지 나타냄.
		- 따라서 곡률에 따라 스탭 크기를 조정하여 안정적인 수렴이 가능함
- 그러나 이차 미분에 대한 연산 비용이 커서 Deep learning에서 잘 사용되지 않음


