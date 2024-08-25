---
layout: post
title: DGM(CS236) Lec04. Maximum Likelihood Learning
subtitle: Maximum Likelihood Learning
date: 2024-08-23 15:14:00 +0900
category: content
tags:
  - vision
use_math: true
---

아래 내용은 Stanford의 CS236(Deep Generative Model) 강의를 듣고 필기한 노트입니다.

### Goal of learning

- 학습의 목표는 데이터를 샘플링할 $P_{data}$를 정확하게 표현하는 $P_\theta$ 를 얻어내는 것
  - Rough approximation 과 computational reasons로 성공하기 어려움.
  - 예를들어 28x28 이미지는 $2^{784}$ 만큼의 state가 존재.
  - $10^7$ 개의 data라 하더라도, sparse한 coverage
- $P_{data}$ 분포를 구성할 수 있는 최적의 approximation을 선택 하는 것이 목표.

### What is "best"

- Density estimation: 전체 분포에 관심
- Specific prediction task: prediction을 하기 위한 분포
  - descriminated model
  - structred prediction: 다음 비디오 프레임 예측, 이미지 캡션
    - 이미지와 캡션간 joint distribution 자체를 학습할 필요 없음
    - conditional distribution 학습! (주어진 조건에 대한 확률 분포!)

### Joint distribution과 Conditional distribution의 차이

- joint distribution(결합 분포)
  - 두 개 이상의 확률 변수들이 함께 발생할 확률을 나타냅니다.
  - $P(X=x, Y=y)$
- conditional distribution(조건부 분포)
  - 하나의 확률 변수가 주어진 상황에서 다른 확률 변수의 분포
  - $P(X=x | Y=y)$
- 조건부 분포는 결합 분포와 주변 분포(marginal distribution)로부터 유도
  - $P(X=x∣Y=y)=\frac{P(X=x,Y=y)​}{P(Y=y)}$
  - 여기서 $P(Y=y)$가 marginal distribution

### KL-divergence

- 두 분포 간의 거리를 측정하는 방식
- p와 q 간의 Kullback-Leibler divergence(KL-divergence)는 아래와 같이 정의
- $D(p||q) = \sum_x{p(x)log\frac{p(x)}{q(x)}}$
- 모든 $p,q$에 대해 $D(p||q) > 0$
- $p, q$가 같은 분포일 때 $D(p||q) = 0$
- $p(x)$ 를 대신하여 $q(x)$를 사용하여 압축하였을 때, 얼마나 많은 extra bits가 필요한가를 측정.
- 예시
  - 100번 수행하는 unbiased coin 던지기면, 1bit로 최대 압축 (heads 0, tails 1)
  - biased coin이고, $P[H] >> P[T]$ 이면, head에 대한 표현 bit가 tail보다 적은 bit여야 더 효율적
  - Morse code 와 같음
- KL-divergence의 정의
  - if your data comes from p, but you use a scheme optimized for q, the divergence $D_{KL}(p||q)$ is the number of extra bits you'll need on average.

### Log-likelihood 관계

- $D(P_{data}||P_\theta) = \sum_{x}{P_{data}(x)log\frac{P_{data}(x)}{P_{\theta}(x)}}$ 를 expectation 관점에서 다시 써보면,
- $D(P_{data}||P_\theta) = E_x \sim P_{data}[log(\frac{P_{data}(x)}{P_\theta(x)})]$ 로 쓸 수 있다.
  - 확률 변수 x 에 대한 기댓값 정의 : $E(x)= \sum_xP(x)*f(x)$
  - 즉 True data distribution과 model의 확률의 비율에 대한 평균
- 위 식을 다시 작성해보면,
  - $D(P_{data}||P_\theta) = E_x \sim P_{data}[log(\frac{P_{data}(x)}{P_\theta(x)})] = E_x \sim P_{data}[log{P_{data}(x)}] - E_x \sim P_{data}[log{P_{\theta}(x)}]$
  - 여기서 첫 번째 term은 $P_\theta$와 독립이므로, 두 번째 term을 최대화 하는 것이 목표(앞에 마이너스가 붙으므로, KL-divergence를 최소화 시키려면 최대화 시켜야함.)
- 즉, $\underset{P_{\theta}}{argmin}\,D(P_{data}||P_\theta) =\underset{P_{\theta}}{argmin}\, - E_x \sim P_{data}[log{P_{\theta}(x)}] = \underset{P_{\theta}}{argmax}\,E_x \sim P_{data}[log{P_{\theta}(x)}]$
- 그런데 일반적으로 $P_{data}$ 를 알 수 없어서, 실제로 $P_{data}$와 얼마나 가까운지 알 수 없고, 여러 파라미터 $\theta$ 로 만든 모델 간 비교만 할 수 있음.

### Monte Carlo Estimation

- E(x)를 샘플로부터 추정 하는 방법.
- 관심있는 값(The quantity of interest)을 기댓값으로 표현하면,
  - $E_x \sim P[g(x)] = \sum_xg(x)P(x)$
- P 분포로부터, T개의 sample을 $x^1, ... , x^T$ 을 생성하고, 이를 가지고 $g(x)$ 평균을 계산하면,
  - $\hat{g}(x^1,...,x^T) \triangleq \frac{1}{T}\sum_{t=1}^{T}g(x^t)$
- $E_p[\hat{g}] = E_x \sim P[g(x)]$
- T가 무수히 커지면 수렴.
