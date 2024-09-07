---
layout: post
title: DGM(CS236) Lec15. Evaluating Generative models
subtitle: Evaluating Generative models
date: 2024-09-07 22:43:00 +0900
category: content
tags:
  - vision
use_math: true
---

아래 내용은 Stanford의 CS236(Deep Generative Model) 강의를 듣고 필기 한 노트입니다.

### Inception Score
- 가정1: 레이블이 있는 데이터셋으로부터 학습된 모델을 평가
- 가정2: 어떤 데이터 포인트 x로부터 레이블 y를 예측할 수 있는 좋은 분류기가 있어야함.
- 위 두 가정을 만족할 때, **Sharpness** 와 **Diversity** 를 가지고 평가.
- Sharpness
	- $c(y\vert x)$ 는 분류기의 예측값
	- $log\,c(y\vert x)$는 분류기의 엔트로피
	- 높은 sharpness value는 생성된 이미지에 대해 잘 예측했음을 의미

$$
S = exp(E_{x\sim p}\left[ \int c(y\vert x)\,log\,c(y\vert x)dy \right])
$$
- Diversity
	- $c(y) = E_{x \sim p}\left[ c(y\vert x)\right]$ 는 marginal predictive distribution
	- 높은 diversity 는 $c(y)$ 가 높은 엔트로피를 가짐을 의미

$$
D = exp(-E_{x \sim p}\left[\int c(y\vert x) \, log\, c(y)dy \right])
$$
- Inception Score 
	- $IS = D \times S$
	- 높을 수록 더 좋은 퀄리티 의미
	- 분류가 불가능 할 때에는 큰 데이터셋에 학습된 분류기를 이용. e.g. InceptionNet with ImageNet
	- Cons
		- $p_\theta$의 샘플만 가지고 측정. $p_{data}$이 쓰이지 않음.
### FID (Frechet Inception Distance)
- $p_\theta$로 부터 생성된 Sample과 테스트 데이터셋의 Feature representation 간의 유사도를 측정.
- FID 계산
	- $G$를 생성된 샘플, $T$를 테스트 데이터 셋으로 정의
	- Feature representation $F_G$와 $F_T$ 를 계산 (e.g. Using Inception Net)
	- 각 $F_G$와 $F_T$ 에 대하여, Multivariate Gaussian 분포를 생성하고, 두 가우시안 분포의 평균과 분산을 $(\mu_G, \Sigma_G)$, $(\mu_T, \Sigma_T)$ 라고 한다.
	- FID는 두 가우시안 분포간의 **Wasserstein-2 거리**로 정의한다.
	- FID가 낮을 수록 좋은 퀄리티를 뜻한다.

$$
FID = \Vert \mu_T - \mu_G \Vert^2 + Tr\left(\Sigma_T + \Sigma_G - 2 \sqrt{(\Sigma_T \Sigma_g)} \right)
$$

