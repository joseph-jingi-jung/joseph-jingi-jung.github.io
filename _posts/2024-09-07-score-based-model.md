---
layout: post
title: DGM(CS236) Lec13~14. score-based models
subtitle: Score-Based Models
date: 2024-09-07 17:13:00 +0900
category: content
tags:
  - vision
use_math: true
---

아래 내용은 Stanford의 CS236(Deep Generative Model) 강의를 듣고 필기 한 노트입니다.

### Recap
#### How to represent probability distribution?
1. P.D.F 또는 P.M.F 를 이용 $p(x)$
	- Autoregressive models, Flow models, Variational autoencoders, Energy-based models
	- Pros
		- MLE를 이용하여 학습
		- Likelihood를 이용하여, 모델간 비교 가능
	- Cons
		- 계산이 불가능한(interactable) partition functions 을 해결하기 위해 특별한 Loss나 아키텍쳐를 구성해야함.
2. Sampling process
	- Generative adversarial network (GANs) (Implicit generative model)
	- Pros
		- 더 나은 생성 결과물
	- Cons
		- Adversarial 학습이 필요함. (학습이 불안정하고 mode collapse에 빠지기도 함)
		- 모델 간 비교를 위한 주요한 없음
		- 학습을 중단시킬 명확한 기준이 없음
3. P.D.F가 미분 가능한 경우, Probability density의 gradiant를 이용
	- Score function, Energy based model

### Recap on energy-based models
- Deep energy-based models (EMBs)
	- $f_\theta(x) \in \mathbb{R}$ , $p_\theta(x) = \frac{e^{f_\theta(x)}}{Z(\theta)}$
- Maximum likelihood training : $\underset{\theta}{max} f_\theta(x_{train}) - log\, Z(\theta)$
	- 여기서 $Z(\theta)$ 알기 어려움
	- **Contrative divergence**
		- $\nabla_\theta f_\theta(x_{train}) - \nabla_\theta log\,Z(\theta) \approx \nabla_\theta f_\theta(x_{train}) - \nabla_\theta f_\theta(x_{sample}), \, x_{sample} \sim p_\theta(x)$
		- 학습 동안 iterative sampling이 필요함. ($x_{sample}$)
	- Minizing Fisher divergence  : $\underset{\theta}{min}\frac 1 2 E_{x~\sim p_{data}}\left[\Vert \nabla_x log\, p_{data}(x) - \nabla log\, p_\theta(x)  \Vert^2_2\right]$
		- **Score matching**

$$
\begin{aligned}
&\frac 1 2 E_{x~\sim p_{data}}\left[\Vert \nabla_x log\, p_{data}(x) - \nabla log\, p_\theta(x)  \Vert^2_2\right]
\\ =& \frac 1 2 E_{x~\sim p_{data}}\left[\Vert \nabla_x log\, p_{\theta}(x))  \Vert^2_2 + tr(\nabla^2_x log\,p_\theta(x))\right] + const
\end{aligned}
$$

### Score estimation by training score-based models
- 스코어 베이스 모델은 직접적으로 score function을 모델링 한다.
	- 에너지나 likelihood를 모델링 하는것이 아니라, 모델은 하나의 vector valued function 이나 여러 vector valued function의 set이 된다. 
	- 모델이 업데이트 되면 다른 vector field를 가지게 된다.
- Given : i.i.d samples ${x_1, x_2, ..., x_n} \sim p_{data}(x)$
- Task : Estimating the score $\nabla_x log\, p_{data}(x)$
- Score model : A learnable vector-valued function $s_\theta(x) : \mathbb{R}^d \rightarrow \mathbb{R}^d$
- Goal : $s_\theta(x) \approx \nabla_x\,log\,p_{data}(x)$
- 유클리디안 거리로 두 벡터의 스코어 비교. (Average Euclidean distance over the space)
	- Fisher divergence: $\frac 1 2 E_{x~\sim p_{data}}\left[\Vert \nabla_x log\, p_{data}(x) - s_\theta(x) \Vert^2_2\right]$
	- 여기서 $\nabla_x log\, p_{data}(x)$ 는 알 수 없으므로, 
	- Score matching:
		- $E_{x~\sim p_{data}}\left[\frac 1 2 \Vert s_\theta(x) \Vert^2_2 + tr(\,\,\nabla_x s_\theta(x)\,\,)\right]$
		- $\nabla_xs_\theta(x)$ 는 Jacobian of $s_\theta(x)$
- 그러나 Score matching 은 scalable 하지 않다.
	- $tr(\nabla_xs_\theta(x))$ 즉 trace of Jacobian $s_\theta(x)$ 는 o(d) 의 역전파로 차원에 선형적으로 scale하다.

### Denoising Score Matching
- 노이즈가 약간 추가된 데이터의 그라디언트를 추정해보자!
- Perturbed distribution
	- $q_\sigma(\tilde{x} \vert x) = N(x; \sigma^2 I)$
	- $q_\sigma(\tilde{x}) = \int p(x) q_\sigma(\tilde{x} \vert x) dx$
- 여기서 score의 추정 $\nabla_{\tilde{x}} log \, q_\sigma(\tilde{x})$ 는 훨씬 쉽다.
- 노이즈의 정도가 작다면, 좋은 근사가 된다 $q_\sigma(\tilde{x}) \approx p(\tilde{x})$
- **Estimate the score of a noise-perturbed distribution**

$$
\begin{aligned}
&\frac 1 2 E_{\tilde{x}\sim p_{data}} \left[ \Vert s_\theta(\tilde{x}) - \nabla_{\tilde{x}} log\, q_\sigma(\tilde{x}) \Vert_2^2 \right]
\\=& \frac 1 2 E_{\tilde{x}\sim p_{data}, \tilde{x} \sim q_\sigma(\tilde{x}\vert x)} \left[ \Vert s_\theta(\tilde{x}) - \nabla_{\tilde{x}} log\, q_\sigma(\tilde{x}|x)\Vert_2^2 \right]
\end{aligned}
$$

- 여기서 $\nabla_{\tilde{x}} log\, q_\sigma(\tilde{x}|x)$ 는 가우시안 노이즈인 경우 연산이 쉽다.
	- $q_\sigma(\tilde{x}|x) = N(\tilde{x}|x, \sigma^2 I)$
	- $\nabla_{\tilde{x}} log\, q_\sigma(\tilde{x}|x) = -\frac{\tilde{x}-x}{\sigma^2}$
- Pros
	- 고차원 데이터에 대하여도 최적화가 매우 효율적이다
- Cons
	- 노이즈가 없는 클린 데이터의 score를 추정하지 못한다.
- 다시 정리하면,

$$
\begin{aligned}
&\frac 1 2 E_{q_\sigma{\tilde{x}}} \left[ \Vert \nabla_{\tilde{x}} log\, q_{\sigma}(\tilde{x}) - s_\theta(\tilde{x}) \Vert^2_2 \right] \,\text{: Score matching}
\\=& \frac 1 2 E_{p(x)} E_{q_\sigma(\tilde{x}\vert x)} \left[ \Vert \frac{1}{\sigma^2}(x - \tilde{x}) - s_\theta(\tilde{x}) \Vert^2_2 \right] \, \text{: denoising}
\end{aligned}
$$
- $s_\theta(\tilde{x})$ 가 데이터에 추가된 노이즈 $(x - \tilde{x})$ 를 추정한다. (denoising)

### Sliced score matching
- 1차원의 문제가 더 풀기 쉬움을 이용.
- 랜덤한 방향으로 Projection 하여 계산.
- Objective : Sliced Fisher Divergence
	- $\frac 1 2 E_{v\sim p_v}E_{x \sim p_{data}}\left[ (v^T\nabla_x log\,p_{data}(x) - v^T s_\theta(x))^2 \right]$
	- 여기서 $v$ 는 랜덤하게 선택된 방향 v 
- Integration by parts
	- $E_{v\sim p_v}E_{x \sim p_{data}}\left[ v^T\nabla_x s_\theta(x)v + \frac 1 2 (v^T s_\theta(x))^2 \right]$
	- $\frac 1 2 (v^T s_\theta(x))^2$ 이 부분 연산은 빠름
	- $v^T\nabla_x s_\theta(x)v = v^T\nabla_x (v^T s_\theta(x))$ 이고, 이 부분이 1D가 되면서 한번의 역전파만 일어남
	- Slightly slower than denoising score matching.
	- Vector projection으로 자코비안 계산 O(d)를 O(1)로 계산!

### Score-bases model sampling
- 에너지 모델처럼 Langevin dynamics sampling 를 적용.
	- score
		- $s_\theta(x)$
	- Follow the scores 
		- $\tilde{x}_{t+1} \leftarrow \tilde{x}_t + \frac \epsilon 2 s_\theta(\tilde{x}_t)$
	- Follow noisy scores (Langevin MCMC) 
		- $Z_t \sim N(0,I)$
		- $\tilde{x}_{t+1} \leftarrow \tilde{x}_t + \frac \epsilon 2 s_\theta(\tilde{x}_t) + \sqrt{\epsilon} z_t$
### Pitfall of score based model
- Maiford hypothesis
	- 데이터가 균등 분포 되지 않고, 한정된 부분에 집중 되어있음
	- 빈 공간에 대해서 즉 $P_{data}(x)=0$ d인 지점에서 $\nabla_x log\,p_{data}(x)$ 구할 수 없음.
- 데이터 밀도가 적은 공간에서 스코어 매칭이 잘 안됨
- 매우 느리게 수렴하는 langevin dynamics

### Gaussian perturbation
- 위 함정들을 해결하는 방법은 **Gaussian perturbation** (노이즈 추가) 에 있다.
- Maiford hypothesis
	- 가우시안 노이즈는 모둔 공간에 분포하여, 해당 문제 해결
- 매우 느리게 수렴하는 langevin dynamics
	- Denoising score matching 으로 연산 속도 빨라지고, 수렴에 더 좋음.
- Low data density regions
	- Multi-scale Noise perturbation 으로 해결.
		- 큰 노이즈는 랑쥬뱅 동역학에서 유용한 방향 정보를 제공하지만 더 이상 실제 데이터 분포를 추정하지 않음. Trade off 관계

### Annealed Langevin Dynamics: Joint Scores to Samples
- 노이즈 를 $\sigma_1, \sigma_2, ..., \sigma_L$ 을 이용하여 샘플링하고, 랑주뱅 동역학에 사용한다.
- 노이즈 레벨을 낮춰간다(Anneal down) 
- 해당 노이즈를 사용한 샘플을 다음 레벨의 초기 값으로 사용.
- Annealed Langevin Dynamics의 결과는 거의 cleaned 이미지와 동일하다.

### Training noise conditional score networks
- 자연스럽게 Denoising score matching 을 이용하는게 적합. 
- 따라서 Goal은 perturbed data distribution의 score를 추정하는 것.
- 또한 각 denoising score matching losses에 weight를 추가함.

$$
\begin{aligned}
&\frac 1 L \sum_{i=1}^L \lambda(\sigma_i) E_{q_{\sigma_i}}(x)\left[ \Vert \nabla_x log\, q_{\sigma_i}(x) - s_\theta(x, \sigma_i) \Vert_2^2 \right]
\\ =& \frac 1 L \sum_{i=1}^L \lambda(\sigma_i) E_{x\sim p_{data}, z\sim N(0,I)} \left[ \Vert \nabla_{\tilde{x}} log\, q_{\sigma_i}(\tilde{x}|x) - s_\theta(\tilde{x}, \sigma_i) \Vert_2^2 \right] + const
\\ =& \frac 1 L \sum_{i=1}^L \lambda(\sigma_i) E_{x\sim p_{data}, z\sim N(0,I)} \left[ \left\Vert s_\theta(x + \sigma_i z , \sigma_i) +\frac{z}{\sigma_i} \right\Vert_2^2 \right] + const
\end{aligned}
$$

### Choosing noise scales
- 노이즈 크기를 어떻게 정할 것인가?
- 가장 큰 노이즈 스케일의 선택 ($\sigma_1$)
	- 학습 데이터에서 pairwise 거리가 가장 큰 것을 가장 큰 노이즈의 크기 ($\sigma_1$)로 간주.
- 가장 작은 노이즈 스케일의  선택 ($\sigma_L$)
	- clean image와 구분하기 힘들 정도로 작은 노이즈 스케일로 정의
- 첫번째와 마지막 사이의 스케일 선택
	- 등비수열 형태를 가질 수 있게 구성
	- 첫번째와 두번째 노이즈는 충분히 큰 overlap 을 가지게 구성하고, 두번째와 세번째는 첫번째와 두번째의 겹침 정도가 동일하게 구성(등비 수열)
	- $\sigma_1 \gt \sigma_2 \gt \sigma_3 \gt ...  \gt \sigma_{L-1} \gt \sigma_L$ 
	- $\frac{\sigma_1}{\sigma_2}=\frac{\sigma_2}{\sigma_3}=\,...\,=\frac{\sigma_{L-1}}{\sigma_L}$

### Choosing the weighting function
- Weighted combination of denoising score matching losses
	- $\frac 1 L \sum_{i=1}^L \lambda(\sigma_i) E_{x\sim p_{data}, z\sim N(0,I)} \left[ \left\Vert s_\theta(x + \sigma_i z , \sigma_i) +\frac{z}{\sigma_i} \right\Vert_2^2 \right] + const$ 
- 각기 다른 score matching Loss를 동일하게 만들 필요가 있음 $\rightarrow \lambda(\sigma_i) = {\sigma_i}^2$

$$
\begin{aligned}
&\frac 1 L \sum_{i=1}^L {\sigma_i}^2 E_{x\sim p_{data}, z\sim N(0,I)} \left[ \left\Vert s_\theta(x + \sigma_i z , \sigma_i) +\frac{z}{\sigma_i} \right\Vert_2^2 \right]
\\=& \frac 1 L \sum_{i=1}^L E_{x\sim p_{data}, z\sim N(0,I)} \left[ \left\Vert {\sigma_i} s_\theta(x + \sigma_i z , \sigma_i) + z \right\Vert_2^2 \right]
\\=& \frac 1 L \sum_{i=1}^L E_{x\sim p_{data}, z\sim N(0,I)} \left[ \left\Vert \epsilon_\theta(x + \sigma_i z , \sigma_i) + z \right\Vert_2^2 \right]\,\,\,\, [\epsilon_\theta(\cdot,\sigma_i) := \sigma_i s_\theta(\cdot, \sigma_i)]
\end{aligned}
$$

### Training noise conditional score networks (NCSN)
- Sample a mini-batch of datapoints $\{x_1, x_2, ..., x_n\} \sim p_{data}$
- Sample a mini-batch of noise scale indices $\{i_i, i_2, ... , i_n\} \sim U\{1, 2, ... , L\}$
- Sample a mini-batch of Gaussian noise $\{z_1, z_2, ..., z_n\} \sim N(0, I)$
- Estimate the weighted mixture of score matching losses
	- $\frac 1 n \sum_{k=1}^n \left[ \left\Vert {\sigma_i}_k s_\theta(x_k + {\sigma_i}_k z_k , {\sigma_i}_k) + z_k \right\Vert_2^2 \right]$
- Stochastic gradient descent

아래 내용들은 다른 영상에서 참고하여 정리
> 참고:
>    1. [PR-400: Score-based Generative Modeling Through Stochastic Differential Equations (youtube.com)](https://www.youtube.com/watch?v=uG2ceFnUeQU)
>    2. [[Open DMQA Seminar] Score-Based Generative Models and Diffusion Models (youtube.com)](https://www.youtube.com/watch?v=d_x92vpIWFM&t=2448s)

### Score-based generative modeling via SDEs
- **노이즈 Scale이 discrete 했는데, 이걸 Inifinite 하게 나눠서 처리 할 수 없을까? -> SDE로 이어짐**
- SDE로 변환 했을 때의 장점
	- High-quality samples
	- **Flexible (+ fast) sampling & exact log-likelihood computation**
	- Controllable generation for inverse prolbem solving
	- Unified framework
### Stochastic Differential Equations (확률미분방정식)
- Forward SDE : $dx=f(x,t)dt + g(t)dw$
- Reverse SDE : $dx = \left[f(x,t) - g^2(t) \nabla_x\,log\,p_t(x) \right]dt + g(t)dw$
- $dx$ : 아주 작은 x의 변화량
- $dt$ : 시간 변화율
- $f(x,t)dt$ : Drift(Deterministic) , 추세선과 같음
- $g(t)$ : diffusion (stochastic vector), Randomness의 추가
- $dw$ : 브라우니언(brownian) 모션, 가우시안 노이즈
	- Brownian motion의 특성
		- Z(0) : Stochastic Process의 출발 위치는 0이다.
		- Z(t)-Z(0)~N(0,t) : Stochastic Process의 0 시점부터 t시점까지 변화된 양은 평균이 0이고, 분산이 t인 정규 분포를 따른다
		- 각 구간이 서로 겹치지 않으면 각 구간마다 변화된 Stochastic Process의 양은 서로 독립이다.
		- z(t)는 연속형이지만 거의 모든 점에서 미분 불가능하다.
- **SDE는 solver가 존재하고, Reverse SDE가 항상 존재함.**
- SDE를 ODE solver로 근사하여, 다양하게 활용(추후 정리)
