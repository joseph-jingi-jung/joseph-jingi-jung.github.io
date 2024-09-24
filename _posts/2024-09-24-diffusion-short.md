---
layout: post
title: Understanding Diffusion Models (short version)
subtitle: DDPM 
date: 2024-09-24 22:56:00 +0900
category: content
tags:
  - vision
use_math: true
---

복습 차원에서 DDPM 전반 부분을 다시 한번 정리하였다. 
식 전개 부분을 적당히 생략하고, 큰 흐름 위주로 다시 작성하였다.

### Variational Diffusion Model
Variational Diffusion Model(VDM)을 가장 쉽게 표현 할 수 있는 방법은 아래 3가지 제약이 있는 Markovian Hierachical Variation Autoencoder 이다.
- Latent 의 차원은 데이터의 차원과 동일하다.
- 각 타임스탭의 latent encoder는 학습하는 것이 아니라(Forward Process), 정해진 선형 가우시간 모델을 사용한다. 즉 이전 타임스탭의 출력 중심의 가우시안 분포이다.
- Latent 인코더의 가우시안 파라미터는 시간에 따라 변화하며, 최종 시간 T에서 latent의 분포는 표준 가우시안이 된다.

따라서,
- $x_0$ : true data sample
- $q(x_{1:T}\vert x_0) = \prod^T_{t=1} q(x_t \vert x_{t-1})$

Foward Process에서 각 잠재변수의 분포가, 이전 계층의 잠재 변수를 중심으로 하는 가우시안 분포 라는 것을 알고 있으므로 파라미터화 하면 아래와 같다.

$$
q_(x_t\vert x_{t-1}) = N(x_t; \underbrace{\sqrt{\alpha_t}x_{t-1}}_{\mu_t(x_t)} , \underbrace{(1-\alpha_t)I}_{\Sigma_t(x_t)})
$$

위 세번째 정의에 의해, 최종 시간 T에서의 latent 분포는 아래와 같다.
 
 $$
 p(x_{0:T}) = p(x_T)\prod^T_{t=1}p_\theta(x_{t-1}\vert x_t) \text{ where } p_{(x_T)} = N(x_T;0, I)
 $$
### VDM의 ELBO
 
$$
\begin{aligned}
log\,p(x)&=log \int p(x_{0:T})dx_{1:T}
\\&= log \int \frac{p(x_{0:T}) q_\phi(x_{1:T}\vert x_0)}{q_\phi(x_{1:T}\vert x_0)} dx_{1:T}
\\&= log\,\mathbb{E}_{q_\phi(x_{1:T}\vert x_0)} \left[\frac{p(x_{0:T})}{q_\phi(x_{1:T}\vert x_0)} \right]
\\&\ge \mathbb{E}_{q_\phi(x_{1:T}\vert x_0)} \left[log\,\frac{p(x_{0:T})}{q_\phi(x_{1:T}\vert x_0)} \right]
\\&= \mathbb{E}_{q_\phi(x_{1:T}\vert x_0)} \left[log \frac{p(x_T)\prod^T_{t=1}p_\theta(x_{t-1}\vert x_t)}{\prod^T_{t=1}q(x_t \vert x_{t-1})} \right]
\\&= \underbrace{\mathbb{E}_{q_\phi(x_1\vert x_0)} \left[log p_\theta(x_0|x_1) \right]}_{\text{reconstruction term}} - \underbrace{\mathbb{E}_{q(x_{T-1}\vert x_0)}\left[ D_{KL}(q(x_T \vert x_{T-1}) \Vert p(x_T)) \right]}_{\text{prior matching term}}  
\\ &\quad \quad - \sum^{T-1}_{t=1} \underbrace{\mathbb{E}_{q_\phi(x_{t-1}, x_{t+1}\vert x_0)} 
\left[ D_{KL} (q(x_t \vert x_{t-1}) \Vert p_\theta(x_t \vert x_{t+1}))\right]}_{\text{consistency term}} 
\end{aligned}
$$

### Reconstruction term
첫번째 스탭의 latent로 오리지널 데이터를 예측 하는 로그 likelihood 이다. 이는 Vanilla VAE와 유사하게 학습 할 수 있으나 생략한다.
### Prior matching term
마지막 latent 분포가 가우시안 Prior과 동일할 때 최소화. 학습 파라미터 없으므로 최적화 불필요
### Consistency term
$x_t$ ​에서의 분포를 **정방향과 역방향 과정 모두에서 일관**되게 만들기 위해 노력함. 
각 매칭되는 타임 스텝에서 디노이징 스탭과 노이징 스탭의 동일하게 매칭 됨을 의미함. 
이 텀은 $p_\theta(x_t \vert x_{t+1})$ 이 가우시안 분포 $q(x_t\vert x_{t-1})$과 동일하게 학습 되었을 때, 최소화.

이 텀을 최적화 할 때, 매 시점에 두 확률 변수 $x_{t-1}, x_{t+1}$ 에 대한 기대 값을 계산해야하여, 몬테카를로 추정의 분산이 높아질 수 있음.

마르코비안 성질을 이용하여, $q(x_t \vert x_{t-1}) = q(x_t \vert x_{t-1}, x_0)$ 로  표현 하여 다시 ELBO 정리

###  $q(x_t \vert x_{t-1}) = q(x_t \vert x_{t-1}, x_0)$을 이용한 ELBO

$$
\begin{aligned}
log\,p(x) &\ge \mathbb{E}_{q(x_{1:T}\vert x_0)} \left[log\,\frac{p(x_{0:T})}{q(x_{1:T}\vert x_0)} \right]
\\&= \underbrace{\mathbb{E}_{q(x_1\vert x_0)} \left[log\, p_\theta(x_0 \vert x_1) \right]}_{\text{reconstruction term}} -
\underbrace{D_{KL}(q(x_T\vert x_0) \Vert p(x_T))}_{\text{prior matching term}} 
\\ & \quad \quad-
\sum^T_{t=2}
\underbrace{\mathbb{E}_{q(x_t \vert x_0)}\left[D_{KL}(q(x_{t-1}\vert x_t, x_0) \Vert p_\theta(x_{t-1}\vert x_t)) \right]}_{\text{denoising matching term}}
\end{aligned}
$$

### Denoising matching term
$p_\theta(x_{t-1} \vert x_t)$ : 현재 노이즈 이미지에서 한 단계 전 이미지로의 전이 확률
$q(x_{t-1}\vert x_t, x_0)$ : 노이즈가 있는 이미지($x_t$)에서 이미 최종 결과($x_0$)를 알고 있을 때 그 전 이미지($x_{t-1}$) 로의 전이 확률
이 둘의 KL 발산을 최소화함으로써 두 전이 확률이 최대한 일치되게 학습.

### Reparameterization trick
Denoising matching term에 p로 샘플링하는 부분이 포함되어있어, 최적화에 어려움이 있다. 
VAE와 동일하게 Reparameterization trick 적용이 필요하다.

그 전에 앞서, $x_t \sim q(x_t \vert x_{t-1})$ 은 아래와 같이 다시 작성 할 수 있다.

$$
\begin{aligned}
q_(x_t\vert x_{t-1}) &= N(x_t; \sqrt{\alpha_t}x_{t-1} , (1-\alpha_t)I) \text{ 이므로, }
\\x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t}\epsilon \quad with \,\, \epsilon \sim N(\epsilon; 0, I)
\\x_{t-1} &= \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon \quad with \,\, \epsilon \sim N(\epsilon; 0, I) 
\end{aligned}
$$

$q(x_t \vert x_0)$ 는 Reparameterization trick을 재귀적으로 적용하여, 유도할 수 있다. 

$$
\begin{aligned}
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1}
\\&= \sqrt{\alpha_t} (\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}) + \sqrt{1-\alpha_t}\epsilon_{t-1}
\\&=\,...
\\&=\sqrt{\prod^t_{i=1}\alpha_i}\,x_0 + \sqrt{1 - \prod^t_{i=1}\alpha_i}\, \epsilon_0
\\&=\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon_0
\\&\sim N\,(x_t; \sqrt{\bar{\alpha}_t}\, x_0, \sqrt{1 - \bar{\alpha}_t}\, I)
\end{aligned}
$$
정리하면,

$$
\begin{aligned}
q_(x_t\vert x_{t-1}) &= N(x_t; \sqrt{\alpha_t}x_{t-1} , (1-\alpha_t)I)
\\x_t \sim q(x_t \vert x_{t-1}) &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t}\epsilon \quad with \,\, \epsilon \sim N(\epsilon; 0, I)
\\ x_t \sim q(x_t \vert x_0) &= (x_t; \sqrt{\bar{\alpha}_t}\, x_0, \sqrt{1 - \bar{\alpha}_t}\, I)
\end{aligned}
$$

$q(x_t\vert x_0)$와 $q(x_{t-1}\vert x_0)$ 을 유도하였으므로, Denoising matching term의 $q(x_{t-1} \vert x_t, x_0)$를 Bayes rule 을 이용해서 유도해보자.

$$
\begin{aligned}
q(x_{t-1} \vert x_t, x_0) &= \frac{q(x_t \vert x_{t-1}, x_0) q(x_{t-1} \vert x_0)}{ q(x_t \vert x_0)} \text{ (by Bayes rule)}
\\&= \frac
{N(x_t; \sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I) \, N\,(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}\, x_0, \sqrt{1 - \bar{\alpha}_{t-1}}\, I)}
{N\,(x_t; \sqrt{\bar{\alpha}_t}\, x_0, \sqrt{1 - \bar{\alpha}_t}\, I)}
\\& \propto exp \left\{ 
-\left[
\frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{2(1-\alpha_t)}
+ \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0)^2}{2(1-\bar{\alpha}_{t-1})}
- \frac{(x_{t} - \sqrt{\bar{\alpha}_{t}} x_0)^2}{2(1-\bar{\alpha}_{t})}
\right]
\right\}
\\& = exp \left\{ 
- \frac 1 2 \left[
\frac{(-2\sqrt{\alpha_t} x_t x_{t-1} + \alpha_t x^2_{t-1}) }{1-\alpha_t}
+ \frac{(x^2_{t-1} - 2\sqrt{\bar{\alpha}_{t-1}}x_{t-1}x_0)}{1-\bar{\alpha}_{t-1}}
+ C(x_t, x_0)
\right]
\right\}
\\&\,(\,q(x_{t-1} \vert x_t, x_0)\text{는 주어진} x_t\text{와 } x_0 \text{에 대하여,  }x_{t-1} \text{을 계산 하므로, } x_t, x_0 \text{만 포함된 항은 상수 항으로 간주} )
\\& \propto exp \left\{ 
- \frac 1 2 \left[
- \frac{2\sqrt{\alpha_t} x_t x_{t-1} }{1-\alpha_t}
+ \frac{\alpha_t x^2_{t-1}}{1-\alpha_t}
+ \frac{x^2_{t-1}}{1-\bar{\alpha}_{t-1}}
- \frac{2\sqrt{\bar{\alpha}_{t-1}}x_{t-1}x_0}{1-\bar{\alpha}_{t-1}}
\right]
\right\}

\\& = exp \left\{ 
- \frac 1 2 \left(\frac{1-\bar{\alpha}_t}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})} \right) \left[
 x^2_{t-1}
- 2 \frac{ 
\sqrt{\alpha_t} (1-\bar{\alpha}_{t-1}) x_t + \sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t) x_0}
{\left(1-\bar{\alpha}_t \right)} x_{t-1}
\right]
\right\}
\\& \propto exp \left\{ 
- \frac 1 2 \left(\frac{1}{\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}} \right) \left[\left(
 x_{t-1}
- \frac{ 
\sqrt{\alpha_t} (1-\bar{\alpha}_{t-1}) x_t + \sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t) x_0}
{\left(1-\bar{\alpha}_t \right)}
\right)^2\right]
\right\}
\\&= N(x_{t-1}; \underbrace{\frac{ 
\sqrt{\alpha_t} (1-\bar{\alpha}_{t-1}) x_t + \sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t) x_0}
{\left(1-\bar{\alpha}_t \right)}}_{\mu_q(x_t, x_0)}, \underbrace{\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} I}_{\Sigma_q(t)}) \quad \text{( by 정규분포 p.d.f)}
\end{aligned}
$$

여기서 중요한 부분은 결국 $q(x_{t-1} \vert x_t, x_0)$ 를 정규 분포로 근사 한 것이다.
**즉, $x_{t-1} \sim q(x_{t-1}\vert x_t, x_0)$ 이 정규분포를 따르고, 그 평균이 $x_t, x_0$로 구성된 함수 $\mu(x_t, x_q)$ 이고, 분산이 $\alpha$로 구성된 $\Sigma_q(t)$ 임을 알 수 있다.**

$$
q(x_{t-1} \vert x_t, x_0) = N(x_{t-1}; \underbrace{\frac{ 
\sqrt{\alpha_t} (1-\bar{\alpha}_{t-1}) x_t + \sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t) x_0}
{\left(1-\bar{\alpha}_t \right)}}_{\mu_q(x_t, x_0)}, \underbrace{\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} I}_{\Sigma_q(t)})
$$

### ELBO의 $D_{KL}(q(x_{t-1}\vert x_t, x_0) \Vert p_\theta(x_{t-1}\vert x_t))$ 의 연산
위에서 ground-truth $q(x_{t-1}\vert x_t, x_0)$이 정규분포를 따름을 확인하였고, $p_\theta(x_{t-1}\vert x_t)$ 이 그와 가장 가까워져야 한다.
여기서 $p_\theta(x_{t-1}\vert x_t)$ 를 정규 분포로 가정하며, GT의 분산이 시간에 따라 고정되어있기 때문에, 동일한 분산으로 가정한다.
따라서 KL Divergence를 다시 작성해보면 아래와 같다.

$$
\begin{aligned}
&\underset{\theta}{argmin}\,D_{KL}\,(q(x_{t-1}\vert x_t, x_0) \Vert p_\theta(x_{t-1}\vert x_t))
\\ &= \underset{\theta}{argmin}\,D_{KL} (N(x_{t-1}; \mu_q, \Sigma_q(t)) \Vert N(x_{t-1};\mu_\theta, \Sigma_q(t)))
\\ &= \underset{\theta}{argmin}\, \frac 1 2 \left[log\,\frac{\vert \Sigma_q(t) \vert}{\vert \Sigma_q(t) \vert} - d + tr(\Sigma_q(t)^{-1}\Sigma_q(t)) + (\mu_\theta - \mu_q)^T\Sigma_q(t)^{-1}(\mu_\theta - \mu_q) \right]
\\ &= \underset{\theta}{argmin}\, \frac {1} {2\sigma^2_q(t)} \left[\Vert \mu_\theta - \mu_q \Vert^2_2\right]
\end{aligned}
$$

여기서 $\mu_q(x_t, x_0)$ 는 $\frac{\sqrt{\alpha_t} (1-\bar{\alpha}_{t-1}) x_t + \sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t) x_0}{\left(1-\bar{\alpha}_t \right)}$ 이고,  여기서 $\alpha_t$ 와 $x_t$는 이미 주어진 값이고, 여기서 $x_0$ 만이 생성 과정에서 모르는 값이기 때문에 이 부분만 대체하여 학습 효율성을 높인다. 따라서 $\mu_\theta(x_t, t)$ 를 다시 써보면 아래와 같다. $\hat{x}_\theta$ 는 $x_t, t$를 입력으로 받는 신경망이다.

$$
\mu_\theta(x_t, t) = \frac{\sqrt{\alpha_t} (1-\bar{\alpha}_{t-1}) x_t + \sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t) \hat{x}_\theta(x_t, t)}{\left(1-\bar{\alpha}_t \right)}
$$

위 식을 가지고 다시 정리를 해보면,

$$
\begin{aligned}
&\underset{\theta}{argmin}\,D_{KL}\,(q(x_{t-1}\vert x_t, x_0) \Vert p_\theta(x_{t-1}\vert x_t))
\\ &= \underset{\theta}{argmin}\, \frac {1} {2\sigma^2_q(t)} \left[\Vert \mu_\theta - \mu_q \Vert^2_2\right]
\\ &= \underset{\theta}{argmin}\, \frac {1} {2\sigma^2_q(t)} \frac{\sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t)}{\left(1-\bar{\alpha}_t \right)} \left[\left\Vert 
\hat{x}_\theta(x_t, t) - x_0
\right\Vert^2_2\right]
\end{aligned}
$$

**따라서, VDM은 임의의 노이즈가 추가된 이미지($x_t$) 와 $t$ 로, 오리지널 이미지($x_0$) 를 신경망을 통해 예측하는 것으로 요약됩니다.

위 식은 Reparameterization trick 을 이용하여 다르게 표현 할 수 있다.

$$
\begin{aligned}
x_0 &= \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_0}{\sqrt{\bar{\alpha}_t}} \quad \text{이므로,}
\\ \mu_q(x_t, x_0) &= \frac{\sqrt{\alpha_t} (1-\bar{\alpha}_{t-1}) x_t + \sqrt{\bar{\alpha}_{t-1}} (1-\alpha_t) x_0}{\left(1-\bar{\alpha}_t \right)}
\\ &= 
\frac{1}{\sqrt{\alpha_t}}  x_t
- \frac{1-\alpha_t}{\sqrt{1- \bar{\alpha}_t}\sqrt{\alpha_t}} \epsilon_0
\\ \therefore \mu_\theta(x_t,t) &= \frac{1}{\sqrt{\alpha_t}}  x_t
- \frac{1-\alpha_t}{\sqrt{1- \bar{\alpha}_t}\sqrt{\alpha_t}} \hat{\epsilon}_\theta(x_t, t) 
\end{aligned}
$$

이 식을 이용하여, KL Divergence를 다시 전개 해보면,

$$
\begin{aligned}
&\underset{\theta}{argmin}\,D_{KL}\,(q(x_{t-1}\vert x_t, x_0) \Vert p_\theta(x_{t-1}\vert x_t))
\\ &= \underset{\theta}{argmin}\, \frac {1} {2\sigma^2_q(t)} \left[\Vert \mu_\theta - \mu_q \Vert^2_2\right]
\\ &= \underset{\theta}{argmin}\, \frac {1} {2\sigma^2_q(t)} 
\left[ \left\Vert
\frac{1}{\sqrt{\alpha_t}}  x_t - \frac{1-\alpha_t}{\sqrt{1- \bar{\alpha}_t}\sqrt{\alpha_t}} \epsilon_0
- \frac{1}{\sqrt{\alpha_t}}  x_t + \frac{1-\alpha_t}{\sqrt{1- \bar{\alpha}_t}\sqrt{\alpha_t}} \hat{\epsilon}_\theta(x_t, t)
\right\Vert^2_2 \right]
\\ &= \underset{\theta}{argmin}\, \frac {1} {2\sigma^2_q(t)}  \frac{(1-\alpha_t)^2}{(1- \bar{\alpha}_t)\alpha_t}
\left[ \left\Vert
  \epsilon_0 - \hat{\epsilon}_\theta(x_t, t)
\right\Vert^2_2 \right]
\end{aligned}
$$

여기서 $\hat{\epsilon}_\theta(x_t, t)$ 는 $x_0$ 에서 $x_t$ 를 결정하는 소스 노이즈 $\epsilon \sim N(\epsilon; 0, I)$를 예측하게 학습하는 신경망이고, 따라서 원래 이미지 $x_0$ 를 예측하는 것과 노이즈를 예측하는 것이 동등하다는 것을 볼 수 있다.
몇 몇 연구에 의하면 노이즈를 예측하는 것이 더 나은 결과를 보인다고 한다.
*정리해보면, 결국 주어진 $x_t$에 대하여  $\mu, x_0, \epsilon_0$ 계산 하는 것은 모두 동등하다고 할 수 있다.

