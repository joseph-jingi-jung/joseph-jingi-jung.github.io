---
layout: post
title: Diffusion Models Beat GANs on Image Synthesis
subtitle: guided diffusion
date: 2024-10-14 23:38:00 +0900
category: content
tags:
  - vision
use_math: true
---

해당 논문의 두 가지 강조할만한 contribution

### 1. AdaGN (Adaptive Group normalization)
각 Residual block의 Group normalization 연산 후 그 결과를 timestep, class embedding과 통합한다. 
구현부 : https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/unet.py#L251

$$
\text{AdaGN}(h,y) = y_s \text{GroupNorm}(h) + y_b
$$

$y=[y_s, y_b]$는 timestep과 class embedding을 linear projection 하여 얻은 값
원래 conditional UNet 에서는 클래스 레이블 임베딩과 타임 스텝 임베딩을 단순히concat하여 사용하였는데, 해당 연구에서는 AdaGN을 추가

AdaGN의 핵심은 스케일(scale)과 이동(shift) 파라미터를 활용하는 것인데, 이 두 값은 정규화된 출력의 크기와 위치를 조정. 각 데이터 샘플에 맞는 스케일과 이동 값을 적용. 즉 특정 클래스의 이미지 생성 과정에서 해당 클래스의 특징을 더 잘 반영할 수 있도록 정규화 값이 조정

### 2. Classfier Guidance
고정된 Diffusion model을 기반으로 $x_t, t$가 주어졌을 때, 클래스 레이블 $y$ 를 예측하는 $p_\phi(y\vert x_t, t)$ 를 학습하고, 샘플링 과정에서 그 classifier의 gradients $\nabla_{x_t}log\,p_\phi(y\vert x_t, t)$ 를 이용하여, 해당 레이블의 이미지가 생성되게끔 유도.

Appendix H의 증명을 통해, 아래와 같이 denosing 과정을 정의할 수 있음

$$
p_\theta, \phi(x_t \vert x_{t+1}, y) = Z p_\theta(x_t \vert x_{t+1})p_\phi(y\vert x_t)
$$

여기서 $Z$는 정규화 상수.
#### Conditional sampling for DDPM

$$
\begin{aligned}
p_\theta(x_t \vert x_{t=1}) &= N(\mu, \Sigma)
\\ log\,p_\theta(x_t \vert x_{t+1}) &= -\frac 1 2 (x_t - \mu)^T \Sigma^{-1} (x_t - \mu) -C
\\ &\text{일 때}
\\log\, p_\theta(y\vert x_t) &\approx log\, p_\theta(y\vert x_t) \vert_{x_t=\mu} + (x_t - \mu) \nabla_{x_t} log\, p_\theta(y\vert x_t) \vert_{x_t=\mu}
\\ & \text{by 테일러 급수 2차, } f(x) = f(a) + f'(a) (x-a)
\\ 
\\ g &= \nabla_{x_t} log\, p_\theta(y\vert x_t) \vert_{x_t=\mu}, C_1 \text{ is constant.}

\\ \therefore log(p_\theta(x_t \vert x_{t+1}) p_\phi(y \vert x_t)) &\approx \underbrace{- \frac(x_t - \mu)^T \Sigma^{-1}(x_t - \mu)}_{ log\,p_\theta(x_t \vert x_{t+1})} + \underbrace{(x_t - \mu)g}_{log\, p_\theta(y\vert x_t)} + C_2
\\&= log\,p(z) + C_4, z\sim N(\mu + \Sigma g, \Sigma)
\end{aligned}
$$

구현부 : https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L367

Gaussian은 unconditional transition 과정과 유사하나, $\Sigma g$ 만큼 평균이 shift됨 
여기서 scale factor s를 추가

#### Conditional sampling DDIM case
모델 $\epsilon_\theta(x_t)$ 에 대하여, score function을 아래와 같이 정의 할 수 있다.

$$
\nabla_{x_t} log\,p_\theta(x_t) = -\frac{1}{\sqrt{1- \bar{\alpha}_t}} \epsilon_\theta(x_t)
$$

이를  $p(x_t)p(y\vert x_t)$의 score function 에 대입한다면, 아래와 같다.

$$
\begin{aligned}
\nabla_{x_t}log\,(p(x_t)p(y\vert x_t)) &= \nabla_{x_t} log\,p_\theta(x_t) + \nabla_{x_t}log\,p_\phi(y \vert x_t)
\\ &= -\frac{1}{\sqrt{1- \bar{\alpha}_t}} \epsilon_\theta(x_t) + \nabla_{x_t}log\,p_\phi(y \vert x_t)
\\
\\ &\text{define new epsilon prediction}
\\ \hat{\epsilon}(x_t) &:=  \epsilon_\theta(x_t) - \sqrt{1- \bar{\alpha}_t} \nabla_{x_t}log\,p_\phi(y \vert x_t)
\end{aligned}
$$

DDIM sampling을 할 때, $\epsilon_\theta(x_t)$ 대신, $\hat{\epsilon}(x_t)$ 를 이용하여 샘플링.
구현부 : https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L384C1-L386C10
