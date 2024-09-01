---
layout: post
title: DGM(CS236) Lec11~12. Energy-Based Models
subtitle: Energy-Based Models
date: 2024-08-31 23:13:00 +0900
category: content
tags:
  - vision
use_math: true
---

아래 내용은 Stanford의 CS236(Deep Generative Model) 강의를 듣고 필기한 노트입니다.

### Recap
- 모델 패밀리
    - Autoregressive Models: $p_\theta(x) = \prod_{i=1}^n\,p_\theta(x_i\vert x_{<i})$
    - Variational Autoencoders: $p_\theta(\mathbf{x}) = \int p(\mathbf{z}) p_\theta\mathbf{(x \vert z)}d\mathbf{z}$
    - Normalizing flow models: $p_\theta(x) = p(\mathbf{z}) \left[ det\,J_{f_\theta}(\mathbf{x}) \right]$, where $\mathbf{z} = f_\theta(\mathbf{x}).$
		- 위 세 모델은 방식에 따라  모델의 구조가 제한됨.
	- GAN
		- 모델의 구조는 매우 유연하나, 아래의 단점이 있다.
			- likelihood는 계산할 수 없음(intractable)
			- 학습 불안정
			- 모델 비교 어려움
			- mode collapse 문제

### Energy-based models (EMBs)
- maxium likelihood 기반이나, 매우 유연한  모델 구조를 사용할 수 있다.
- 학습이 안정적이다
- 상대적으로 샘플 퀄리티가 좋다
- 다른 구조와 같이 구성하여 사용하기 유연하다.

### Parameterizing probability distributions
- 확률 분포 $p(x)$는 아래의 특징을 가진다
	- Non-negative : $p(x) \geq 0$
	- sum-to-one : $\sum_x p(x) = 1$ or ($\int p(x)dx = 1$ for 연속확률변수)
		- 총량이 고정되어 있으므로, $p(x_{train})$ 이 커지면, $x_{train}$ 이 다른 변수 외에 더 높은 가능도(likelihood)를 가지는 것이 보장된다.
- None-negative 한 함수는 많으나 sum-to-one이 어렵다.
- 즉, $g_\theta(\mathbf{x}) \geq 0$은 쉬우나, $g_\theta(\mathbf{x})$가 일반적으로 정규화(normalized) 되지 않는다.
- **Solution :**
  
$$
p_\theta(\mathbf{x}) = \frac{1}{Z(\theta)} g_\theta(\mathbf{x}) = \frac{1}{\int g_\theta(\mathbf{x})d\mathbf{x}}g_\theta(\mathbf{x}) = \frac{1}{\text{Volume}(g_\theta)}g_\theta(\mathbf{x})
$$

- 여기서 $Z(\theta)$ 를 partition function, normlaization constant 또는 total volume 이라고 함.
- 위 정의에 의해, $\int p_\theta(\mathbf{x})d\mathbf{x} = \int \frac{g_\theta(\mathbf{x})}{Z(\theta)}d\mathbf{x} = \frac{Z(\theta)}{Z(\theta)} = 1$
- 따라서 $g_\theta(\mathbf{x})$을 볼륨이 계산 될 수 있는 것으로 선택해야함.
	- Gaussian 
		- function :  $g(\mu, \sigma)(x) = e^{-\frac{(x-\mu)^2}{2\sigma^2}}$. 
		- Volume : $\sqrt{2\pi\sigma^2}$
	- Exponential
		- function : $g_\lambda(x) = e^{-\lambda x}$
		- Volume : $\frac{1}{\lambda}$
	- Exponential Family
		- function : $g_\theta(x) = h(x)exp(\theta \cdot T(x))$.
		- Volume : $exp(\,A(\theta)\,)$, where $A(\theta)=log \int h(x) exp(\theta \cdot T(x))dx$
		- Normal, Possion, exponential, Bernoulli, beta, gamma, Dirichlet, Wishart, etc.
- 약간의 제약을 생기지만 훨씬 분포의 자유도를 줌.
- 이러한 방식은 Volume 계산이 가능하기 때문에, 아래 방법들로 학습이 가능함
	- Autoregressive : Products of normalized objects $p_\theta(x)p_{\theta '(x)}(y)$
		- $\int_x\int_y p_\theta(x)p_{\theta '(x)}(y)dx\,dy = \int_x p_\theta(x) \int_y p_{\theta '(x)}(y)dydx = \int_x p_\theta(x)dx = 1$
		- $\int_y p_{\theta '(x)}(y)dy = 1$ 이므로
	- Latent variables : Mixture of normalized objects $\alpha\,p_\theta(x) + (1-\alpha)p_{\theta '}(x)$
		- $\int_x \alpha p_\theta(x) + (1-\alpha)p_{\theta '}(x)dx = \alpha + (1- \alpha) = 1$
- Volume/normaliation constant of $g_\theta(x)$ 가 계산 하기 어려운 경우에는 어떻게 할까? 
	- *이 제약을 없애는 것이 EBM*

### Energy-based model
- EBM 의 정의
	- $p_\theta(x) = \frac{1}{\int exp(f_\theta(x))dx}exp(f_\theta(x)) = \frac{1}{Z(\theta)}exp(f_\theta(x))$
		- $exp(f_\theta(x))$ 는 Non-negative 특성을,
		- $Z(\theta)$ 는 partition function
- 왜 Exponentional 일까?
	- 확률에서 매우 큰 분산을 측정하고 싶음(Want to capture very large variations)
	- Exponential families. 많은 분산이 이러한 형태로 사용됨
	- 확률적 물리(Maximum entropy, 제2열역학) 에서 이러한 분산이 가정으로 사용됨.
		- $-f_\theta(x)$ 는 **energy** 라고 불림. 이에 Energy-based model로 명명
		- 직관적으로, $x$ 가 낮은 에너지(높은 $f_\theta(x)$) 를 가지면, 가능도가 높아짐(more liekly)
- 장점
	- 어떠한 형태의 함수($f_\theta(x)$)도 사용할 수 있음.
- 단점
	- $p_\theta(x)$ 로부터 샘플링이 어려움
	- likelihood $p_\theta(x)$의 평가와 최적화가 어려움. 즉 학습이 어려움
	- feature 학습이 없음. (그러나 잠재 변수를 추가할 수 있음)
- 차원의 저주
	- x의 차원이 커질 수록 $Z(\theta)$의 연산량이 기하급수적으로 늘어남
- 그러나 어떠한 Task는 $Z(\theta)$를 필요 하지 않음.

### Applications of Energy-based models
- EBM $P_\theta(x)$ 에 대하여, $x, x'$을 평가하려면, $Z(\theta)$ 가 필요함.
- 그러나 그 비율은 $Z(\theta)$가 불필요.
	- $\frac{p_\theta(x)}{p_\theta(x')} = exp(f_\theta(x) - f_\theta(x'))$
- 이는 아래의 활용이 가능함
	- Anomaly detection
	- denoising
	- object recognition
	- sequence labeling
	- image retoration
	- 이러 한 활용에는 상대 비교(relative comparisions)로 학습이 가능.

### 학습에 관한 직관
- 목표 : $\frac{exp\{f_\theta(x_{train})\}}{Z(\theta)}$의 최대화. 즉 분자는 높이고, 분모는 낮추기.
- 직관
	- 모델이 정규화 되어있지 않기 때문에, un-normalized log-probabilty $f_\theta(x_{train})$ 을 키우는 것이, $x_{train}$ 의 상대적 가능도를 높인다는 보장을 할 수 없다. (분자 높이기)
	- 그래서 오답을 동시에 내려 주는 것이 필요하다.  (분모 낮추기)
- 아이디어:
	- $Z(\theta)$ 를 직접적으로 계산하는 것 대신, Monte Carlo estimate 를 하자 (아주 소수 샘플 사용)
- **Contrastive divergence algorithm**
	- $p_\theta$ 로부터 샘플을 뽑고, $X_{sample} \sim p_\theta$.
	- $\nabla_\theta(f_\theta(x_{train})-f_\theta(x_{sample}))$ 로 Gradient descent를 수행한다.
	- 즉, Training 데이터의 가능도가 모델의 sample 데이터의 가능도보다 크게 만든다.
- Maximum log-likelihood
	- $max_\theta \,log (\frac{exp\{f_\theta(x_{train})\}}{Z(\theta)} ) = max_\theta\, f_\theta(x_{train} )- log \, Z(\theta)$
- Gradient of log-likelihood

$$
\begin{aligned}
&\nabla_\theta f_\theta(x_{train}) - \nabla_\theta\,log\,Z(\theta)
\\ &= \nabla_\theta f_\theta(x_{train}) - \frac{\nabla_\theta Z(\theta)}{Z(\theta)}
\\ &= \nabla_\theta f_\theta(x_{train}) - \frac{1}{Z(\theta)} \int \nabla_\theta \, exp(f_\theta(x))dx
\\ &= \nabla_\theta f_\theta(x_{train}) - \frac{1}{Z(\theta)} \int exp(f_\theta(x)) \nabla_\theta f_\theta(x)dx
\\ &= \nabla_\theta f_\theta(x_{train}) -  \int \frac{exp(f_\theta(x))}{Z(\theta)} 
\nabla_\theta f_\theta(x)dx
\\ &= \nabla_\theta f_\theta(x_{train}) - E_{x_{sample}} \left[ \nabla_\theta f_\theta(x_{sample}) \right]
\\ &\approx  \nabla_\theta f_\theta(x_{train})  - \nabla_\theta f_\theta(x_{sample})
\\ & where \,\, x_{sample} \sim exp(f_\theta(x_{sample}))/Z(\theta)
\end{aligned}
$$

> 참고
>  첫번째 수식 변환 : $f(x) =log\,g(x)$ 일 때, $f'(x) = \frac{g'(x)}{g(x)}$ .
>  세번째 수식 변환 : chain rule 적용. $\int \nabla_\theta \, exp(f_\theta(x))dx = \int exp(f_\theta(x)) \nabla_\theta f_\theta(x)dx$, $\nabla e^x = e^x$
>  다섯번째 수식 변환 : $\frac{exp(f_\theta(x))}{Z(\theta)}$ 는 모델의 확률. $\int p(x) f(x) = E_x[f(x)]$
>  여섯번째 수식 변환 : Single step monte carlo 적용.

- 그렇다면 샘플링은 어떻게?

### Sampling from energy-based models
- 바로 샘플링 할 방법은 없음 (각 가능한 샘플의 가능도를 쉽게 계산할 수 없으므로)
- 그러나 두 샘플 $x, x'$ 을 비교 할 수 있음.
- **Markov Chain Monte Carlo** 이용 (iterative approach) - *Metropolis-Hastings MCMC*
	1. $x^0$를 랜덤하게 초기화, $t=0$
	2. $t=0$부터, $t = T-1$ 까지 아래를 반복
		1. $x' = x^t + \text{noise}$, (이때 noise는 아주 작은 변환)
		2. 만약 $f_\theta(x') \gt f_\theta(x^t)$ 이면, $x^{t+1} = x'$
		3. 아닌 경우, 특정한 확률 ($exp(f_\theta(x') - f_\theta(x^t))$) 로 $x^{t+1} = x'$
			- 약간의 explore
	- 이론적으로 $T \rightarrow \infty$ 이면, $x^T$ 는 $p_\theta(x)$ 에 수렴.
	- 실제론 많은 iteration이 필요하고, 차원에 따라 매우 느려진다.
		- Sampling is expensive!
		- 즉 inference가 느리고, training때 sampling이 쓰이므로, traing 도 느림.

### Sampling form EBMs: unadjusted Langevin MCMC
- Unadjusetd Langevin MCMC:
	1. $x^0$를 랜덤하게 초기화, $t=0$
	2. $t=0$부터, $t = T-1$ 까지 아래를 반복
		1. $\mathbf{z}^t \sim N(0, I)$
		2. $\mathbf{x}^{t+1} = \mathbf{x}^t + \epsilon \nabla_\mathbf{x}\, log p_\theta(\mathbf{x}) \vert_\mathbf{x=x^t} + \sqrt{2\epsilon}\mathbf{z}^t$ 
			- $\epsilon$ 은 step size
			- $\nabla_\mathbf{x}\, log p_\theta(\mathbf{x}) \vert_\mathbf{x=x^t}$ 는 log likelihood의 기울기 방향
- 특성
	- $T \rightarrow \infty$ 이고 $epsilon \rightarrow 0$ 이면, $x^T$ 는 $p_\theta(x)$ 에 수렴.
	- 연속 에너지 기반 모델에서, $\nabla_x\, log \, p_\theta(x) = \nabla_x f_\theta(x)$
		- $\nabla_x\, log \, p_\theta(x) = \nabla_x f_\theta(x) - \nabla_x log\, Z(\theta)$ 인데, $log\, Z(\theta)$ 는 $x$와 무관하므로, 0
	- 차원이 늘어날 수록 수렴이 느려짐.

> 참고
> - Langevin dynamics(랑주뱅 동역학) 
> 	- 물리학에서 미시적 입자의 운동을 설명하는 확률적 방정식입니다. 이 방정식은 결정론적 힘과 확률적 힘이 결합된 형태로 표현되며, 입자의 열적 움직임을 모델링합니다.
> 	- 입자의 운동 방정식
> 		- $\frac{dx}{dt} = -\nabla \, U(x) + \sqrt{2D} \cdot \epsilon (t)$
> 			- $U(x)$ 는 포텐셜 에너지 함수
> 			- $D$ 는 확산 계수
> 			- $\epsilon (t)$ 는 백색 잡음(white noise)
> 	- Unadjusted Langevin MCMC의 업데이트 식은 이 동역학에서 유도된 것으로, 에너지가 높은 상태에서 낮은 상태로 이동하는 경향(즉, 그래디언트를 따라 이동함)과 함께 확률적 노이즈가 추가되어 있습니다

- Sampling은 그렇다 치고, Training 때 sampling 없이 할 수 있는 방법이 없을까?

### Score function
- Energy-based model : $p_\theta(x) = \frac{exp(f_\theta(x))}{Z(\theta)}$, $log\,p_\theta(x) = f_\theta(x) - log\,Z(\theta)$
- 이때의 (Stein) Score function은 아래와 같다

$$
s_{\theta}(x) := \nabla_x\,log\,p_\theta(x) = \nabla_x\,f_\theta (x) - \nabla_x\,log\,Z(\theta) = \nabla_x\,f_\theta(x)
$$

- $\nabla$ 의 대상이 $\theta$가 아닌 $x$ 즉 샘플 변화이다.

### Score mathing
- Fisher divergence between $p(\mathbf{x})$ and $q(\mathbf{x})$
	- p와 q가 비슷하면, 비슷한 vector field of gradient를 가질거야
	- 두 vector field of gradient 간의 Avg L2 distance

$$
D_F(p,q) := \frac{1}{2}E_{x\sim p}[\Vert \nabla_x\,log\,p(x) - \nabla_x\, log\,q(x) \Vert^2_2]
$$


- $p_{data}(x)$ 와 energy base model $p_\theta(x) \propto \, exp(f_\theta(x))$ 간의 Fisher divergence를 최소화 해보자.

$$
\begin{aligned}
& \frac{1}{2}E_{x\sim data}[\Vert \nabla_x\,log\,p_{data}(x) - s_\theta(x) \Vert^2_2]
\\ &= \frac{1}{2}E_{x\sim data}[\Vert \nabla_x\,log\,p_{data}(x) - \nabla_x\, f_\theta(x) \Vert^2_2]
\end{aligned}
$$

- 여기서 $\nabla_x\,log\,p_{data}(x)$ 를 알 수 없다. $\rightarrow$ *부분 적분(Integration by Parts)을 이용해보자.*

$$
\begin{aligned}
& \frac{1}{2}E_{x\sim data}[(\nabla_x\,log\,p_{data}(x) - \nabla_x\, f_\theta(x) )^2] \text{ (Univariate case, 1d case)}
\\ &= \frac{1}{2}\int p_{data}(x)[(\nabla_x\,log\,p_{data}(x) - \nabla_x\, f_\theta(x) )^2]dx
\\ &= \frac{1}{2}\int p_{data}(x)(\nabla_x\,log\,p_{data}(x))^2dx +  \frac{1}{2}\int p_{data}(x)(\nabla_x\,log\,p_{\theta}(x))^2dx
\\ & \quad\quad\quad - \int p_{data}(x)\nabla_x\,log\,p_{data}(x)\nabla_x\,log\,p_\theta(x)dx
\end{aligned}
$$

- 여기서 마지막 항에 대하여 부분 적분을 적용해보면, $\int f'g = fg - \int g'f$

$$
\begin{aligned}
&- \int p_{data}(x)\nabla_x\,log\,p_{data}(x)\nabla_x\,log\,p_\theta(x)dx
\\&= - \int p_{data}(x)\frac{\nabla_x\,p_{data}(x)}{p_{data}(x)}\nabla_x\,log\,p_\theta(x)dx
\\&= - \int \nabla_x\,p_{data}(x)\,\nabla_x\,log\,p_\theta(x)dx
\\&= -p_{data}(x)\,\nabla_x\,log\,p_\theta(x)|^{\infty}_{x=\infty} + \int \nabla^2_x\,log\,p_\theta(x) \, p_{data}(x) dx
\\&= \int \nabla^2_x\,log\,p_\theta(x) \, p_{data}(x) dx
\end{aligned}
$$

> 참고 :
> - 확률 밀도 함수 $p_{data}(x)$ 가 x가 $\pm\,\infty$ 로 갈때  0으로 수렴한다고 가정. 
> - 따라서,  $-p_{data}(x)\,\nabla_x\,log\,p_\theta(x)\vert^{\infty}_{x=\infty} = 0$

- Univariate score matching을 다시 적어보면, 

$$
\begin{aligned}
& \frac{1}{2}E_{x\sim data}[(\nabla_x\,log\,p_{data}(x) - \nabla_x\, f_\theta(x) )^2] \text{ (Univariate case, 1d case)}
\\ &= \frac{1}{2}\int p_{data}(x)(\nabla_x\,log\,p_{data}(x))^2dx +  \frac{1}{2}\int p_{data}(x)(\nabla_x\,log\,p_{\theta}(x))^2dx
\\ & \quad \quad \quad - \int p_{data}(x)\nabla_x\,log\,p_{data}(x)\nabla_x\,log\,p_\theta(x)dx
\\ &= \frac{1}{2}\int p_{data}(x)(\nabla_x\,log\,p_{data}(x))^2dx +  \frac{1}{2}\int p_{data}(x)(\nabla_x\,log\,p_{\theta}(x))^2dx + \int \nabla^2_x\,log\,p_\theta(x) \, p_{data}(x) dx
\\ &= const + \frac{1}{2}\int p_{data}(x)(\nabla_x\,log\,p_{\theta}(x))^2dx + \int \nabla^2_x\,log\,p_\theta(x) \, p_{data}(x) dx
\\ &= const + E_{x\sim p_{data}}[\frac{1}{2}\nabla_x\,log\,p_{\theta}(x))^2 + \nabla^2_x\,log\,p_\theta(x)]
\end{aligned}
$$

- Multivariate score maching 

$$
\begin{aligned}
& \frac{1}{2}E_{x\sim data}[\Vert \nabla_x\,log\,p_{data}(x) - \nabla_x\, f_\theta(x) \Vert^2_2] 
\\ &= const + E_{x\sim p_{data}}[\frac{1}{2}\Vert\nabla_x\,log\,p_{\theta}(x)\Vert^2_2 + tr(\nabla^2_x\,log\,p_\theta(x))]
\end{aligned}
$$

> 참고:
> - $\nabla_x^2\,log\,p_\theta(x)$ 는 $log\,p_\theta(x)$의 Hessian

- *이러한 변환은  $p_{data}$ 의 score에 대한 dependency가 사라지게 한다.*

### Score matching 알고리즘
- 데이터 포인트 $\{ \mathbf{x_1, x_2, ..., x_n}\} \sim p_{data}(\mathbf{x})$ 미니 배치를 샘플링한다
- Score matching loss with the empirical mean

$$
\begin{aligned}
& \frac{1}{n}\sum_{i=1}^n\,[\frac{1}{2}\Vert\nabla_x\,log\,p_{\theta}(x_i)\Vert^2_2 + tr(\nabla^2_x\,log\,p_\theta(x_i))]
\\ =\, &\frac{1}{n}\sum_{i=1}^n\,[\frac{1}{2}\Vert\nabla_x\,f_{\theta}(x_i)\Vert^2_2 + trace(\nabla^2_x\,f_\theta(x_i))]
\end{aligned}
$$

- Stochastic gradient descent

- 특성
    - EBM에서 sampling 할 필요가 없음
    - 그러나 Hessian 연산 $tr(\nabla_x^2\,log\,p_\theta(x_i))$ 이 일반적으로 큰 모델에서 어렵다.

