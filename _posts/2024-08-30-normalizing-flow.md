---
layout: post
title: DGM(CS236) Lec07~08. Normalizing Flow Models
subtitle: Normalizing Flow Models
date: 2024-08-30 11:51:00 +0900
category: content
tags:
  - vision
use_math: true
---

아래 내용은 Stanford의 CS236(Deep Generative Model) 강의를 듣고 필기한 노트입니다.

### Recap
- 모델 패밀리
    - Autoregressive Models: $p_\theta(x) = \prod_{i=1}^n\,p_\theta(x_i\vert x_{<i})$
    - Variational Autoencoders: $P_\theta(x) = \int p_\theta(x,z)dz$
- Autoregressive Model은 likehood는 계산가능하지만, 피쳐를 직접적으로 학습할 방법이 없음
- Variational Autoencoder는 feature 학습이 가능하지만(latent vector $z$), $p_\theta(x)$ 즉 marginal likelihood를 계산하기 어려움(intractable)

- **그럼 계산가능한 likelihood를 가진 latent variable model을 디자인 할 수 없을까?**

### Key idea
- 간단한 분포 (Easy to sample and evaluate densities)를 역변환(**invertible transformation**)을 통해 복잡한 분포로 맵핑해보자.


### Flow model with variational autoencoder
Flow model은 VAE와 유사하다.
1. 간단한 분포로부터 시작: $z \sim N(0, I) = p(z)$
2. $p(x \vert  z)$를 통해 변환 = $N(\mu_\theta(z), \Sigma_\theta(z))$
3. $p(z)$는 간단하지만, Marginal, $p_\theta(x)$는 매우 유연하고 복잡함. 그러나 $p_\theta(x)=\int p_\theta(x, z)dz$ 는 모든 x를 생성할 수 있는 모든 z를 봐야하므로 계산하기 매우 비싸다. (interactible)
4. *만약 $p(x\vert z)$ 와 $p(z\vert x)$ 를 쉽게 변환 할 수 있게 디자인 한다면?* 
    - $x = f_\theta(z)$를 deterministic 하고 invertible function of $z$로 만든다면 가능(다만 $z$와 $x$ 같은 차원이어야함)
    - 어떠한 $x$에 대하여도 대응되는 유니크한 $z$가 존재.

### Change of variables formula
- Change of variables (1D case):
    - if $X = f(Z)$ and $f(\cdot)$ is monotone with inverse $Z = f^{-1}(X) = h(X)$, then
    $$
    \begin{aligned}
    p_x(x) &= p_z(h(x))\vert h'(x)\vert
    \\ p_x(x) &= p_z(z)\frac{1}{f'(z)}
    \end{aligned}
    $$
- Change of variables (N D case):
    - Let $X = AZ$ for a square invertible matrix $A$, with inverse $W = A^{-1}$.
    - X는 $\vert det(A) \vert$ 공간의 평면체에 고르게 분포된다.
    - A는 선형이동!!
    $$
    \begin{aligned}
    p_x(x) &= p_z(Wx) \frac{1}{\vert det(A) \vert}
    \\ p_x(x) &= p_z(Wx) \vert det(W) \vert
    \end{aligned}
    $$
    - because if $W = A^{-1},. det(W)=\frac{1}{det(A)}$ 
- Change of variables (General case):
    - 행렬 A로부터 *선형 변환* 된 경우, 볼륨의 변화는 행렬 A의 행렬식이다. 
    - 행렬 A로부터 *비선형 변환* $\mathbf{f}(\cdot)$에 대하여, *linearlized change(선형화된 변화 - 극소지점을 봤을때 선형)* 의 볼륨은 $\mathbf{f}(\cdot)$의 자코비안의 행렬식이다.
    - The mapping between $Z$ and $X$, given by $\mathbf{f}:\mathbb{R}^n \rightarrow \mathbb{R}^n$, is invertible such that $X= \mathbf{f}(Z)$ and $Z=\mathbf{f}^{-1}(X)$ 
    $$
    \begin{aligned}
    p_x(\mathbf{x}) &=& p_z(\mathbf{f}^{-1}(\mathbf{x}))  \left| \partial \mathbf{f}^{-1}(\mathbf{x}) \over \partial \mathbf{f} \right|
    \\
    \end{aligned}
    $$
    - 이 때에, VAE와 달리 $x, z$는 continuous하고 같은 차원에 있어야함.
    - Invertible matrix $A$에 대하여, $det(A^{-1}) = det(A)^{-1}$ 이므로,
    $$
    \begin{aligned}
    p_x(\mathbf{x}) &=& p_z(\mathbf{f}^{-1}(\mathbf{x}))  \left| \partial \mathbf{f}(\mathbf{z}) \over \partial \mathbf{z} \right|^{-1}
    \\
    \end{aligned}
    $$

### Normalizing flow models
Normalizing flow model에서, $Z$ 와 $X$ 사이의 매핑은, 다음 함수 $\mathbf{f}_\theta : \mathbb{R}^n \rightarrow \mathbb{R}^n$ 이고, **deterministic** 하고 **invertible** 하다. 
따라서 $X = \mathbb{f}_\theta(Z)$ 이고, $Z = \mathbb{f}_\theta^{-1}(X)$ 이다.
- Change if variables를 사용하면, marginal likelihood $p(x)$는 아래와 같다.
$$
\begin{aligned}
p_x(\mathbf{x};\theta) &=& p_z(\mathbf{f}_\theta^{-1}(\mathbf{x}))  \left| \partial \mathbf{f}_\theta^{-1}(\mathbf{x}) \over \partial \mathbf{f} \right|
\\
\end{aligned}
$$

- Normalizing: Change of variables는 역변환 적용 후 후 normalized density(자코비안 행렬식)를 사용.
- Flow: 역변환이 서로서로 합성되어(composed) 구성
$$
\begin{aligned}
\mathbf{z}_m &= \mathbf{f}_\theta^m \circ ... \, \circ \mathbf{f}_\theta^1(\mathbf{z}_0)
\\ &= \mathbf{f}_\theta^m(\mathbf{f}_\theta^{m-1}(...)(\mathbf{f}_\theta^1(\mathbf{z}_0))))
\\ &\triangleq \mathbf{f}_\theta(\mathbf{z}_0) 
\end{aligned} 
$$
- 간단한 분포 $\mathbf{z}_\theta$로 부터 시작 (e.g., 가우시안)
- 연속된 M개의 역변환을 적용하여 $\mathbf{x}=\mathbf{z}_M$ 를 얻음
- Change of variables에 의해,
$$
\begin{aligned}
p_x(\mathbf{x}; \theta) = p_z(\mathbf{f}_\theta^{-1}(\mathbf{x})) \prod_{m=1}^M  \left| \frac{ \partial (\mathbf{f}_\theta^m)^{-1}(\mathbf{z}_m)}{\partial \mathbf{z}_m }\right|
\\
\end{aligned} 
$$
> note: determinant of product equals product of determiants 

- 즉 NN을 아래와 같이 설계해야함
    - invertible 해야함.
    - 효율적으로 invert 가능해야함.
    - Jacobian이 determinant를 쉽게 계산 할 수 있어야함.


### Learning and Inference
- 데이터셋 D 에대하여 maximum likelihood를 통해 학습하면,
$$
\begin{aligned}
\underset{\theta}{max} \, log p_x(D;\theta) = \sum_{x \in D} log p_z( \mathbf{f}_\theta^{-1} (\mathbf{x}) + log \left| det(\frac{ \partial (\mathbf{f}_\theta^m)^{-1}(\mathbf{x})}{\partial \mathbf{x} }) \right| )
\end{aligned} 
$$
- Exact likelihood evaluation
- **Sampling** via forward transformation z -> x
$$
\begin{aligned}
\mathbf{z} \sim p_z(\mathbf{z}) \,\,\, \mathbf{x} = \mathbf{f}_\theta(\mathbf{z})
\end{aligned}
$$

- 이 때에, 자코비안의 행렬식을 쉽게 계산하기 위해 자코비안 행렬이 특별한 구조를 가지게 학습한다. 
    - 예, *삼각 행렬* 의 행렬식은 대각의  곱과 같다.

### Triangular Jacobian
- 만약 $x_i = f_i(\mathbf{z})$ 가 $\mathbf{z}_{\leq i}$ 에만 의존한다면, 자코비안은 하삼각행렬(lower triangular structure) 구조를 가진다. (autoregressive model과 유사.)
- 이렇게 되면 행렬식 연산이 선형 시간(O(n)) 에 계산 가능하다.

> 참고
$$
\begin{gathered}
\mathbf{x} = (x_1, ..., x_n) = \mathbf{f}(\mathbf{z}) = (f_1(\mathbf{z}), ... , f_n(\mathbf{z}))
\\ J = 
\left(
\begin{matrix}
\frac{\partial f_1}{\partial z_1} && ... && \frac{\partial f_1}{\partial z_n}
\\ \vdots && \ddots && \vdots
\\ \frac{\partial f_n}{\partial z_1} && ... && \frac{\partial f_n}{\partial z_n}
\end{matrix}
\right)
\end{gathered}
$$

## NICE(Nonlinear Independent components estimation) - Additive coupling layers
확률변수 $z$를 두 분리된 서브셋 $\mathbf{z}_{1:d}, \mathbf{z}_{d+1:n}$ for any $ 1 \leq d \le n$으로 나눈다.
- Forward mapping $\mathbf{z} \rightarrow \mathbf{x}$:
    - $\mathbf{x}_{1:d} = \mathbf{z}_{1:d}$ (identity transformation) 변환 없음
    - $\mathbf{x}_{d+1:n} = \mathbf{z}_{d+1:n} + m_\theta(\mathbf{z}_{1:d})$
        - 여기서 $m_\theta(\cdot)$은 $\theta$를 파타미터로 하는 Neural network이고, $d$ 입력에 $n-d$ 출력을 내어놓는다. 
        - vector만 더하기 때문에 단순한 이동(shift).
- Inverse mapping $\mathbf{x} \rightarrow \mathbf{z}$
    - $\mathbf{z}_{1:d} = \mathbf{x}_{1:d}$ (identity transformation) 변환 없음
    - $\mathbf{z}_{d+1:n} = \mathbf{x}_{d+1:n} - m_\theta(\mathbf{x}_{1:d})$
        - $\mathbf{x}_{1:d} = \mathbf{z}_{1:d}$ 이므로, shift를 x로 표현 할 수 있음.
- Jacobian of forward mapping $\mathbf{z} \rightarrow \mathbf{x}$:
    - 단순히 Shifting 이었기 때문에, Jacobian의 대각은 모두 Identity 이다.
    - $\mathbf{x}_{1:d} = \mathbf{z}_{1:d}$ 또는  $\mathbf{x}_{d+1:n} = \mathbf{z}_{d+1:n} + m_\theta(\mathbf{z}_{1:d})$ 이므로 
    $$
    \begin{gathered}
    J = \frac{\partial \mathbf{x}}{\partial \mathbf{z}} = 
    \left(
    \begin{matrix}
    I_d && 0
    \\ \frac{\partial \mathbf{x}_{d+1:n}}{\partial \mathbf{z}_{1:d}} && I_{n-d}
    \end{matrix}
    \right)
    \\
    \\ det(J) = 1
    \end{gathered}
    $$
    - 좌상단 항: $\frac{\partial \mathbf{x}_{1:d}}{\partial \mathbf{z_{1:d}}} = \frac{\partial \mathbf{z}_{1:d}}{\partial \mathbf{z}_{1:d}} = I$
    - 우상단 항:$\frac{\partial \mathbf{x}_{1:d}}{\partial \mathbf{z_{d+1:n}}} = \frac{\partial \mathbf{z}_{1:d}}{\partial \mathbf{z}_{d+1:n}} = 0$ ($\mathbf{z}_{1:d}$ 와 $\mathbf{z}_{d+1:n}$) 는 서로 의존 관계가 없음.
    - 좌하단 항: $\frac{\partial \mathbf{x}_{d+1:n}}{\partial \mathbf{z_{1:d}}}$ 은 식 그대로
    - 우하단 항: $\frac{\partial \mathbf{x}_{d+1:n}}{\partial \mathbf{z_{d+1:n}}} = \frac{\partial (\mathbf{z}_{d+1:n} + m_\theta(z_{1:d}))}{\partial{z_{d+1}:n}} = I + 0$ 이므로, $I$
   
- 행렬식이 1이므로 **Volume preserving transformation** 이다.

