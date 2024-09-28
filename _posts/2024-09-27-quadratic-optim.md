---
layout: post
title: Quadratic optimization in CV, PCA
subtitle: Quadratic optimization in CV, PCA
date: 2024-09-27 22:38:00 +0900
category: content
tags:
  - vision
use_math: true
---

CS485 ML for vision 과목을 듣고 정리 한 내용이다.
PCA와 Quadratic optimization 에 대해 다룬다.


### Gradient method
- Gradient descent 는 로컬 미니멈 혹은 맥시멈을 찾는 iterative optimization algorithm 이다. 
	- $\mathbf{x}_{i+1} = \mathbf{x}_{i} - \gamma_i \nabla(\mathbf{x}_{i}), \, i \ge 0$
- 함수 f 가 covex 이면, 모인 로컬 미니마가 글로벌 미니마이다.

### Lagrange multipliers for constrained optimization
- 라그랑지안 멀티플라이어는 equality constraints 가 제약으로 있는 함수의 최대/최소 값을 찾는 전략이다.
	- $\mathcal{L}(x,y, \lambda) = f(x,y) - \lambda \cdot g(x,y)$ . 여기서 $\lambda$는 상수 이다.
- 풀려고 하는 문제는 아래와 같다.

$$
\begin{gather}
\nabla_{x,y,\lambda} \mathcal{L}(x,y,\lambda) = 0
\\ \nabla_{x,y} f(x,y) = \lambda \nabla_{x,y} g(x,y)
\\ \nabla_\lambda \mathcal{L}(x,y, \lambda) = 0 \rightarrow g(x,y)=0
\end{gather}
$$

### Maximum variance formulation of PCA
- PCA는 차원 축소, 손실 압축, 피쳐 추출과 데이터 시각화에 쓰이는 기법이다.
- PCA는 *투영된 데이터의 분산이 최대화 되는* 더 낮은 차원의 선형 공간으로 데이터를 orthogonal projection(직교 투영) 하는 것으로 정의된다.
- 주어진 데이터셋 $\{\mathbf{x}_n\}, n= 1,...,N \, \mathbf{x}_n \in R^D$ 을 D 보다 훨씬 작은 M차원에 투영하였을 때 그 분산이 최대화 되는 것을 목표로 한다.

### Maximum vriance formulation of PCA (with M=1)
- 공간의 방향은 $\mathbf{u}_1 \in R^D\, s.t. \, \mathbf{u}^T_1 \mathbf{u}_1 = 1$ 로 정의
- 각 데이터 포인트 $\mathbf{x}_n$은 scalar value $\mathbf{u}^T_1 \mathbf{x}_n$ 가 된다.

$$
\begin{aligned}
\text{mean} &= \mathbf{u}^T_1 \mathbf{\bar x}, \, \mathbf{\bar x} = \frac 1 N \sum^N_{n=1}\mathbf{x}_n
\\ \text{varience} &= \frac 1 N \sum^N_{n=1} \{ \mathbf{u}^T_1 \mathbf{x}_n - \mathbf{u}^T_1\mathbf{\bar x} \}^2 = \mathbf{u}^T_1 \mathbf{Su}_1
\\ \text {covarience matrix } S &= \frac 1 N \sum^N_{n=1}(\mathbf{x}_n - \mathbf{\bar x})(\mathbf{x}_n - \mathbf{\bar x})^T
\end{aligned}
$$

- 우리의 목적은 투영된 분산의 최대화 이므로, 목적 함수 는 $J = \mathbf{u}^T_1 \mathbf{Su}_1$ 이 된다.
- 여기서 $\mathbf{u}_1$ 에 대하여 정규화 조건 $\mathbf{u}_1^T \mathbf{u}_1 = 1$ 을 만족해야 한다. (제약조건)
- 이를 라그랑주 멀티플라이어 공식으로 작성해보면 아래와 같다.

$$
\begin{gather}
\text{objective function } f(\mathbf{u}_1) = \mathbf{u}_1^T \mathbf{Su}_1
\\ \text{s.t. } g(\mathbf{u}_1) = 1 - \mathbf{u}_1^T \mathbf{u}_1 = 0
\\ L = \mathbf{u}_1^T \mathbf{Su}_1 + \lambda_1(1-\mathbf{u}_1^T \mathbf{u}_1)
\end{gather}
$$

- 라그랑주 멀티플라이어의 풀이를 이용하기위해 $\mathbf{u}_1$ 으로 편미분하고 그 값이 0이라고 하면 아래와 같다.

$$
\begin{aligned}
\frac {\partial L}{\partial \mathbf{u}_1} &= \frac {\partial}{\partial \mathbf{u}_1}(\mathbf{u}_1^T \mathbf{Su}_1)  + \frac {\partial}{\partial \mathbf{u}_1} \lambda_1(1-\mathbf{u}_1^T \mathbf{u}_1)
\\ &= 2\mathbf{Su}_1 - 2\lambda_1 \mathbf{u}_1 = 0
\\ \therefore  \mathbf{Su}_1 &= \lambda_1 \mathbf{u}_1
\end{aligned}
$$

- 따라서 고윳값의 정의 ($A\mathbf{x} = \lambda \mathbf{x}$) 에 따라,  **$\mathbf{u}_1$ 을 공분산 행렬 $\mathbf{S}$ 에 대한 고유벡터(eigenvector) 라고 할 수 있다.**
- $\mathbf{u}_1^T \mathbf{u}_1 = 1$ 이므로, **분산 $\mathbf{u}^T_1 \mathbf{Su}_1 = \lambda_1$ 임을 알 수 있다.**

### Maximum vriance formulation of PCA 해석
- 가장 큰 고윳값 $\lambda_1$ 에 해당하는 고유벡터 $\mathbf{u}_1$ 일 때, 최대 분산을 얻을 수 있다.
- 이러한 고유벡터를 주성분(*principal component*) 라고도 부른다.
- M차원의 subspace로 일반화하면, M개의 고유벡터($\mathbf{u_1, u_2, ..., u_M}$)와 공분산 행렬 $\mathbf{S}$에 대응되는 가장큰 고윳값($\lambda_1, \lambda_2, ... \lambda_M$) 을 얻을 수 있다.
- 여기서 고유벡터끼리는 직교하기 때문에, $\mathbf{u}^T_i \mathbf{u}_j$ 는 i,j가 같은 경우 1, 아닌 경우 0이 된다. 

$$
\mathbf{u}^T_i \mathbf{u}_j = \delta_{ij} \text{ (크로네커 델타: 두 인덱스가 같으면 1, 다르면 0이 되는 함수)}$$

### Minimum error formulation of PCA
- 이외에도 reconstrunction error를 최소화 하는 방식으로도 접근 할 수 있다.

$$
J = \frac 1 N \sum^N_{n=1}\Vert \mathbf{x}_n - \mathbf{\tilde x}_n \Vert^2
$$

### Low-dimensional computation of Eigenspace, when D >> N
- 차원 수 보다 데이터의 개수가 훨씬 작은 경우
- 데이터셋 ${\mathbf{x}_n}, n=1,...,N \,,\,\mathbf{x}_n \in R^D$ 를  M << D 인 M 차원으로 투영하는 것이 목표이다.
	- 일반적으로, 고유벡터 $\mathbf{u}_i$ 를 Matrix $AA^T$ 에 대하여 계산한다. ($S = \frac 1 N AA^T$ 를 간소화한 것)
	- 여기서 행렬 $AA^T$ 는 $D \times D$ 로 매우 크다. (not practical)
- $D \times D$ 인 $AA^T$ 대신 $N \times N$인 $A^T A$ 를 이용하여, 계산해보자.
	- 참고: 

$$
S = \frac 1 N \sum^N_{n=1}\underbrace{(\mathbf{x}_n - \mathbf{\bar x})}_{D \times N}\underbrace{(\mathbf{x}_n - \mathbf{\bar x})^T}_{N \times D}
$$

- $A^TA$를 이용하여 고유벡터 $\mathbf{V}_i$ 를 구한다.

$$
A^TA \mathbf{v}_i = \lambda_i \mathbf{v}_i
$$

- $\mathbf{u}_i$ 와 $\mathbf{v}_i$ 는 어떤 관계인가?

$$
\begin{gather}
A^TA \mathbf{v}_i = \lambda_i \mathbf{v}_i 
\\ \rightarrow AA^TA \mathbf{v}_i = \lambda_i A\mathbf{v}_i 
\\ \rightarrow SA\mathbf{v}_i = \lambda_i A \mathbf{v}_i
\\ \rightarrow S\mathbf{u}_i = \lambda_i\mathbf{u}_i \text{ where } \mathbf{u}_i = A \mathbf{v}_i
\\ \therefore AA^T \text{ 와 } A^TA \text{ 는 제약조건 } s.t. \mathbf{u}_i = A \mathbf{v}_i \text{ 일 때, 동일한 고유값과 고유 벡터를 가진다.}
\end{gather}
$$

- 참고사항
	- $AA^T$ 는 최대 D 고윳값과 고유벡터를 가진다
	- $A^TA$는 최대 N 또는 N-1 개의 고윳값과 고유벡터를 가진다
	- $A^TA$로 부터의 M개의 고유값은 $AA^T$의 M개의 고유값과 대응된다.

- $AA^T$의 M best 고유벡터를 $\mathbf{u}_i = A \mathbf{v}_i$ 를 이용하여 계산한다.

### Limitations of PCA
- Unsupervised
	- PCA는 데이터의 분산을 최대화 시키는 방향을 찾눈다 (Unsupervised) 
	- LDA(Linear Discriminant Analysis) 는 다른 클래스를 구분하는 최적의 방향을 찾는다 (discriminative or supervised)
- Linear model
	- PCA는 선형 모델이다.
	- 데이터가 비선형 manifold로 구성된다면, PCA는 Kernel Trick 을 이용하여 Kernel PCA로 확장 되어야 한다.
- Gaussian assumption
	- PCA는 데이터를 가우시안 분포로 가정하여, 2차 통계량(2nd order statistics) 에 집중하여 데이터를 처리함
	- 반면, ICA는 데이터를 고차 통계량을 활용하여 비선형적 독립성을 찾아 낼 수 있음
- Holistic bases
	- PCA는 전체적인(Holistic) 패턴을 반영하지만, 부분적인 해석이나 직관적 이해에 적합하지 않음.
	- 반면 NMF(Non-negative Matrix Factorization)은 부분적인 component를 검출함.
- Uniform prior on the subspace
	- 공분산 행렬을 기반으로 계산된 고유 벡터로부터 잠재 공간이 span함.
	- 따라서, PCA가 데이터를 변화하는데 있어서 잠재공간(subspace) 내의 모든 방향에 대해 균등한 사전 확률을 가정함.
	- PPCA(Probabilistic PCA)는 전통적인 PCA와는 달리, 확률 모델을 기반으로 데이터를 분석합니다. 기본 PCA는 단순히 데이터를 저차원 공간으로 투영하는 반면, PPCA는 데이터를 생성할 수 있는 확률적 메커니즘을 모델링합니다. 즉, 데이터가 정규 분포에서 나온다고 가정하고, 이를 바탕으로 차원을 축소합니다.
