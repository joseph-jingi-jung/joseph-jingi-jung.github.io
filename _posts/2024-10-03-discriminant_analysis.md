---
layout: post
title: Discriminant analysis
subtitle: Fisherfaces
date: 2024-10-03 11:35:00 +0900
category: content
tags:
  - vision
use_math: true
---

CS485 ML for vision 과목을 듣고 정리 한 내용이다.
PCA와 LDA를 통한 데이터 구분을 다룬다 (Fisherfaces)

## Motivation
- Least square 관점에서 데이터를 가장 잘 분리하는 투영(Projection)
	- PCA는 데이터를 잘 표현하는데 유용함
	- Pooling 혹은 Projecting 이 데이터를 클래스에 따라 구분하는데 필요한 필수적인 데이터를 제거함
	- PCA는 데이터의 분산을 최대화 하는 방향을 찾음. (unsupervised / generative)
	- LDA(Linear discriminant analysis) 혹은 MDA(Multiple Discrimininant Aanlysis) 는 데이터를 각기 다른 클래스로 구분하는데 최적화된 방향을 찾음 (supervised /discriminative)

## Fisher Linear Discrimiant(FLD)

먼저 2 class problem(binary classification) 으로 간주해보면, 데이터는 D 차원에서 하나의 라인으로 투영 된다.
N개의 D차원 데이터 샘플 $x_1, ... , x_N$ 이 $N_1$ 개의 $C_1$ class 샘플과 $N_2$ 개의 $C_2$  class 샘플로 구성되어있다고 하자.
$x$ 를 선형 변환하여 $y$로 나타내는 것을 구성하려고 한다.

$$
y = \mathbf{W^T x}, \quad y_1, ..., y_n \in R^1
$$
그러면 best direction 인 $\mathbf{w}$ 를  찾는 것이 목표가 된다.

$$
\begin{aligned}
\text{class mean } m_i &= \frac 1 N_i \sum_{\mathbf{x} \in c_i} \mathbf{x}
\\ \text{class mean of projected points } \tilde{m}_i &= \frac 1 N_i \sum_{y \in c_i} y = \frac 1 N_i \sum_{\mathbf{x} \in c_i} \mathbf{W^T x}
\\ \text{distance between projected class mean } \vert \tilde{m}_1 - \tilde{m}_2 \vert &= \vert \mathbf{w}^t(m_1 - m_2) \vert
\end{aligned}
$$

투영된 샘플의 흩어짐 정도는 아래와 같이 정의 할 수 있다.

$$
\tilde{s}^2_i = \sum_{y \in c_i} (y - \tilde{m}_i)^2
$$

따라서, 투영된 데이터의 분산은 아래와 같이 정의 할 수 있고,

$$
\frac 1 N (\tilde{s}^2_1 + \tilde{s}^2_2)
$$

따라서 총 클래스간 흩어짐 정도는 $\tilde{s}^2_1 + \tilde{s}^2_2$ 가 되고, 이를 작게 만들어야한다.
목적 함수는 아래와 같이 정의 가능하다.

$$
J(\mathbf{w}) = \frac {\vert \tilde{m}_1 - \tilde{m}_2 \vert^2}{\tilde{s}^2_1 + \tilde{s}^2_2}
$$

여기서 흩어짐 정도는 작게 하고, 투영된 샘플의 클래스 간 거리는 커져야 한다.
투영된 샘플의 평균 거리는 커야지 잘 분리되고, 분산이 작아야 겹치는 부분이 적어지므로 위 식을 직관적으로 이해할 수 있다.

within-class scatter matrix 는 $\mathbf{S}_w = \mathbf{S_1 + S_2}$ 이다.
$\tilde{s}^2_i$ 를 다시 작성하면 아래와 같다.

$$
\begin{aligned}
\tilde{s}^2_i &= \sum_{x \in c_i}(\mathbf{w^T x - w^T m_i})^2
\\ &=\sum_{x \in c_i}\mathbf{w^T(x - m_i) (x - m_i)^T w})
\\ &= \mathbf{w^T S_i w}
\\
\\ \therefore \tilde{s}^2_1 + \tilde{s}^2_2 &= \mathbf{w^T(s_1 + s_2)w} = \mathbf{w^T S_w w} 
\end{aligned}
$$


유사하게 scatter matrix 도 아래와 같이 계산할 수 있다.

$$
\begin{aligned}
\vert \tilde{m}_1 - \tilde{m}_2 \vert^2 & = (\mathbf{w^T m_1} - \mathbf{w^T m_2})^2
\\ &= \mathbf{w}^T\mathbf{(m_1 - m_2)(m_1 - m_2)^T w}
\\ & = \mathbf{w^T S_B w} \text{ , where } \mathbf{S_B} = \mathbf{(m_1 -m_2)(m_1 - m_2)^T}
\end{aligned}
$$

따라서 목적 함수를 다시 작성해보면,

$$
J(\mathbf{w}) = \mathbf{\frac{w^T S_B w}{w^T S_w w}}
$$

이러한 형태는 **generalized rayliegh quotient** 라고도 함.
이 비율을 최대화 하는 것은 분모를 상수화한 상태에서 분자를 최대화 한 것과 동일하다.
예를들어보면 아래와 같다

$$
\underset{\mathbf{w}}{max}\, \mathbf{w^T S_B w} \text{  subject to  } \mathbf{w^T S_w w} = k \text{ ,  } k \text{ 는 상수}
$$

위 식은 라그랑주 승수법(Lagrange multiplier) 로 문제를 풀 수 있다. 

$$
L = \mathbf{w^T S_B w} + \lambda(k - \mathbf{w^T S_W w})
$$

라그랑주 승수법이 L의 gradient가 0 인 것을 이용하여 문제를 푸므로, L 에 대한 gradient를 구하면 아래와 같다.

$$
\begin{gather}
L = \mathbf{w^T(S_B - \lambda S_w)w + \lambda k}
\\ \mathbf{w} \text {에 대하여 미분한다면,}
\\ 2 \mathbf{(S_B - \lambda S_W)w} = 0
\\ \therefore \mathbf{S_B w} = \lambda \mathbf{S_W w} 
\end{gather}
$$

이는 eigenvector, eigenvalue를 구하는 문제이다.
만약 $\mathbf{S_w}$ 가 nonsingular (invertible) 하다면, 문제가 매우 쉽다

$$
\mathbf{S^{-1}_W S_B w} = \lambda \mathbf{w}
$$

여기서 각 $\mathbf{w}$ 와 $\lambda$ 는 $\mathbf{S^{-1}_W S_B}$ 의 고유벡터, 고윳값이다.
그러나 $\mathbf{S_w}$ 가 nonsingluar 한 것은 일반적이지 않다.

## Multiple Discriminant Analysis
위의 Fisher's Linear Discriminant를 여러 클래스 c 와 M 개의 판별함수 (discriminant function)  로 일반화 해보면 아래와 같다.
D 차원의 공간에서 M 차원의 subspace로 투영된다.

Within-class 와 Between-class scatter 행렬은 아래와 같이 정의 된다.

$$
\begin{gather}
\mathbf{S}_w = \sum^c_{i=1} \mathbf{S}_i \text{ where } \mathbf{S_i} = \sum_{\mathbf x \in c_i}\mathbf{(x - m_i)(x - m_i)}^T
\\ \mathbf{S_B} = \sum_{i=1}^c N_i \mathbf{(m_i - m)(m_i - m)^T} \text{ where } \mathbf{m} \text{ is global mean}
\end{gather}
$$

고유 벡터, 고유값을 찾기 위한 일반화 함수는 아래와 같다.

$$
\mathbf{S_B w_i} = \lambda \mathbf{S_W w_i}\,,\, i=1, ... , M\text{ for eigenvalues } \lambda_i
$$

만약 $\mathbf{S_w}$가 full lank라면, 그 해답은 $\mathbf{S_W^{-1} S_B}$ 의 최대 M 개의 고윳값과 그에 해당하는 고유벡터가 된다.

## Fisher face
N 개의 샘플이미지 $\{\mathbf{x}_n\}, n = 1, ..., N\text{ and } \mathbf{x}_n \in R^D$  가 있고, 각 이미지는 c개의 class로 $\{ C_i \}, i = 1, ... , c$   구분된다고 하자.
D 차원의 이미지 공간을 M 차원의 피쳐 공간으로 선형 변환 한다고 할 때, 
이 feature vectors $y_n \in R^M$ 은 아래의 선형 변환으로 구성된다.

$$
y_n = \mathbf{W^T x}_n \text{ where } \mathbf{W} \in R^{D \times M} \text{ onrthnomal한 컬럼을 가지는 행렬}
$$

### Eigenfaces (Review)
전체 scatter matrix 또는 공분산 행렬 $\mathbf{S_T}$ 은 아래와 같이 정의된다.

$$
\begin{gather}
\mathbf{S_T} = \sum_n \mathbf{(x_n - m)(x_n - m)^T}
\\\text{where } m \in R^D \text{는 모든 샘플의 평균}
\end{gather}
$$

선형 변환 $\mathbf{W^T}$ 가 적용된, feature vector $y_n \in R^M, n= 1,...,N$ 의 scatter matrix 는 $\mathbf{W^T S_T W}$ 이다.

PCA에서는 total scatter matrix의 행렬식(determinant) 을 최대화 하는 $\mathbf{W}_{opt}$ 가 선택 된다.

$$
\mathbf{W}_{opt} = \underset{\mathbf{w}}{argmax} \vert \mathbf{W^T S_T W} \vert = [ \mathbf{w_1, w_2, ... , W_M} ]
$$

여기서 $\mathbf{w}_i\, ,\, i= 1, ... , M$ 은 $\mathbf{S_T}$ 의 상위 M개의 고윳값에 대응하는 D차원의 고유벡터 이다.
이 접근 방법에서는 $\mathbf{S_T = S_B + S_W}$ 이므로, between-class 와 within-class 의 scatter가 둘 다 최대화 된다. 

### Fisherfaces
클래스 레이블링된 학습 데이터를 이용해서, 판별을 더 잘 할 수 있는 방향으로 feature space의 차원을 줄이려고 한다.
Class에 대응하는 선형 변환을 통한 차원 축소로, EIgenface 보다 더 나은 성능을 얻는다.
FLD(Fisher's Linear Discriminant)는  between-class scatter와 within-class scatter의 비율을 최대화 하는 $\mathbf{W}$ 를 선택한다.
각각 은 아래와 같다.

$$
\begin{gather}
\mathbf{S_B} = \sum_{i=1}^c N_i \mathbf{(m_i - m)(m_i - m)^T}
\\ \mathbf{S_W} = \sum_{i=1}^c \sum_{x \in c_i} \mathbf{(x_i - m_i) (x_i - m_i)^T}
\\ m_i \text{ 는 클래스 } c_i \text{ 의 평균, } N_i \text{ 는 각 클래스의 샘플 수}
\end{gather}
$$

여기서 $\mathbf{S_W}$ 가 non-singular 라면, $\mathbf{W}_{opt}$ 는 투영된 샘플의 between-class scatter와 within-class scatter의 행렬식의 비율을 최대화 하는 orthonomal 행렬이 선택된다.

$$
\mathbf{W}_{opt} = \underset{\mathbf{w}}{argmax} \mathbf{\frac{\vert w^T S_B w \vert}{\vert w^T S_w w \vert}} = [\mathbf{w_1, w_2, ... , w_M}]
$$

여기서 $\mathbf{w}_i$ 는 상위 M개의 고윳값에 대응하는 $\mathbf{S_B}$ 와 $\mathbf{S_W}$  의 고유벡터 이다. 

$$
\mathbf{S_B W_i} = \lambda_i \mathbf{S_W w_i}, \quad i= 1,...,M
$$

### Fisherfaces 의 과정
1. c 개의 class로 구성된 학습용 이미지 $\mathbf{x}_n$ 을 준비한다. 
2. class의 평균 ($\mathbf{m}_i$) 과 전체 평균 $\mathbf{m}$ 을 구한다.
3. $\mathbf{m}_i - \mathbf{x}$ 을 계산하여, $\mathbf{S}_B$ 를 계산한다. rank($\mathbf{S}_B$) 는 c - 1이다.
	- 클래스 중심 벡터가 전체 평균과의 상대적인 차이로 표현되기 때문에, 독립적으로 표현 가능한 차원이 하나 적기 때문
4. $\mathbf{x} - \mathbf{m}_i$ 를 계산하여, $\mathbf{S}_W$ 를 계산한다. rank($\mathbf{S}_W$) 는 N - c 이다.
5. 아래의 일반화된 고유값/고유벡터 문제를 만든다.
	- $\mathbf{S_B W_i} = \lambda_i \mathbf{S_W w_i}, \quad i= 1,...,M$
	- $\mathbf{S}_B$  의 rank가 c-1 이기 때문에, 최대 c - 1개의 고윳값으로 이루어진다.
	- $\mathbf{S_W}$ 의 rank 가 최대 N - c 이고, 보통 N 이 D 보다 작다. 따라서  $\mathbf{S_W}$ 의 랭크가 그 차원보다 작아 종종 singluar 하여 역행렬이 존재하지 않는다.
6. $\mathbf{S_W}$ 가 singular 한 경우를 극복하기 위해, PCA를 중간에 사용한다.
	- PCA로 feature space $M_{pca}(\leq N-c)$ 차원으로 축소하고, FLD 로 $M_{lda} (\leq c-1)$로 차원을 축소한다.
	- 따라서 아래와 같다.

$$
\begin{gather}
\mathbf{W}^T_{opt} = \mathbf{W}^T_{lda} \mathbf{W}^T_{pca}
\\ \mathbf{W}_{pca} = \underset{\mathbf{W}}{argmax} \vert \mathbf{W^T S_T W} \vert
\\ \mathbf{W}_{lda} = \underset{\mathbf{W}}{argmax} \frac{\vert \mathbf{W^T W^T_{pca} S_B W_{pca} W} \vert}{\vert \mathbf{W^T W^T_{pca} S_W W_{pca} W} \vert}
\end{gather}
$$

### Fisherfaces의 결과
- $Rank (\mathbf{S}_W) = 182 (= N - c), Rank(\mathbf{S}_b) = 25 (=c-1)$ 일 때, 
- $M_{pca} = 25$ 로 차원 축소 후 LDA를 적용하면, PCA와 적용한것과 크게 차이 나지 않음
	- PCA로 많은 정보들이 cut 되어서, 구분에 필요한 정보를 살리지 못함
- $M_pca = 150$ 으로 차원 축소 후 LDA를 적용한 경우, D 보다는 작아 Singularity 문제는 해결하면서도 구분에 필요한 정보를 살려서 생성된 generalized eigenvectors 가 PCA만 했을 때의 eigenvectors와 차이가 남
	- PCA만 한 경우는 low frequency information에 집중하는 것으로 보임.
	- PCA/LDA 한 경우는 High frequency information에 집중하는 것으로 보임
