---
layout: post
title: Online/Incremental Learning with PCA
subtitle: Merging and splitting eigenspace models
date: 2024-09-28 23:10:00 +0900
category: content
tags:
  - vision
use_math: true
---

## Online/Incremental Learning, Merging and splitting eigenspace models

### Why learning online?
- 실제 데이터는 지속적으로 변화가 있음
- tight budget 상태에서의 학습
### Computational considerations
- PCA : O($n^3$) - eigenvalue decomposition. 

### Incremental PCA
- Batch VS Incremental
	- In batch computation : 모든 관찰을 eigenspace model 계산에 사용
	- In incremental computation : 존재하는 eigenspace model을 새로운 관측치로 업데이트
- 요구사항
	- 평균의 변화를 다룰 수 있어야 함.
	- 여러 관측치를 한번에 더 할 수 있어야함
- Pros and cons
	- Pros 
		- 모든 관측치를 한번에 필요로 하지 않음. 따라서 필요한 저장공간을 줄일 수 있고, 거대한 문제를 다룰 수 있게 만듬
		- 모든 관측치를 한번에 처리 할 수 있다하더라도, batch 연산 대비 훨씬 빠르게 연산 할 수 있음
	- Cons
		- 정확도 문제가 있을 수 있음. 몇 개의 증분 업데이트만 이루어질 때 부정확성이 작게 있을 수 있음.

### Eigenspace models and notations
- N 데이터 벡터, $x \in R^D$의 공분산 행렬은 아래와 같다.

$$
S = \frac 1 N \sum^N_{n=1}(\mathbf{x}_n - \mathbf{\bar x}) (\mathbf{x}_n - \mathbf{\bar x})^T, S \in R^{D \times D}
$$

- PCA는 공분산 행렬을 아래와 같이 분해 할 수 있다.

$$
\begin{gather}
if \quad N = 1,\, S\mathbf{u} = \mathbf{\lambda u}, S \in R^{D \times D}, \mathbf{u} \in R^{D \times 1}
\\ if \quad N > 1, N = n,\, S[ \mathbf{u_1\cdots u_n}] = [\lambda_1\mathbf{u}_1 \cdots \lambda_n\mathbf{u}_n ], \mathbf{u}_i  \in R^{D \times 1}
\\ S[ \mathbf{u_1\cdots u_n}] = [\mathbf{u}_1 \cdots \mathbf{u}_n ] 
\begin{bmatrix}
\lambda_1 & & \\
& \ddots & \\
& & \lambda_n
\end{bmatrix}
\\ S\mathbf{u} = \mathbf{u} \Lambda
\\ S = \mathbf{u \Lambda u}^T 
\\ \therefore S \approxeq \mathbf{P \Lambda P}^T
\\ \text{ where P 는 첫 d 개 고유벡터 컬럼들, } P \in R^{D \times d}
\\ \Lambda \text{는 첫 d 개 고윳값 대각 행렬, } \Lambda \in R^{d \times d}
\end{gather}
$$

### Incremental PCA
- Problem setting
	- 입력: 두 eigenspace models $\{\mathbf{u}_i, N_i, \mathbf{P}_i, \mathbf{\Lambda}_i\}_{i=1,2}$
	- 출력: 두 데이터가 합쳐진 eigenspace models $\{\mathbf{u}_3, N_3, \mathbf{P}_3, \mathbf{\Lambda}_3\}$
- 합쳐진 평균(1st order)

$$
\mathbf{\mu}_3 = (N_1 \mathbf{\mu}_1 + N_2 \mathbf{\mu}_2 ) / N_3
$$

- 합쳐진 공분산(2nd order)

$$
\mathbf{S}_3 = \frac{N_1}{N_3} \mathbf{S}_1 + \frac{N_2}{N_3} \mathbf{S}_2  + \frac{N_1 N_2}{N_3} \mathbf{(\mu_1 - \mu_2)(\mu_1 - \mu_2)^T} 
$$

- $\mathbf{P}_3$ 의 계산

$$
\begin{gather}
\mathbf{P}_3 = \mathbf{\Phi R} = h([\mathbf{P}_1, \mathbf{P}_2, \mathbf{\mu}_1 - \mathbf{\mu}_2]) \mathbf{R}
\\ \Phi \text{ 는 합쳐진 공분산 행렬을 충분히 span 하는 orthonomal matrix}
\\ \text{i.e, the sufficient spanning set,}
\\ \mathbf{R} \text{은 회전 행렬, }
\\ h \text{는 orthnormalization 함수, e.g. 그람슈미트 직교화, QR Decomposition}
\end{gather}
$$

- 여기서 $[\mathbf{P}_1, \mathbf{P}_2, \mathbf{\mu}_1 - \mathbf{\mu}_2]$ 는 concatenate 를 의미($d_1 + d_2 + 1$). 이는 아직 orthgonal 하지 않음 
- 따라서 이를 orthnormalization 과정을 거쳐, 서로 독립적이고 직교하게 만듬
- 각 $\mathbf{P}_1, \mathbf{P}_2$ 는 데이터셋1, 2를 sufficient spanning 하는 set이다.
- 이 둘과 $\mu_1 - \mu_2$ 를 추가로 더해서, basis vector set을 구성하고, 이는 all three dimensional data vectors로 span.
- 이것은 아직 방향이 정해져있지 않기 때문에, 방향은 Rotation Matrix $\mathbf{R}$ 로 정해짐.
	- Data variance를 최대화 하는 방향으로 구함

- $\mathbf{P}_3 = \mathbf{\Phi R}$ 을 이용해서, eigenproblem을 더 작은 eigenproblem 으로 만든다.

$$
\begin{gather}
\mathbf{S}_3 \approxeq \mathbf{P_3 \Lambda_3 P_3^T}
\\ \Phi\text{를 이용한 선형 변환을 해도 공분산 구조를 유지 할 수 있으므로, }
\\ S_3 \text{ 양쪽으로 } \Phi^T \text{ 와 } \Phi \text{ 로 선형변환해줌}
\\ \rightarrow \mathbf{\Phi^T S_3 \Phi} \approxeq \mathbf{\Phi^T P_3  \Lambda_3 P^T_3 \Phi}
\\ = \mathbf{\Phi^T \Phi R  \Lambda_3 R^T \Phi^T \Phi} = \mathbf{R  \Lambda_3 R^T}
\end{gather}
$$
- 여기서 $\mathbf{\Phi^T S_3 \Phi}$ 를 가지고 있는 값으로 연산할 수 있고, $\Phi$ 의 차원이 $d_1 + d_2 + 1$ 로  D 보다 훨씬 작으므로, 더 효율적으로 $\mathbf{R}$과 $\mathbf{\Lambda}_3$ 를 decompose 할 수 있다. 
- 배치 모드에서 eigenvector를 구할 때 $O(min(D, N_3)^3$ 의 연산이 필요하다
- 반면 Incremetnal PCA를 사용하면 $O((d_1 + d_2 + 1)^3)$으로 계산량이 줄어든다.


