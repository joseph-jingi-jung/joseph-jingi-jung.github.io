---
layout: post
title: lec 3,4 Pruning and Sparsity
subtitle: Pruning and Sparsity
date: 2024-11-18 19:02:08 +0900
category: content
tags:
  - lightweight
use_math: true
---
MIT 6.5940 Fall 2024 TinyML and Efficient Deep Learning Computing 을 듣고 정리한 포스트이다.

## Neural network pruning
일반적으로 프루닝을 아래와 같이 수식화 한다.

$$
\begin{gather}
\underset{W_p}{\text{argmin}}L(x; W_p) \\
\text{subject to} \\
\Vert W_p \Vert_0 \le N
\end{gather}
$$

여기서 L은 뉴럴넷 학습의 목적 함수를 나타낸다.
$x$ 는 입력, $W$는 원본 가중치, $W_p$ 는 프루닝 된 가중치
$\Vert W_p \Vert_0$ 는 $W_p$ 중 0이 아닌 가중치의 수를 계산하고, N은 그 목표 값이다.

프루닝의 목적은 중복되는 synapses와 neurons 를 줄여서 네트워크를 줄이는 것이다.

![image]({{site.url}}/assets/img/pruing-1.png)

푸루닝을 통해 0 주변의 weight를 일부 제거한다. 
그 비율에 따라 accruacy drop 이 발생하고, Pruning 된 결과의 Finetuning을 통해 accuracy를 어느정도 복원한다.
절대적으로 적은 량의 weight를 사용한다면, accruacy drop은 불가피하다.

## Pruning in the industry
### Hardware support for sparsity
Sparse Matrix를 효율적으로 연산하기 위한 Hardware support 가 존재한다.
- EIE
- 2:4 sparsity in A100 GPU - 2x peak performance, 1.5X measured BERT speedup
- AMD의 XILINX

## Pruning Granularity
- Fine-grained / Unstructred 
	- 어떤 위치에 상관없이 프루닝. 따라서 runing ratio가 최대
	- 가속하기 어려움 (Not GPU friendly)
- Corase-grained / Structured
	- 덜 유연한 푸르닝 선택
	- Condense 할 수 있음
	- 가속하기 쉬움 (단순히 더 작은 크기의 행렬)

![image]({{site.url}}/assets/img/pruning-2.png)

- Convolution 레이어는 4가직 차원으로 구성됨
	- $C_{in}, C_{out}, k_h, k_w$
	- 더 다양한  푸루닝 세분화 정도(granularities) 를 제공
	- 종류
		- Fine-grained
		- Pattern-based pruning
		- Vector-level Pruning
		- Kernel-level pruning
		- Channel-level pruning

![image]({{site.url}}/assets/img/pruning-3.png)


- Fine-grained pruning
	- Flexible pruning indices
	- 압축 효율이 좋음
	- 가속에는 특별한 하드웨어가 필요함(eg. EIE)
- Pattern-based Pruning: N:M sparsity
	- M개 중 N개 압축 ex(2:4) - 50% sparsity
	- 이 경우 NVIDIA의 Ampere GPU 부터 2배 정도의 속도 효율이 있음
	- 보통 정확도를 유지함
- Channel-level pruning
	- More regular, less degree of freedom
	- 채널 수를 줄이기 때문에 직접적인 가속을 가져옴
	- 단점으로는 압축 효율이 낮음
	- Uniform shrink와 Channel Prune 이 존재
		- Channel Prune은 layer 마다 적합한 sparsity ratio를 찾아야함
		- 강화학습(AMC; Automatic model compression) 통해서 sparcity 찾음

## Pruning criterion
### Selection of Synapses to prune
네트워크의 파라미터를 제거할 때, 덜 중요한 파라미터부터 제거할 수록, 푸루닝 된 네트워크가 더 나은 성능을 보임

#### Magnitude-based pruning
가중치의 절대값이 더 클수록 더 중요한 가중치로 판단.
- element-wide pruning일 때
	- $Importance = \vert W \vert$
- row-wise pruning,
	- L1-norm : $Importance = \sum_{i \in S}\vert w_i \vert$
	- L2-norm : $Importance = \sqrt{\sum_{i \in S} \vert w_i \vert^2}$
#### Scaling-based pruning
Scalning factor는 각 채널의 output의 multiplier이며, 학습 가능한 파라미터이다. 이 scalining factor를 기준으로 프루닝 함
scaling factor가 작은 채널을 프루닝.
이는 batch normalization layer로 부터 가져올 수 도 있음

$$
z_0 = \gamma \frac{z_i - \mu_{batch}}{\sqrt{\sigma^2_{batch}} + \epsilon} + \beta
$$

여기서 $\gamma$ 값을 scaling factor로 보고, 프루닝

### Second-order based pruning
프루닝 네트워크의 loss function 간 에러를 최소화.
테일러 급수를 사용하여 그 에러를 근사

$$
\begin{aligned}
\delta L &= L(\mathbf{x; W}) - L(\mathbf{x; W_p = W - \delta W}) \\
&= \sum_i g_i \delta w_i + \frac 1 2 \sum_i h_{ii} \delta w_i^2 + \frac 1 2 \sum_{i \neq j} h_ij \delta w_i \delta w_j + O (\Vert \delta W \Vert^3) \\
& where\,g_i = \frac{\partial L}{\partial w_i}, h_i,j = \frac{\partial^2 L}{\partial w_i \partial w_j} \\
&= \sum_i g_i \delta w_i + \frac 1 2 \sum_i h_{ii} \delta w_i^2 + \frac 1 2 \sum_{i \neq j} h_ij \delta w_i \delta w_j + \cancel{O(\Vert \delta W \Vert^3)} \\
&by \, L\text{이 거의 2차 함수 임을 가정}\\
&= \cancel{\sum_i g_i \delta w_i} + \frac 1 2 \sum_i h_{ii} \delta w_i^2 + \frac 1 2 \sum_{i \neq j} h_ij \delta w_i \delta w_j + \cancel{O(\Vert \delta W \Vert^3)} \\
&by \, \text{네트워크가 수렴하므로, 첫번째 텀 무시} \\
&= \cancel{\sum_i g_i \delta w_i} + \frac 1 2 \sum_i h_{ii} \delta w_i^2 + \cancel{\frac 1 2 \sum_{i \neq j} h_ij \delta w_i \delta w_j} + \cancel{O(\Vert \delta W \Vert^3)} \\
&by \, \text{파라미터 삭제로 발생ㅇ하는 에러는 각각 독립으로 cross term 무시} \\
& \therefore \delta L_i = L(\mathbf{x; W}) - L(\mathbf{x; W_p \vert } w_i = 0) \approx \frac 1 2 h_{ii} w_i^2\, where\, h_{ii}=\frac{\partial^2 L}{\partial w_i \partial w_j}\\
& \therefore importance_{w_i} = \vert \delta L_i \vert  = \frac 1 2 h_{ii} w_i^2s
\end{aligned}
$$

### Selection of Neurons to prune
덜 중요한 뉴런을 제거할 수록, 프루닝되 네트워크의 성능이 좋아짐.
- 뉴런 푸르닝은 coarse-grained weight pruning 임
	- Linear Layer의 neuron pruning
	- Convolution layer의 channel pruning

#### Percentage of zero based pruning
RELU activation이 0을 만들어냄
weight의 크기처럼, 배치 간 채널 별 0의 비율을 기준으로 뉴런의 중요성을 측정
0의 비율이 더 작을 수록, 더 중요한 뉴런

Regression-based pruning은 생략

## Pruning Ratio
### Finding pruning ratios
각 레이어 별로 sensitivity를 분석해야함.
어떠한 레이어들은 더 sensitive하고 (e.g. first layer), 어떤 레이어들은 redundant함.
프루닝 비율을 결정하기 위해, 이 sensitivity를 분석해야함.

#### Analyze the sensitivity of each layer
1. 모델에서 하나의 레이어를 선택함.
2. pruning ratio를 특정 범위 내에서 선택하여, Prune
3. 각 pruning ratio 별로 정확도의 변화를 관찰
4. 모든 레이어에 대해서 해당 과정을 반복
5. 전체 pruning rate이 원하는 수준이 되는 threshold를 선택

![image]({{site.url}}/assets/img/pruning-4.png)

그러나 위 과정은 최적의 해가 아님. 각 레이어간의 상호작용을 고려하지 않음
위 과정을 최적화할 방법이 필요함

#### AMC: AutoML for Model Compression
AMC는 강화 학습 문제를 풀기 위해 다음과 같이 설정함.
- 상태: 레이어 인덱스, 채널 수, 커널크기, FLOPs 등을 포함하는 features
- 액션: 연속된 수(Pruning ratio) $\alpha \in [0,1)$
- 에이전트: DDPG agent
- 보상: -Error or $-\infty$

#### NetAdapt
룰 기반 반복적이고 점진적인 방법. 전체적인 리소스 제한에 만족하는 각 레이어별 프루닝 비율을 찾는 것이 목표

각 iteration 별로, 특정량 $\Delta R$ 의 latency를 줄이는 것이 목표
- 모든 레이어 각각에 대하여, latency reduction이 $\Delta R$ 을 만족할 만큼 프루닝
- 짧게 fine-tune 하고 정확도 측정
- 모든 레이어 중 가장 정확도가 높은 푸루닝된 레이어를 선택. 
- 전체 제약을 만족하면서 할 때까지 반복.
- 마지막에 최종적으로 Long-term fine-tune을 통해서 정확도 복원

## Fine-tuning / Training
Pruning 후에는 모델의 성능이 떨어진다.
Fine-tuning을 통해, pruned 네트워크의 정확도를 복원하는 것을 돕고, 더 높은 pruning ratio를 가질 수 있게 한다.
일반적으로 파인튜닝에 쓰이는 Learning rate 는 원 학습의 learning rate의 1/100 또는 1/10 이다.
점진적 반복으로 pruning 하는 경우가 효과가 좋다.

![image]({{site.url}}/assets/img/pruning-5.png)

### Regularization
신경망을 학습 하거나 프루닝된 네트워크를 학습 할 때, Regularization이 도움을 준다.
L1, L2 Regularization은 non-zero weight 의 크기를 작게 만들어 주고 이는 pruning에 도움을 준다.

Sparsity를 위한 시스템/하드웨어 지원 부분은 생략.
