---
layout: post
title: lec 5,6 Quantization
subtitle: Quantization
date: 2024-11-18 19:11:08 +0900
category: content
tags:
  - lightweight
use_math: true
---
MIT 6.5940 Fall 2024 TinyML and Efficient Deep Learning Computing 을 듣고 정리한 포스트이다.

Bit-width가 작을 수록 적게 에너지를 소모한다.
## Numeric Data types

### Integer
- Unsigned Integer
	- n-bit Range: $[0, 2^n-1]$
- Signed Integer
	- Sign-Magnitude representation
		- n-bit Range: $[-2^{n-1} + 1, 2^{n-1}-1]$
		- 첫 자리는 sign bit로 사용
	- Two's complement representation
		- n-bit Range: $[-2^{n-1}, 2^{n-1}-1]$
		- $000..00 = 0$
		- $100..00 = -2^{n-1}$
		- 다른 자리는 모두 양수이고 그 값을 합산.
### Fixed-Point Number
- 소수 표현, Two's complement representation 적용
- Fraction bit의 수만큼 $2^{-n}$ 을 곱해줌.

$$
\begin{aligned}
&00110001 \\&= (0*(-2^7)+0*(2^6)+1*(2^5)+1*(2^4)+0*(2^3)+0*(2^2)+0*(2^1)+1*(2^0)) * 2^{-4}\\& = 49 * 0.0625 \\ &= 3.0625
\end{aligned}
$$

### Floating-point number
#### IEEE 754, 32bit floating point number
- signbit + 8bit Exponent + 23 bit Fraction(significant / mantissa) 로 표현
	- $(-1)^{\text{sign}}\times (1+ \text{Fraction}) \times 2^{\text{Exponent}-127}$
	- $127(\text{Exponenet Bias}) = 127 = 2^{8-1}-1$
- Subnormal numbers (1): Exponent=(모두 0) = 0
	- Subnormal: $(-1)^{sign} \times \text{Fraction} \times 2^{(1-127)}$, Exponent 파트는 고정
	- 0의 표현
		- 0 00000000 00000000000000000000000 = 0
		- 1 00000000 00000000000000000000000 = 0
	- 가장 큰 subnormal 값
		- 0 00000000 11111111111111111111111 
		  = $(2^{-23}+2^{-22}+...2^{-1})*2^{-126} = (1-2^{-23})*2^{-126}$
- Subnormal numbers (2): Exponent=(모두 1)
	- 무한대 표현
		- 0 11111111 00000000000000000000000 = $\infty$
		- 1 11111111 00000000000000000000000 = $-\infty$
	- NaN 표현
		- - 11111111 ----------------------- = NaN(Not a number)
		- - 는 0 or 1

![image]({{site.url}}/assets/img/quant-1.png)

- Exponent width 는 Range를 Fraction width는 Precision에 상관

|             | Exponent<br>(bits) | Fraction<br>(bits) | Total<br>(bits) |
| ----------- | ------------------ | ------------------ | --------------- |
| IEEE FP32   | 8                  | 23                 | 32              |
| IEEE FP16   | 5                  | 10                 | 16              |
| Google BF16 | 8                  | 7                  | 16              |

- 예제, FP16 1 10001 1100000000
	- sign: -
	- Exponent: $2^4 + 2^0 - 2^{5-1}-1 = 17-15 = 2$
	- Fraction: $2^{-1}+2^{-2} = 0.75$
	- Decimal Answer = $-(1+0.75) * 2^2 = -1.75*4 = -7$

이외에도, Nvidia 의 FP8 (E4M3), FP8(E5M2) 등 다양

### Quantization 이란?
Quantization은 연속적이거나 크기가 큰 값의 집합에서 입력을 제한하여 이산적인 값의 집합으로 변환 하는 과정이다. 
여러가지 quantization 방식이 존재한다.

#### K-means-based quantization
- 저장 공간 - 정수 가중치와 Floating-point 코드북으로 구성
- 연산 - Floating-Point 연산
따라서 저장 공간에만 이점이 있고, 연산에는 이점이 없음

가중치로부터 N개의 중심점(Centroid)을 구하고, 그것으로 코드북을 구성한다. 그리고 weight는 그 코드북의 중심점과 매칭되는 정수 index로 표시한다.

ex) FP32 4x4 가중치
저장공간: 16 x 32b = 512 bit = 64B -> 16 x 2bit + 32bit x 4 = 160bit = 20B (3.2x 작아짐)

따라서 N-bit quantization을 한다고 할 때, 파라미터의 수 $M >> 2^N$  이면,
32bit x M = 32M bit -> N bit x M = NM bit (codebook 은 작아서 무시)
즉 32/N x 작아짐.

-  Fintuning
32bit weight의 gradient를 구하고, 중심점 별로 그룹화 한 후, 그 중심점 값에 그 gradient의 합 x LR 을 빼줌으로 써 fine-tuned centroids(중심점) 을 구할 수 있음

- Accuracy 분석
Pruing + Quantization 한 것이, Pruing only 또는 Qunatization only 보다 더 좋은 효율을 보임

이외에도 허프만 코딩을 적용한 quantization도 존재.

## Linear Quantization
- Storgage: 정수 가중치
- Computation: 정수 연산
둘 다 Save가 있음

- 정의
	- 정수를 실수로 매핑하는 하는 affine mapping
	- 참고, affine transform
		- 선형 변환과 평행 이동을 결합한 변환.
		- $f(x) = Ax + b$
따라서, 

$$
\begin{aligned}
\text{weights(32bit)} &= \text{quantized weights(2bit int)} - \text{zero point(2bit int)} \times \text{scale(32bit float)} \\
&= \text{reconstructed weights (32bit float)}
\\
\\
\underbrace{r}_{\text{floating point}} &= (\underbrace{q}_{\text{Integer}} - \underbrace{Z}_{\text{Integer}})\,\times\,\underbrace{S}_{\text{floating point}}
\end{aligned}
$$

여기서 zero point와 scale을 어떻게 정하느냐가 중요

q의 Bit width에 따라, $q_{min}, q_{max}$ 가 지정됨

| Bit Width | $q_{min}$  | $q_{max}$     |
| --------- | ---------- | ------------- |
| 2         | -2         | 1             |
| 3         | -4         | 3             |
| 4         | -8         | 7             |
| N         | $-2^{N-1}$ | $2^{N-1} - 1$ |
여기서

$$
\begin{aligned}
r_{max} &= S(q_{max}-Z) \\
r_{min} &= S(q_{min}-Z)  \\
r_{max} - r_{min} &= S(q_{max}-q_{min}) \\
\therefore S &= \frac{r_{max} - r_{min}}{q_{max}-q_{min}}
\\
\\
r_{min} &= S(q_{min}-Z) \\
Z &= q_{min} - \frac{r_{min}}{S} \\
\therefore Z &= \text{round}(q_{min} - \frac{r_{min}}{S})
\end{aligned}
$$

그리고 Matrix multiplication을 고려해보면,

$$
\begin{gather}
Y = WX \\
S_Y(q_Y - Z_Y) = S_W(q_W - Z_W) \cdot S_X(q_X - Z_X)\\
q_Y = \frac{S_W S_X}{S_Y}(q_W - Z_W)(q_X - Z_X) + Z_Y \\
q_Y = \frac{S_W S_X}{S_Y}(q_W q_X - Z_W q_X - Z_X q_W + Z_W Z_X) + Z_Y \\
q_Y = \frac{S_W S_X}{S_Y}\underbrace{(q_W q_X - Z_W q_X \underbrace{- Z_X q_W + Z_W Z_X}_{\text{Not runtime}})}_{\text{N bit Int mult, 32bit int add/sub}} + \underbrace{Z_Y}_{\text{N bit Int Add}}
\end{gather}
$$

여기서 $\frac{S_W S_X}{S_Y}$ 는 경험적으로 0에서 1 사이이고, 다음과 같이 표현 할 수 있다. 

$$
\frac{S_W S_X}{S_Y} = 2^{-n} M_0\quad , \text{where } M_0 \in [0.5,1)
$$

이는 $M_0$ 의 Bit shift 이다.

이 때, $W$ 의 $Z$ 즉 $Z_W$ 는 0일 때를 따로 고려 한다. (가중치의 평균이 0일 경우가 많으므로)
그러면 $Z = 0$ 이고, Symmetric Linear quantization이라고 한다. 
참고로, Bit width가 N 이면, $q_{min} = -2^{N-1},\, q_{max} = 2^{N-1}-1$

$$
S = \frac{r_{min}}{q_{min}-Z} = \frac{-\vert r \vert_{max}}{q_{min}} = \frac{\vert r \vert_{max}}{2^{N-1}}
$$

Matrix의 곱 또한 다은과 같이 표현한다.

$$
q_Y = \frac{S_W S_X}{S_Y}(q_W q_X - Z_X q_W) + Z_Y
$$

여기서, bais 까지 고려하면,

$$
\begin{gather}
Y = WX + b \\
S_Y(q_Y - Z_Y) = S_W(q_W - \underbrace{Z_W}_{=0}) \cdot S_X(q_X - Z_X) + S_b(q_b - Z_b)\\
S_Y(q_Y - Z_Y) = S_W  S_X(q_W q_X - q_W Z_X) + S_b(q_b - Z_b) \\ \\
Z_b\text{는 노멀 분포이므로} Z_b = 0,\,S_b\text{는 강제로 }S_W S_X \text{ 사용} \\ \\
S_Y(q_Y - Z_Y) = S_W  S_X(q_W q_X \underbrace{- q_W Z_X + q_b}_{q_{bias} = q_b - Z_X q_W}) \\
S_Y(q_Y - Z_Y) = S_W  S_X(q_W q_X  + q_{bias}) \\
q_Y = \frac{S_W  S_X}{S_Y}(q_W q_X  + q_{bias}) + Z_Y
\end{gather}
$$

이렇게 수식이 간소화된다

$$
q_Y = \underbrace{\frac{S_W  S_X}{S_Y}}_{\text{Shift N-bit int}}
\underbrace{(q_W q_X  + q_{bias})}_{\text{N-bit Int Mult,32bit Int Add}} + 
\underbrace{Z_Y}_{\text{N bit Int Add}}
$$

따라서 $q_Y$  연산은 모두 정수 연산이 된다. 참고로, $q_b, q_{bias}$ 모두 32bit

Convolution layer도 다음과 같이 표현이 가능하다.

$$
\begin{gather}
Y = \text{Conv}(W, X) + b \\
q_Y = \frac{S_W  S_X}{S_Y}(\text{Conv}(q_W,q_X)  + q_{bias}) + Z_Y
\end{gather}
$$

마찬가지로, $q_Y$  연산은 모두 정수 연산

![image]({{site.url}}/assets/img/quant-2.png)

## Post-Training Quantization

### Quantization Granularity
- Per tensor quantization - 전체 tensor 기준
- Per channel quantization - 채널별로 스케일링 factor 공유
- Group quantization (Blackwell archi 이후 유용)
	- Per-vector quantization
	- shared micro-exponent(MX) data type (4bit 이하를 위해)

#### Symmetric Linear quantization on weight
Per-tensor quantization
- 큰 모델에 대해서는 잘 동작하나, 작은 모델에서 정확도 저하
- output channel 간 weight range를 보면, 큰 차이가 있는 경우가 원인
- Per-channel quantization으로 해결

#### Per-channel weight quantization
- 채널 별로 스케일링 factor를 사용하기 때문에, error가 적음
- 다만, 더 많은 scaling factor를 저장해야함.

Group quantization 부분은 생략

### Dynamic range clipping
#### Linear quantization on activations
weights와 달리, activation의 range는 입력에 따라 다양함
floating-point range를 결정하려면, 모델을 배포하기 전에 activation에 대한 통계가 확보 되어야함.
#### Collect activations statistics before deploying the model
1. Type 1: 학습 도중에 측정
	- Exponential moving averages(EMA) 로 측정
	- $\hat{r}^{(t)}_{max, min} = \alpha \cdot r^{(t)}_{max,min} + (1- \alpha)\hat{r}^{(t-1)}_{max,min}$
	- 이는 학습 할 때 측정해야 하는데 요즘은 Pretrained를 받아서 많이 함.
2. Type 2: 소수의 "calibration" 배치를 학습된 FP32 모델에 수행.
	- Calibaration set가 입력을 대표한다고 가정.
	- 해당 입력으로, r의 분포를 구하고 정당한 수준으로 $\vert r\vert_{max}$ 를 구해야함.
		- minimize the mean-square-error  between input X and (reconstructed) quantized inputs Q(X),
		- $\underset{\vert r \vert_{max}}{\text{min}} = \mathbb{E} \left[ (X-Q(X))^2 \right]$
	- 입력을 Gaussian 또는 Laplace 분포로 가정하면, 최적의 clipping values는 다음에서 선택된다.
		- $\vert r \vert_{max} = 2.83b, 3.89b, 5.03b \text{ for } 2, 3, 4\text{ bits}$
		- 그러나 이 경우는 희귀한 경우임
	- 따라서, 다른 방식이 필요한데 정보의 loss를 최소화하는 방법이다.
	- 정수 모델은 오리지날 실수 모델과 같은 정보를 인코딩하기 때문에, 이 정보의 loss 를 최소화 한다.
		- 정보의 loss(loss of information) 는 KL-Divergence로 측정한다.
		- $D_{KL}(P\Vert Q) = \sum^N_i P(x_i) log \frac{P(x_i)}{Q(x_i)}$
	- 또는 MSE가 convex 여서, Newton-Rapshon method로 global optimum 을 구함 (OCTAV int4)


## Quantization-aware Training

![image]({{site.url}}/assets/img/quant-3.png)

학습 과정에서 Full precision 가중치 W는 유지됨
작은 그라디언트들이 정밀도 손실 없이 누적됨
모델이 학습되면, quantized 가중치만 inference 과정에서 사용됨.

이때에 gradients의 역전파는 simulated quantization을 통해서 어떻게 전파 될까?
### Straight-Through Estimator (STE)
Quantization은 이산 값이므로, 미분은 항상 0임. 
그라디언트가 0이므로, 신경망은 아무것도 학습 할 수 없음
Straight-through estimator(STE)는 마치 항등 함수 인 것 처럼, gradient를 그대로 전달한다.

$$
g_W = \frac{\partial L}{\partial W} = \frac{\partial L}{\partial Q(W)}
$$

즉, 손실 L 에 대해 W 에 대한 gradient $\frac{\partial L}{\partial W}$ 를 구하고, Q(W)에 이 gradient를 적용한다.
양자화된 Q(W)가 STE에 의해 업데이트되면, 그에 따라 $S_W$ (scale 값)과 $q_W$ (양자화된 가중치)도 함께 업데이트

Binary/Ternary Quantization은 생략

## Mixed-precision Quantization
레이어 별로, 다른 Bit Widths를 적용하는 것.
RL을 가지고, Hardware와 맵핑하여 automation
