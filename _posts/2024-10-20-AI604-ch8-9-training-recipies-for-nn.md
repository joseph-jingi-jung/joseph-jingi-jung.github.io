---
layout: post
title: AI604 - CH8/9 Training recipes for neural nets
subtitle: Training recipes for neural nets
date: 2024-10-20 22:58:08 +0900
category: content
tags:
  - vision
use_math: true
---
AI604 수업을 수강 후 정리한 내용이다. Stanford의 CS231n과 맞닿아 있다.

### Activation Functions

#### Sigmoid
- $\sigma(x) = \frac{1}{1 +e^{-x}}$
- 숫자를 0, 1 범위로 압축
- 뉴런과 유사 
- 문제점
	- 포화된 뉴런(Saturated neurons)이 기울기를 죽임
		- 포화 -> 출력이 상한이나 하한에 도달함을 의미. 이 때에 기울기가 0
		- Sigmoid의 출력이 0 중심이 아님
			- 또한 Gradiant가 항상 Postive임
			- input에 따라서 upstream/downstream gradient의 방향이 항상 같음
			- Minibatch가 이를 조금 완화해줌.
		- exp() 연산이 비쌈

#### TanH
- 숫자를 -1, 1 범위로 압축
- 0 중심
- 문제점
	- 포화된 뉴런(Saturated neurons)이 기울기를 죽임

#### ReLU
- $f(x) = max(0,x)$
- 장점
	- 포화 상태가 없음 (Does not saturate)
	- 연산이 효율적임
	- sigmoid와 tanh 대비 거의 6배 빠른 수렴 속도
- 단점
	- 0 중심 출력이 아님
	- 성가신 점 (x <0 에 대하여, 더 이상 업데이트가 없음)
		- Leaky ReLU 같은 것 도입

#### Leaky ReLU
- $f(x) = max(0.01x, x)$
- 장점
	- 포화 상태가 없음 (Does not saturate)
	- 연산이 효율적임
	- sigmoid와 tanh 대비 거의 6배 빠른 수렴 속도
	- x < 0, 지점에서 더 이상 죽지 않음

##### Parametric Recifier (PReLU)
- $f(x) = max(\alpha x, x)$

#### Exponential Linear Unit (ELU)

$$
f(x) = 
\begin{cases}
x & \text{if }x \gt 0 \\
\alpha(\text{exp}(x) -1) & \text{if }x \leq 0 
\end{cases}
$$

- ReLU의 모든 장점 가져옴
- Zero mean output에 좀 더 가까워짐
- Leaky ReLU 대비 음수 포화영역(Negative saturation) 에서 좀 더 노이즈에 강함
- 단점
	- exp 연산이 필요함

#### Scaled Exponential Linear Unit (SELU)

$$
f(x) = 
\begin{cases}
\lambda x & \text{if }x \gt 0 \\
\lambda (\alpha e^x - \alpha) & \text{if }x \leq 0 
\end{cases}
$$

- Scaled version of ELU
- 깊은 네트워크에서 좀 더 잘 동작함
- Self-Normalizing 특성을 가짐. BatchNorm 없이도 깊은 네트워크 학습 가능하게 함

#### Activation functions - Summary
- Just use ReLU
- 짜내야 할 경우 Leaky ReLU, ELU, SELU, GELU 등 써봐라
- sigmoid와 tanh 는 쓰지 말 것 

### Data Preprocessing
- Zero-centered data
	- 평균을 0으로 맞추기
	- ReLU 같은 곳에서 입력에 따라 그라디언트 부호가 정해질 수 있음
		- zero-mean 데이터로 정규화하여, 양수와 음수 부호의 비율을 맞춤(대칭성)
- normalized data
	- 평균 0, 분산 1
- decorrelated data
	- 데이터 상관 관계 제거 (주로 PCA)
	- 따라서 diagonal covariance matrix 가지게 함
- whitend data
	- PCA 후 분산을 고유값으로 나눠 모든 특성의 분산이 1이 되게 조정
	- covariance matrix가 Identity matrix가 되게 함

#### Data processing for images
- subtract the mean image
	- 하나의 평균 이미지
- subtract per-channel mean
	- 3개의 평균 값 이용 평균 0
- subtract per-channel mean and divided by per-channel std
	- 3개의 평균 이용 평균0, 분산 1
- PCA나 whitening 은 잘 쓰지 않음

### Weight Initialization
- Q. 모든 Weight, Bias 가 0 이면?
	- 모든 출력이 0이 되고, 모든 gradients가 같아짐.
	- 대칭성 문제를 해결 할 수 없음(No symmetry breaking). 
		- 동일한 역전파 기울기로 인해 동일한 학습 경로

#### Idea1. small random numbers
- Gaussian. 평균 0, 표준편차 0.01
- 작은 네트워크에서는 잘 동작하나, 깊은 네트워크에서 문제 발생
- tanh 를 activation으로 여러 층을 쌓음
	- activation 값이 평균 0에 작은 표준편차(ex. 0.05)로 만들어지고, 역전파 시에 그 activation의 입력값을 사용하게 되어 gradient가 0에 가까워지게 됨.

#### Idea2. large random numbers
- Gaussian 평균 0, 표준편차 0.05
- tanh를 activation으로 여러 층 쌓음
	- $XW$ 값이 한 방향으로 커지거나, 작아져서 activation의 결과는 -1 또는 1에 가까워짐 (saturated)
	- Local graidients가 0 이 되고, 학습이 어려워짐

#### Xavier initialization
- Gaussian 평균 0, $std = \frac {1} {\sqrt{D_{in}}}$
- Activation이 고르게 scaled 됨. No single peak
- Conv layer에 대해서는 $D_{in} = K^2 * \text{input channel}$
- 유도 과정 : Variance of output = variance of input

$$
\begin{aligned}
y &= Wx \\
y_i &= \sum^{D_{in}}_{j=1} x_jw_j \\
Var(y_i) &= D_{in} Var(x_i w_j)\quad \text{Assume x, w are iid} \\
&= D_{in}* (E[x_i^2]E[w_i^2] - E[x_i]^2E[w_i]^2) \\
& \text{(Assume x, w independent)} \\
&= D_{in}*(Var[x_i] Var[W_i] + E[X_i]^2 Var[W_i] + E[W_i]^2Var[x_i]) \\
& \text{(Assume x, w are zero-mean)} \\
&= D_{in}* Var[x_i]*Var[w_i] \\
&\therefore \text{if }Var(w_i) = \frac{1}{d_{in}}\text{ then } Var(y_i) = Var(x_i)
\end{aligned}
$$

- Activation function을 ReLU로 바꾸면, activation이 0으로 collapse 되는 문제가 발생 (no learning)
	- ReLU correction : $std = \frac {2} {\sqrt{D_{in}}}$

#### Residual Network initialization
- MSRA(He) Initialization
	- ReLU 계열에서 주로 사용되는 초기화.
	- $Var(F(x)) = Var(x)$ 
	- 각 층의 출력 분산이 입력 분산과 같게 유지되도록 하는 목표
- Residual network에서는 첫 conv에는 MSRA(He) 로 초기화하고, 두 번째 conv 에는 0으로 초기화해서 $Var(x + F(x)) = Var(x)$ 가 되게 함.

### Regularization

#### L2 regularization (Weight decay)
- $R(W) = \sum_k \sum_l W^2_{k,l}$

#### L1 regularization
- $R(W) = \sum_k \sum_l \vert W_{k,l} \vert$

#### Elastic net(L1 + L2)
- $R(W) = \sum_k \sum_l \beta W^2_{k,l} + \vert W_{k,l} \vert$

#### Regularization common pattern
- training : 랜덤함을 추가하기
- testing : 랜덤함을 통합하여 주변화 (Marginalize over randomness) 
	- 주변화(Marginalize) : 변수를 통합 또는 합하여 제거하는 과정. 
	- 무작위성으로 인해 발생하는 변화를 통합하여 그 영향을 제거하거나 다루는 것을 의미

#### Dropout
- 특정 뉴론을 0으로 랜덤하게 설정.
- 확률이 Hyperparameter
- 네트워크가 중복된 표현을 가지도록(redundant representation) 강제한다.
- 즉 특성들이 상호 적응(co-adaptation) 하지 않게 한다.
- 다른 해석으로는 파라미터를 공유하는 큰 앙상블 모델
- 각각의 binary mask 가 하나의 모델
- Test time에는 drop하지 않고, dropout 확률로  결과를 나눠줌
- Inverted dropout
	- Drop 과 scale을 학습 할 때 해서, test time에는 아무것도 하지 않게 함.
- FCN에 주로 dropout 적용

#### Data augmentation
- Horizontal Flips
- Random crops and scales
- Color Jitter
	- Simple : Randomize contrast and brightness
	- Complex
		- PCA를 적용하고, color offeset을 주성분으로부터 샘플링
		- training 이미지에 offset을 더 해줌 
- 다양하게 혼합하여 쓸 수 있음

#### DropConnect
- 랜덤 connection을 Drop
	- Weight를 0으로 설정

#### Fractional pooling
- 비정수 배율의 풀링 윈도우의 경우 경계를 정확하게 맞출 수 없음
- 여기에 랜덤 함을 주어, 다양한 변형된 데이터 학습

#### Stochastic Depth
- Training : 일부 residual block을 스킵함 in ResNet
- Testing : 전체 네트워크 사용

#### Cutout
- Training : 특정 이미지 영역을 0으로 설정 (잘라내기)
- Testing : 전체 이미지 사용
- 비고
	- Feature map 에도 적용할 수 있음
	- 작은 데이터셋에서 잘 동작하나, 큰 데이터셋에서는 덜 사용됨

#### Mixup
- Training : 랜덤하게 blend 된 이미지로 학습
- Testing : 원본 이미지 사용
- 비고
	- Sample blend를 이용하고, 이에 따라 Target label 도 변환
		- ex) 고양이 40% 강아지 60%로 블랜딩하면, label 도 cat:0.4, dog:0.6

#### CutMix
- Training : 이미지의 특정 부분을 잘라 붙여서 학습
- Testing : 원본 이미지 사용
- 비고 
	- Blend 대신 랜덤 한 영역에 다른 이미지의 일부 영역을 붙임
	- Pixel의 비율로 Target label 조정.
	- 뭘 잘라서 어디에 붙일지 중요
		- SaliencyMix (세일리언 믹스)
			- 중요한 부분 찾아내서 붙이기

### Learning rate schedules
- Learning rate 해석
	- Very high learning rate : loss exploding
	- high learning rate : 수렴이 너무 초반에 발생. 거기서 머물고 더 나아지지 않음
	- good learning rate : 적당한 시점에서 수렴하고 saturated
	- low learning rate : 정답으로 가는 건 보장하나, 수렴이 너무 느림
- 따라서, Large learning rate에서 시작해서 점점 줄여가야함

#### Learning rate Decay
- Step
	- 고정된 시점에서 고정된 비율로 decay
- Cosine
	- $\alpha_t = \frac 1 2 \alpha_0 (1+ cos(t \pi / T))$
	- cosine 커브에 맞춰 LR decay
- Linear
	- $\alpha_t = \alpha_0(1- t/T)$
	- 일정 비율로 줄이기
- Inverse Sqrt
	- $\alpha_t = \alpha_0 / \sqrt{t}$
	- Inverse sqrt 모양으로 줄어듬
- Constant
	- $\alpha_t = \alpha_0$
	- 여전히 fixed 도 많이 쓰임
	- 모델의 동작 확인
	- 마지막에 Squeeze 할 때 다른 LR Decay 방식을 사용

### Early Stopping
- Validation 의 accuracy 가 줄어 드는 부분 즈음에서 stop!

### Choosing Hyperparameters
- Grid search
	- hyperparmeter grid를 만들어서 평가
- Random search
	- Range 내에서 임의의 hyperparameter 선택
- Common Strategy
	- Step1. check initial loss
		- 모델이 예상대로 잘 동작하는 지 확인
	- Step2. Overfit a small sample
		- 작은 Data(5-10 mini batches)에 대해서 잘 동작하는 지 확인.
		- Loss 가 작게 떨어지면, LR이 너무 작음
		- Loss 가 터지면? LR이 너무 높음
	- Step3. Find LR that makes loss go down
		- 이전 Step에서 사용한 모델을 이용
		- 전체 데이터셋 학습
		- 작은 weight decay 적용.
		- Early strage(~100 iters) 내에서 급격히 loss 줄어드는 LR 찾기
	-  Step4. Coarse grid, train for 1~5 epochs
		- Step3의 LR 기준으로, 여러 LR 과 Weight decay 테스트
		- 적당한 Epoch으로
	- Step5. Refine grid, train longer
		- Step4에서 찾은 Best model로 LR decay 없이 길게 학습
	- Step6. Look at learning curves
		- Train Loss 와 train/val accuracy 확인
		- 손실값 정체(Loss plateaus) 가 발생하면LR decay 적용 해봄
			- 손실값이 줄어들고 있는데 LR decay를 하면 너무 일찍 낮춘 택
		- Accuracy가 계속 증가하면, 더 학습 할 필요가 있음
		- Train / val 갭이 크면, 과적합을 의미
			- 규제를 더 적용하거나, 더 많은 데이터 필요
		- Train / val 갭이 너무 작다면, underfitting을 의미
			- 학습을 더 길게 하거나, 더 큰 모델이 필요.
	- Step7. GOTO step5

### Model Ensembles
- 여러 독립적인 모델을 학습하고, 그 결과를 평균
- 약 2% extra performance를 보여줌 (Downstream task 에 따라 다름)

#### Model Ensembles - Tips and tricks
- 여러 독립적인 모델 학습 대신, 한 모델의 여러 snapshot을 이용하기도 함.
	- DropOut
	- Cyclic learning rate schedules
		- LR을 순환하게 조정해서, Local minima에서 빠져나와 다른 local minima로 수렴하게 만듦
	- Polyak averaging
		- 학습 중 매 반복(iteration) 에서 계산된 모델의 가중치를 이전 반복에서 얻은 가중치들과 평균화 하여 최종 가중치를 얻음
		- EMA와 유사

### Transfer learning

#### Transfer learning with CNNs
- pre-trained model 의 마지막 FC 레이어만 빼고, 이를 feature extractor로 사용 
- feature extractor 부분은 Freeze 하고 그 외 부분 학습
- Bigger Dataset
	- Fine-tuning 해서 사용
	- 전체 weight를 조정
- Some tricks
	- Fine tuning 전에 feature extraction을 이용해서 먼저 학습해볼 것 
	- Fine tuning 할 때는 더 작은 LR을 사용
	- 가끔은 feature extraction의 lower layer는 freeze 하고 사용
		- Lower layer는 lcoal visual feature를 다룸
			- Orientation, strong edge etc.
			- 일반적인 이미지 모두 가지는 성질

|                            | Dataset <br>simliar to <br>ImageNet | Dataset <br>very different <br>from ImageNet    |
| -------------------------- | ----------------------------------- | ----------------------------------------------- |
| Very <br>little<br>data    | Feature extraction 사용               | 문제 발생<br>여러 다른 stage에서<br>linear classifier 테스트 |
| quite<br>a lot of <br>data | 일부 레이어만 Fine tune                   | 많은 레이어를 Fine tune                               |

- Transfer learning은 이미 만연함(pervasive)
	- LiDAR sematic segmentation case
		- 2D RGB dataset 학습한 모델을 pretrained로 이용
		- 3D data와 2D data 간의 큰 모달리티(modality) 갭이 있지만, 여전히 pre-trained model이 효과적임
