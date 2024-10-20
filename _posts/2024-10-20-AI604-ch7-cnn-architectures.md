---
layout: post
title: AI604 - CH7 CNN Architectures
subtitle: CNN Architectures
date: 2024-10-20 22:58:07 +0900
category: content
tags:
  - vision
use_math: true
---
AI604 수업을 수강 후 정리한 내용이다. Stanford의 CS231n과 맞닿아 있다.

### AlexNet
- ImageNet challenge 에서 16.4 Error rate 기록
- 227x 227 input, 5 Conv Layers, Max Pooling, 3 FCN, ReLU
- Conv의 연산량 계산
	- 입력 : 3 x 227 x227
	- Conv filter : 64 x 3 x 11 x 11
	- stride 4, pad 2
	- 출력
		- C : 64
		- H/W : (227 - 11 + 4) / 4 + 1 = 56
	- Memory
		- Num of output elem : 64 x 56 x 56 = 200,704
		- Bytes per elem : 4
		- KB = 200704 x 4 / 1024 = 784 KB
	- params
		- weight : 64 x 3 x 11 x 11 
		- bais : 64 ($C_{out}$)
		- Num of weights : 64 x 3 x 11 x 11 + 64 = 23,296
	- flop
		- flop - number of floating point operations (multiply+add)
		- (Num of output elem) x (ops per output elem)
		- $(c_{out} \times H' \times W') * (C_{in} \times K \times K)$
		- 200,704 x (3 x 11 x 11) = 72,855,552 = 73 MFLOP
- Pooling 연산량 계산
	- 입력 : 64 x 56 x 56
	- 출력 : 
		- kernel 3, stride 2
		- floor((W-K) / S + 1) = floor((56-3)/2 + 1) = 27
		- 64 x 27 x 27
	- Memory
		- Num of output elem : 64 x 27 x 27
		- KB = 64 x 27 x 27 x 4 / 1024 = 182.25 KB
	- params
		- 0
	- flop
		- (Num of output elem) x (ops per output elem)
		- $(c_{out} \times H' \times W') * (K \times K)$
		- (64 x 27 x 27) x (3 x 3) = 419,904 = 0.4 MFLOP
- FC 연산량 계산
	- 입력 : 9216
	- 출력 : 4096
	- Memory
		- 4096 x 4 / 1024 = 16
	- params
		- $C_{in} \times C_{out} + C_{out}(\text{Bias})$
		- 9216 x 4096 + 4096 = 37,725,832
	- flop
		- 9216 x 4096 = 37,748,736 = 38 MFLOP
- Memory/Params/FLOPS로 본 Trends
	- 대부분의 메모리 사용량은 초기 레이어에서 사용됨
	- 대부분의 파라미터 수는 FC Layer에서 사용됨
	- 대부분의 FLOPS 는 Conv layer에서 사용됨
### VGGNet
- Design rules
	- 모든 Conv를 3x3 kernel 에 stride1, pad 1 적용
	- 모든 max pool 은 2x2 kernel 에 stride 2 적용
	- Pool 이후 채널 수 를 2배로
- VGG의 선택
	- receptive field를 키우는 것이 목적
		- 5 x 5 커널 하나와 3 x 3 커널 둘은 Receptive field 동일
		- 커널 크기를 키우는 대신 더 쌓는 것을 택함
		- Params 수, FLOPs 모두 줄어듬
	- Max pool 하고 채널 두배로
		- 메모리는 줄이고 ( C x 2H x 2W -> 2C x H x W)
			- 4HWC -> 2 HWC
		- Params 수는 늘어남 ( 3 x 3 kernel, C(C_in) to C(C_out)  -> 3 x 3 kernel, 2C to 2C )
			- $9C^2$ -> $36 C^2$
		- FLOPS 는 그대로
			- $(c_{out} \times H' \times W') * (K \times K)$
			- (C x 2H x 2W) x (C x 3 x 3) -> (2C x H x W) x (2C x 3 x 3)
			- $36HWC^2 \rightarrow 36HWC^2$
		- 각 공간 해상도에서의 Cov 층들이 동일한 양의 연산을 수행
- AlexNet 대비 훨씬 큰 네트워크 -> 7.3 수준으로 error rate 줄임
### GoogLeNet
- 효율성에 많은 기여
	- 파라미터수, 메모리 사용량, 계산량을 줄임
- Stem network
	- 시작 부분에서 공격적으로 다운 샘플링해서 연산량을 줄임 (AlexNet, VGG 모두 대부분의 연산이 초반에서)
	- 초반에 큰 커널과 큰 Max Pooling 이용 (224 to 28)
- Inception module
	- 여러 병렬적 branch를 가진 Local Unit
	- 이 모듈이 네트워크에 여러 번 반복
	- 1x1 Bottle neck 레이어를 둬서, 비싼 conv 연산 전에 채널 수를 줄여줌
- Global Average Pooling
	- 비싼 FC 레이어 대신, Global Average pooling을 이용하여 공간 정보를 collapse 함.
- **Deeper networks with computational efficiency**
	- 22 Layers
	- Efficient Inception module
	- avoid exepensive FC layers
	- 12x less params than AlexNet
	- 27x lesse params than VGG-16
	- 6.7 Error rate
### ResNet
- 100+ layer를 어떻게 학습할지 고민
	- 단순히 깊게만 쌓았더니, shallow model 보다 test 결과가 나쁘게 나옴 
		- Overfitting 된게 아닐까?
	- 그런데 Training error 도 shallow model 보다 성능이 떨어짐 (Underfitting)
	- 일반적인 생각
		- deeper model이 shallow 모델을 따라 할 수 있지 않을까?
			- Shallow 모델을 복사해서 앞에 두고, identity 레이어를 추가하면 될 것 같은데
			- 그러면 적어도 shallow model 만큼의 성능은 나와야 할것 같은데.
		- 가설 : Optimization 문제 일 것이다.
			- Deeper model 일 수록 최적화 하기 어렵고, 특히 Identity function을 학습하지 못할 것이다.
		- 해결책 :
			- 네트워크가 identity function을 학습하기 쉽게 바꾸면 좋지 않을까?
			- **Shortcut(Skip connection)**
- Residual networks
	- 여러 residual block을 쌓은 네트워크
	- VGG와 유사하게 구성 하고, 중간 중간에 skip connection을 추가
	- Network를 stage로 나누고, 각 stage의 첫 블록에서 해상도를 반으로 줄이고, 채널 수를 두 배로 늘림
	- GoogleNet 처럼 공격적인 Stem 을 둠
	- 마지막에 GAP 대신 하나의 Single FCN 을 사용.
	- BottleNeck Residual block
		- Basic 은 (3x3, c->c) conv, (3x3, c->c) conv, shortcut 으로 구성
			- $9HWC^2 + 9HWC^2  = 18HWC^2$
		- Bottleneck 은 
			- (1x1, 4c->c)conv, (3x3, c->c)conv, (1x1, c->4c)conv
			- 1x1 conv를 이용해 채널을 줄였다가 다시 늘림
			- $4HWC^2 + 9HWC^2 + 4HWC^2 = 17HWC^2$
		- Depth는 늘리되 계산량은 줄임
#### Improving Residual network
- ReLU 위치의 변화
	- Original : ReLU after residual
		- conv -> batch -> Relu -> conv -> batch -> shortcut -> Relu 순
		- shortcut 이후에 Relu 를 수행하기 때문에, 결과는 non-negative가 되고  Identity가 안됨
	- "Pre-activation" ResNet Block
		- ReLU inside residual
		- batch -> Relu -> conv -> batch -> relu -> conv -> shortcut
	- 약간의 성능 향상이 있으나, 많이 쓰이진 않음
#### ResNet Training recipe
- Batch Norm after conv
- Xavier initialization
- SGD+Momentum(0.9)
- LR: 0.1 시작, validation error가 정체될 때 10으로 나눔
- mini-batch: 256
- Weight decay : 1e-5
- No dropout
#### Model Summary
- Inception-v4 : Resnet + Inception
- VGG : 가장 큰 메모리와 가장 큰 연산량
- GoogLeNet : 매우 효율적임
- AlexNet : 적은 연산 수, 많은 파라미터
- ResNet : 간단한 디자인, 적당한 효율성, 높은 정확도
#### Improving ResNet (2)
- "BottleNeck" residual block 을 Inception model 처럼 병렬로 합침
- ResNeXt
	- Grouped convolution 적용
	- depth에 대한 개선이 없을 때는, width를 넓혀 보는 것이 답이 될 수도 있다.
- Squeeze-and-excitation networks
	- residual block에 "Squeeze-and-excite" 브랜치를 추가
	- GAP -> FC -> FC -> Sigmoid 를 통해 일종의 activation 추가.
	- C x 1 x 1 -> c/16 x 1 x 1 -> c x 1 x 1 -> c x 1 x 1
	- 마지막 sigmoid 를 residual block의 output에 scale 하는 부분이 Attention 과 유사하게 동작.
### Densely Connected Neural network
- Dense blocks는 feedforward 방향의 모든 다른 레이어에 연결됨
	- 기울기 소실 문제 완화(Alliviates vanishing gradient)
	- 특성 전파 강화(strengthens feature propagation)
	- 특성 재사용 촉진(Encourages feature reuse)
- Identity branch 와 유사히 보이나, 좀 더 vanishing grad 문제에 focus
- 요즘은 Transformer 가 standard
- CNN -> 로컬 피처가 중요하다는 inductive bias
### MobileNets
- 효율성 높이는 방향