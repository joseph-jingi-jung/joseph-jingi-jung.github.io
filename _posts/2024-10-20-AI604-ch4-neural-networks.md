---
layout: post
title: AI604 - CH4 Neural networks
subtitle: Neural networks
date: 2024-10-20 22:58:04 +0900
category: content
tags:
  - vision
use_math: true
---
AI604 수업을 수강 후 정리한 내용이다. Stanford의 CS231n과 맞닿아 있다.

- Linear classifier에는 취약점이 있음
	- 싱글 template per class의 경우, 같은 class여도 여러 mode 존재하는데 대응하기가 어려움
- non-linear로 Feature transform을 해보자!
	- Feature transform 이후, feature space에서 Linear classifier로 구별이 가능해짐

### Image Features

#### Image Features: Color histogram
- texture와 위치 정보를 무시하고, 이미지 전체에 대한 color histogram. Color 를 여러 bin으로 나누고 각각 count
- Global feature

#### Image Features: Histogram of Oriented gradients (HoG)
1. 각 픽셀에서의 edge 방향과 강도를 계산
2. 8x8 구역으로 이미지를 나눔
3. 각 region 별로, edge strength로 weighted 된 edge directions 들의 히스토그램을 계산
- 예제
	- 320x240 이미지를 40x30 구역으로 나눔.
	- 각 구역 별로 9개의 방향이 존재
	- feature vector는 30x40x9
		- 어떤 Patch 는 모든 엣지가 약함 (focus out 된 배경)
		- 대각선 엣지가 있는 경우 대각선 방향에 강한 direction
		- 어떤 부분은 모든 방향으로 강한 direction 가짐
	- 이는 Texture와 위치 정보를 가져옴. 작은 이미지 변화에 강함

#### Image Features: Bag of Words (Data-Driven!)
- Step1 : Build codebook
	- 임의로 패치를 추출하고, 그 패치를 모아 visual words codebook을 생성
- Step2 : Encode Images
	- visdual word로 이미지를 재구성. (이미지에서 해당 visual word가 등장하는 빈도수로 histogram 생성)

#### Integrated Image Features
- Color Histogram, HoG, Bag of Words를 concat 하여 하나의 Image feature로 표현

#### 2011년 ImageNet Chanllenge
- 이미지 별로 10k 개의 패치를 추출
- SIFT를 이용한 128차원, 컬러 히스토그램 이용한 96 차원 데이터를 PCA이용해 64차원으로 차원 축소.
- SVM 을 SGD로 학습

### Neural Network
- 2-layer Neural network

$$
f = W_2 \text{max}(0, W_1 x)
$$

- 첫 번째 레이어 : bank of templates
- 두 번째 레이어 : recombine templates
	- 다른 템플릿들을 한 클래스의 다양한 모드를 커버하는데 사용

#### Activation Functions
- 비선형성을 주입

#### Space Warping
- Matrix의 곱을 선형 변환으로 보면, space warping 임
- 여기서 Activation Function을 통해, 선형 변환된 공간에서 선형 구분을 할 수 있게 만듦

#### Universal Approximation
- 4 개의 hidden units로 bump 함수를 만들 수 있음
- 4 K 개의 hidden units로 K개의 bumps의 합을 만들 수 있음
- 이러한 bumps로 어떠한 함수든 근사할 수 있음

#### Convex Functions
- 어떠한 함수 $f : X \in \mathbb{R}^N \rightarrow \mathbb{R}$ 이 convex 이면, 모든 $x_1, x_2 \in X, t \in [0, 1]$ 에서 아래를 만족한다

$$
f(tx_1 + (1-t)x_2) \leq tf(x_1) + (1-t)f(x_2)
$$

- Convex 함수는 최적화 하기 쉽고, 이론적으로 global minium에 수렴함을 보장함.
- Linear classifier는 convex 함수임 즉 global optimal이 존재
- Neural net의 loss는 convex 처럼 보이지만, 종종 명확하게 non-convex임.

