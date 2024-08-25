---
layout: post
title: 만들면서 배우는 생성 AI, 3장
subtitle: 변이형 오토 인코더(VAE)
date: 2024-08-14 15:14:00 +0900
category: content
tags:
  - vision
use_math: true
---

# Takeaway

- 오토인코더(Autoencoder)
  - 어떤 항목의 인코딩과 디코딩 작업을 수행하도록 훈련된 신경망
  - 출력이 가능한 원본 아이템에 가까워지게함.
  - 인코더는 디코더가 정확하게 재구성할 수 있도록 가능한 많은 정보를 내포한 임베딩(embedding) 벡터를 구성.
  - **원본 임베딩이 없는 벡터를 이용하여 새로운 아이템 생성할 수 있으므로 생성 모델로 사용**
  - 예시에서는 Fashion MNIST 이미지를 2차원으로 임베딩
    - Test 데이터셋의 임베딩을 시각화의 분석
      - 아이템 별로 퍼짐 정도가 고르지 않음
      - 분포가 비대칭이고 경계가 정해져 있지 않음
      - 포인트가 거의 없는 색상 사이에는 간격이 큼
    - Test 데이터셋의 임베딩을 이용한 생성의 문제점
      - 정의한 공간에서 균일하게 포인트하면, 특정 아이템을 디코딩할 확률이 더 높음
      - 분포가 정의되지 않아, 어떤식으로 포인트를 선택해야하는지 알 수 없음
      - 원본 이미지가 인코딩 되지 않는 구멍이 존재.
  - 문제점 해결을 위해 변이형 오토 인코더 필요
- 변이형 오토인코더(Variational autoencoder)
  - 오토인코더는 각 이미지가 잠개 공간의 한 포인트에 직접 매핑
  - **VAE는 각 이미지가 잠재 공간에 있는 포인트 주변의 다변량 정규 분포에 매핑**
  - 다변량 표준 정규분포(multivariate standard normal distribution)
    - 표준 정규 분포 개념을 1차원 이상으로 확장
    - 그 중에 평균 벡터가 0 이고 공분산 행렬이 단위 벡터인 다편량 분포
  - **인코더는 각 입력을 평균 벡터와 분산 벡터로 매핑** (차원간 상관관계X $\rightarrow$ 공분산 고려X)
    - z_mean 과 z_log_var 정의
    - point $z= z\_mean + z\_sigma *epsilon$
    - $z\_sigma = exp(z\_log\_var * 0.5)$
    - $epsilon \sim N(0, I)$
  - 재매개변수화 트릭(Reparameterization trick)
    - 표준 정규 분포에서 epsilon을 샘플링한 다음 특정 평균과 분산을 갖도록 샘플을 수동으로 조정.
    - back propagation을 가능하게 만들어 중요
  - $z\_mean$과 $z\_log\_var$ 로부터 샘플링된 $z$가 latent vector
  - VAE 에서는 reconstruction loss 외에도 KL divergence가 추가됨.
    - KL divergence
      - 한 확률 분포가 다른 분포와 얼마나 다른지를 측정
      - 따라서 VAE에서 평균 $z\_mean$과 분산이 $z\_sigma$ 인 정규 분포가 표준 정규 분포(평균 0, 분산 0)와 얼마나 다른지를 측정.
        $$ kl_loss = -0.5 \* sum(1 + z_log_var - z_mean^2 -exp(z_log_var)) $$
      - 모든 차원의 $z\_mean$과 $z\_log\_val$ 가 0일 때, loss 최소화
    - KL divergence 의 효과
      - 잠재 공간에서 포인트를 선택할 때 사용할 수 있는 잘 정의된 분포(표준 정교 분포)를 가지게 함.
      - 포인트 군집 사이에 큰 간격이 생길 가능성을 줄임(모든 인코딩 된 분포가 표준 정규 분포에 가깝게 강제하였기 때문)
    - _참고로, Loss 를 평균 할 경우, 이미지가 평균화 되게 생성됨_
  - VAE 분석
    - CelebA 데이터셋의 attribute index를 기준으로 feature vector를 추출 (해당 index latent vector의 평균)
    - Vector를 latent space에서 빼거나 더 했을 때, 해당 feature(smiling, glasses 등)가 더해지거나 빼지는 것을 확인 할 수 있다.
    - 두 백터 사이의 조합으로 자연스럽게 얼굴 합성 또한 가능하다.
