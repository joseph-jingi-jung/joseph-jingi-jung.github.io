---
layout: post
title: TIL, 2024-09-21
subtitle: DDPM 코드 구현 마무리 및 Convex optimization
date: 2024-09-21 23:07:00 +0900
category: til
tags:
- general
use_math: true
---

오랜만에 TIL을 작성해본다.
한동안 작성하지 못하였는데, 다시 습관을 들여봐야겠다.

- DDPM 구현의 마무리
    - 만들면서 배우는 생성 AI의 Diffusion 코드를 참조하여 DDPM 코드 구현을 마무리하였다.
    - Forward process와 Denoising 과정을 볼 수 있는 코드를 구현하여, 디버깅에 용의하게 하였다.
    - EMA와 L2 Loss를 적용하여, 조금 더 성능이 올라가게 유도하였다.
- Convex 최적화의 수강
    - Convex 최적화 강의 초반부를 수강하였다.
    - Convex 문제를 정의하고, Gradient Descent와 Newton's Method를 통해 최적화 문제를 푸는 방법을 알게되었다.
    - Newton's Method가 GD의 Alpha를 구한 다는 설명과, $A^TA$가 역행렬을 가질때, Least-squares solution 으로 이어지는 부분이 인상 깊다.

