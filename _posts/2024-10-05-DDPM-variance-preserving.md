---
layout: post
title: DDPM Variance preserving
subtitle: DDPM Variance preserving
date: 2024-10-05 13:05:00 +0900
category: content
tags:
  - vision
use_math: true
---

## DDPM의 Variance preserving

DDPM에서 **variance preserving**의 의미는 시간이 지나면서 분산이 증가하거나 변하지 않고 일정하게 유지되는 것
 각 스텝에서 $x_t$​의 분산이 원래 데이터 $x_0$​의 분산과 일정한 관계를 유지하는지 확인

$$
x_t = \sqrt{1-\beta_t} x_{t-1} + \epsilon_t,\quad\epsilon_t \sim N(0, \beta_t I)
$$

이 과정의 반복으로 $x_t$ 를 $x_0$ 에 대한 함수로 표현하면

$$
x_t = \sqrt{\alpha_t} x_0 + \sqrt{1- \alpha_t} \epsilon, \quad \epsilon \sim N(0,I)
$$

여기서 $\alpha_t = \prod^t_{i=1}(1-\beta_i)$ 는 시간 t까지의 노이즈 스케일링

$x_t$의 분산을 계산해보면, 

$$
\begin{aligned}
Var(x_t) &= Var(\sqrt{\alpha_t}x_0 + \sqrt{1- \alpha_t} \epsilon) 
\\ &= \alpha_t\,Var(x_0) + (1- \alpha_t)\, Var(\epsilon)
\\ &= \alpha_t \cdot 1 + (1- \alpha_t) \cdot 1 \quad(x_0 \text{는 정규화된 데이터로 분산이1})
\\ &= 1 
\end{aligned}
$$

따라서 $Var(x_t)$ 의 분산은 항상 1로 유지됨을 확인 할 수 있음

