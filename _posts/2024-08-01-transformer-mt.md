---
layout: post
title: TIL, 2024-08-01
subtitle: Transformer를 이용한 기계 번역 구현(2)
date: 2024-08-01 23:07:00 +0900
category: til
tags:
- nlp
---
pytorch의 Transformer 모듈을 이용하여, 기계번역을 구현해보았다.
이전에 구현한 내용에서 학습할 때 버그를 많이 수정하였다.
어느정도 학습이 된후, validation 데이터를 이용하여, 순차 생성해보았을때 그럴싸한 결과물이 보기이 시작하였다.
기존 코드의 문제는 크게 2가지 였고 아래와 같다.

- Loss가 떨어지지 않는 문제
    - 내용 : 학습 epoch이 늘어나도 loss 평균이 0.x로 떨어지지 않음
    - 원인 : target data와 model의 결과 shape가 다름.
        - loss 연산 시, target의 batch, seq_len 순서 변경하여 해결

- 추론 오작동 문제
    - 내용 : train loss, val loss는 점차 떨어지나, 추론시 모든 토큰이 0으로 나옴.
    - 원인 :
        - target mask의 방향을 transpose 상태로 주어, 학습 시 마스킹이 제대로 되지 않음.
        - decoder에 teacher forcing을 위한 input이 주어지지 않았음.
        - model의 input과 output에 대한 토큰 오 설정.
            - encoder input에는 패딩만
            - decoder input 에는 [BOS], [EOS] 토큰 및 패딩 추가
            - decoder output에는 [EOS] 토큰 및 패딩 추가


