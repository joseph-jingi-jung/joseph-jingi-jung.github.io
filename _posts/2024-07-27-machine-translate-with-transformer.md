---
layout: post
title: TIL, 2024-07-27
subtitle: Transformer를 이용한 기계 번역 구현(1)
date: 2024-07-27 22:55:00 +0900
tags:
- nlp
---
pytorch의 Transformer 모듈을 이용하여, 기계번역을 구현해보았다.
아직 완전히 완성하진 못하였으나, 학습이 되는 것 까진 확인하였다.
- 구현 내용
    - AIHUB의 한영 말뭉치 데이터 전처리 및 토큰화
    - PositionalEncoding 구현
    - PaddingMask 구현
    - nn.Transformer를 이용한 모델 구현
    - train 코드 구성.
- todo
    - colab에서 학습 가능하게 수정.
    - valdiation과 test 시에 디코딩 스탭에 맞춰 텍스트 생성
    - BELU score 구현


