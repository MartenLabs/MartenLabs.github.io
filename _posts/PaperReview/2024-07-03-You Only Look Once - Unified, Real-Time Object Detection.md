---
title: YOLOv1 논문 리뷰
date: YYYY-MM-DD HH:MM:SS +09:00
categories: [논문 리뷰, ComputerVision]
tags:
  [
    ComputerVision,
    ObjectDetection,
    YOLOv1
  ]
pin: true
math: true
mermaid: true
---





| 태그                 | #ComputerVision                                            |
| ------------------ | ---------------------------------------------------------- |
| 한줄요약               | Yolo                                                       |
| Journal/Conference | #CVPR                                                      |
| Link               | [Yolo](https://arxiv.org/abs/1506.02640)                   |
| Year(출판년도)         | 2015                                                       |
| 저자                 | Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi |
| 원문 링크              | [Yolo](https://arxiv.org/pdf/1506.02640)                   |


---

<br/>
<br/>
<br/>

# 핵심 요약

- 객체 탐지를 위한 bounding box 탐색과 객체 분류 문제를 하나의 문제로 통합하는 Yolo 모델을 제시한다.
- 통합된 구조를 통해 기존의 object detection 보다 더욱 빠른 예측을 수행할 수 있다.

<br/>
<br/>
<br/>

## Introduction

기존의 객체 인식 시스템은 이미지 내에서 분류해야하는 객체를 탐색하는 방법과, 탐색한 객체를 실제로 분류하는 모델이 구분되어 있었다. 각각의 모델을 따로 학습하여야하는 만큼, 학습 파이프라인이 느리고 최적화가 어려운 단점이 있었다.

![](https://tera.dscloud.me:8080/Images/DataBase/논문_Yolo/1.png)

논문은 위 그림과 같이 하나의 모델만으로 객체가 존재할법한 다양한 bounding box 에서의 분류를 동시에 수행한다. 단 한번만 이미지를 참조하기에, 논문에서 제시하는 모델의 이름은 YOLO(You Only Look Once)라 부른다. 해당 모델의 장점은 다음과 같다.

- Detection을 단일한 regression 문제로 치환하여 빠른 속도를 보장한다.
- 예측을 수행할 때 이미지 전체를 사용하여 추론하여, contextual 한 정보를 습득한다.
- 객체의 일반화된 표현을 학습할 수 있다.

또한, YOLO는 SOTA 성능을 달성하지 못하고 작은 물체를 식별하는데 어려움을 겪으나, 큰 물체는 실시간으로 빠르게 탐색할 수 있음을 강조한다.

<br/>
<br/>
<br/>

## Unified Detection

YOLO 는 전체 이미지로부터 Bounding box 와 box 의 class 를 동시에 예측한다.

![](https://tera.dscloud.me:8080/Images/DataBase/논문_Yolo/2.png)

이를 위해 먼저 입력 이미지를 S*S 크기의 격자로 구분합니다. 각 격자는 bounding boxes B와 각 box 에 대한 신뢰 점수를 예측한다. 해당 점수는 박스안에 객체가 존재할 확률과, 박스가 실제 박스와 겹치는 정도(intersection over union, IOU) 를 곱하여 계산한다. ($Pr(Object)∗IOU^{truth}_{pred}$)

각각의 bounding box 는 x,y,w,h,신뢰도로 구성된다. (x,y)는 grid cell의 경계를 기준으로 한 box의 중심 좌표이며, 너비와 높이 전체 이미지를 기준으로 한다. 신뢰도는 예측된 box와 ground truth box 사이의 IOU를 나타낸다.

또한 각 grid cell은 C 개의 class 에 대한 조건부 확률 $Pr(Class_i\vert Object)$ 을 예측한다. 이 때, bounding box 의 갯수 B 에 관계 없이 grid cell 당 한 번의 예측만 수행한다.

테스트 진행시에는 조건부 확률에 개별 박스의 신뢰도 예측을 곱하여, Box 별 class 신뢰도를 계산한다. 테스트 데이터셋은 Pascal VOC 를 사용하였다.

<br/>
<br/>
<br/>

## Network Design

![](https://tera.dscloud.me:8080/Images/DataBase/논문_Yolo/3.png)

GoogLeNet 모델을 기반으로 하되, inception module 을 간소화된 형태로 교체하여 사용한다. 위 그림과 같이 Image Feature 를 추출하는 24개의 Conv Layer 와, 예측을 수행하는 2개의 Fully Connected Layer 로 구성된다.

빠른 객체 인식의 가능성을 확인하기위해 conv layer 를 9개만 사용하고 Filter 갯수를 줄인 Fast YOLO 모델 또한 사용한다.

<br/>
<br/>
<br/>

## Training

ImageNet 1000 데이터셋을 사용하여 앞의 20개 Conv layer 를 Pre-training 한다. 다음으로, detection 을 수행할 수 있도록 4개의 conv layer 와 2개의 fc layers 를 추가한니. 이 때, 세밀한 객체도 인식할 수 있도록 입력 해상도를 224X224에서 448X448로 높힌다.

마지막 FC layer 는 클래스 확률과 bounding box 좌표를 예측한다. x,y,w,h 값은 [0,1] 사이 값을 가지도록 normalization 한다.

활성 함수는 마지막 Layer 를 제외하고 Leaky ReLU(0.1) 를 사용한다. 또한 손실 함수는 학습의 편의성을 위해 SSE(sum squared error)를 사용하지만 다음과 같은 문제점이 존재한다.

- Average Precision 을 최대화하려는 모델의 목표와 일치하지 않음
- 분류 오차와 localization 오차에 동등한 가중치 부여
- 대부분의 grid cell은 객체가 포함되지 않아 신뢰 점수가 0이 될 가능성이 높아, gradient 가 사라질 수 있음
- 큰 박스와 작은 박스의 오차를 동일하게 다룸

이를 해결하기 위해

- bounding box 좌표 예측의 loss를 증가시키고, 객체를 포함하지 않는 boxes에 대한 예측 신뢰도의 loss 기여를 감소하기 위해 $\lambda_{coord}=5, \lambda_{noobj}=0.5$ 를 사용하여 조정한다.
- 큰 박스에서의 작은 편차가 작은 박스에서 작은 편차보다 덜 중요하게 여겨질 수 있도록, 기존의 w,h 를 사용하지 않고, bounding box 의 w,h 의 제곱근을 예측한다.

YOLO 는 grid cell 당 여러개의 bounding box predictor 가 존재한다. 논문은 객체당 한 개의 bounding box 가 할당될 수 있도록, IOU 가 가장 높은 한 개의 box predictor 를 선택해 해당 객체를 “책임”지고 예측하도록 한다.

학습시 사용하는 loss function 은 아래와 같다.

![](https://tera.dscloud.me:8080/Images/DataBase/논문_Yolo/4.png)

- 첫번째 항 : Object가 존재하는 grid cell i의 predictor bounding box j에 대해 x,y의 loss를 계산한다.
- 두번째 항 : Object가 존재하는 grid cell i의 predictor bounding box j에 대해 w,h의 loss를 계산한다.
- Object가 존재하는 grid cell i의 predictor bounding box j에 대해, confidence score의 loss를 계산한다.
- Object가 존재하지 않는 grid cell i의 predictor bounding box j에 대해, confidence score의 loss를 계산한다.
- Object가 존재하는 grid cell i에 대해 conditional class probability의 loss 계산

loss function 은 객체가 grid cell 안에 존재하지 않는 경우에 대해서는 객체 분류 오차(조건부 확률)를 계산하지 않습니다. 또한, bounding box 를 “책임”지는 box predictor 의 오차만을 사용한다.

기타 학습 디테일은 아래와 같다.

- PASCAL VOC 2007, 2012 데이터셋 사용
- 135 epochs, batch size : 64, momentum : 0.9, decay : 0.0005
- learning rate : 0.001~0.01 (~75 epoch, epoch 따라 증가) → 0.001 (~105 epoch) → 0.0001 (~135 epoch)
- dropout(rate:0.5), data augmentation : random scaling/translation/exposure/saturation

<br/>
<br/>
<br/>

## Inference

앞서 말했듯, Yolo 는 단일 네트워크 만으로 테스트 이미지에 대한 예측을 수행합니다. PASCAL VOC 에서는 이미지당 98개의 bounding boxes 및 확률을 한번에 예측해낸다.

이 때, Non-maximal suppression 이라는 방법을 사용하여 객체당 한 개의 Box 만을 할당한다. 이를 통해, mAP(mean Average Precision)를 2~3% 증가시킬 수 있다.

<br/>
<br/>
<br/>

## Limitation of YOLO

Grid Cell 당 하나의 클래스만 예측하므로, Cell 안에 여러개의 작은 물체가 존재하는 경우 분류하기 어렵다.
데이터로부터 Bounding box 형태를 학습하므로, 일반적이지 않은 형태의 bounding box 를 결정하기 어렵다.

Bounding box 의 크기가 달라지는 경우에도 loss 가 동일하여, 작은 box의 localization 이 부정확해질 수 있다.

<br/>
<br/>
<br/>

## Experiments

### Comparison to Other Real-Time Systems

![](https://tera.dscloud.me:8080/Images/DataBase/논문_Yolo/5.png)

일반적인 영상은 30FPS(Frame Per Second) 로 촬영하므로, 이를 기준으로 실시간 처리가 가능한지를 나눈다. 실시간 처리가 가능한 모델을 기준으로, Fast YOLO 는 가장 빠르면서도 두번째로 높은 성능(mAP)을 보였으며, YOLO 는 가장 높은 mAP를 달성했다.

<br/>

### VOC 2007 Error Analysis

아래와 같은 기준으로 PALCAL VOC 2007 예측 결과를 분석하여 비교한다.

- Correct : class 정답, IOU > 0.5
- Localization : class 정답, 0.1 < IOU < 0.5
- Similar : 유사한 class, IOU > 0.1
- Other : class 오답, IOU > 0.1
- Background : 모든 객체에 대해서 IOU < 0.1

![](https://tera.dscloud.me:8080/Images/DataBase/논문_Yolo/6.png)

YOLO 는 기존 최고 성능 모델(Fast R-CNN)과 비교하였을 때 Localization Error 가 크고 많은 비중을 차지한다. 반면 Background Error 는 훨신 적은 비중을 차지하며, 전체 이미지를 전역적으로 파악하여 객체와 배경을 용이하게 구분해냈음을 강조한다.

<br/>

### Combining Fast R-CNN and YOLO

![](https://tera.dscloud.me:8080/Images/DataBase/논문_Yolo/7.png)


YOLO 의 낮은 Background error 를 활용할 수 있도록, Fast R-CNN 과 YOLO 를 ensemble 한 모델을 검증한다. 단독 모델로 성능이 더 높은 다른 모델을 ensemble 한 경우보다, YOLO 를 결합하였을 때 더 성능이 높아지는 것을 확인할 수 있다. 또한, YOLO는 빠르게 구동할 수 있으므로, 결합 모델의 연산 속도는 Fast R-CNN 만을 연산하는 속도와 비슷하다.

<br/>

### VOC 2012 Results

![](https://tera.dscloud.me:8080/Images/DataBase/논문_Yolo/8.png)

PASCAL 2012 데이터셋에서의 성능을 비교한 결과이다. Fast R-CNN + YOLO 모델이 일부 class 에서 높은 점수를 달성하였다. 
또한, YOLO 단독으로는 실시간 모델임에도 불구하고 VGG-16을 사용한 R-CNN 과 비슷한 성능을 달성하였다.

<br/>

### Generalizability : Person Detection in Artwork

![](https://tera.dscloud.me:8080/Images/DataBase/논문_Yolo/9.png)

YOLO 모델의 일반화 성능을 확인하기 위해, PASCAL VOC 로 학습한 모델을 다른 데이터셋으로 검증하여 비교한다. 일반 사진이 아닌 예술작품으로 구성된 Picasso / People-Art 데이터셋으로 검증하였을 때, YOLO 는 다른 모델에 비하여 성능이 크게 감소하지 않았으며 가장 높은 성능을 보인다.

![](https://tera.dscloud.me:8080/Images/DataBase/논문_Yolo/10.png)

는 이미지에 실제로 YOLO를 적용한 사례이다. 학습한 자연 이미지와 pixel 차원에서는 다르지만 전체적인 형태는 유사한 예술작품의 객체도 원활하게 인식하는 것을 확인할 수 있다.

<br/>
<br/>
<br/>

## Conclusion

객체 탐지와 객체 분류를 동시에 수행할 수 있는 YOLO 모델을 제시한다. 통합된 파이프라인으로 빠른 객체 인식이 가능하며, 실시간 객체 인식 문제에서 SOTA 를 달성하였다.




