---
title: Saliency Map 논문 리뷰
date: YYYY-MM-DD HH:MM:SS +09:00
categories: [논문 리뷰, ComputerVision]
tags:
  [
    ComputerVision,
    ExplainableAi,
    Saliency Map
  ]
pin: true
math: true
mermaid: true
---




| 태그                 | #ComputerVision #ExplainableAi                                                                 |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| 한줄요약               | Saliency Map                                                                                   |
| Journal/Conference | #NIPS                                                                                          |
| Link               | [Saliency Map](https://www.robots.ox.ac.uk/~vgg/publications/2014/Simonyan14a/)                |
| Year(출판년도)         | 2014                                                                                           |
| 저자                 | K. Simonyan, A. Vedaldi, A. Zisserman                                                          |
| 원문 링크              | [Saliency Map](https://www.robots.ox.ac.uk/~vgg/publications/2014/Simonyan14a/simonyan14a.pdf) |

---

<br/>
<br/>
<br/>

# 핵심요약

- 이미지 분류 모델이 학습한 것을 가시화 하는 방법론.
- 딥러닝 모델을 분석하는 툴로써 사용할 수 있으며, weakly supervised object segmentation에 적용 가능함.

<br/>
<br/>
<br/>

## Introduction

딥러닝 모델은 많은 발전을 통해서 높은 성능을 계속 갱신해가고 있지만, 내부의 operation이 어떤 작용을 하고 있는 지에 대한 분석은 부족한 것이 사실이다. 본 연구는 이미지 분류 모델에 대한 분석을 진행할 수 있는 Saliency map이라는 방법을 제안하며, 이를 통해서 모델의 분석을 넘어 Weakly supervised object segmentation task에도 적용이 가능한 것을 실험을 통해서 보인다.

<br/>
<br/>
<br/>

## Class Model Visualization

이미지 데이터셋에 대해서 학습이 진행된 모델은 어떤 class에 대해서 이미 어느정도의 지식을 가지고 있다. 인간을 예로 들면, 실제 모습을 보지 않더라도 강이지의 모습을 떠올리는 것이 가능하다. 딥러닝 모델에 이러한 아이디어를 접목시킨 것이 Class Model Visualization 방법이다.

이 방식은 Image를 생성하는 작업인데, 딥러닝 모델의 파라미터는 고정을 하고, 입력 이미지의 각 픽셀의 값들을 업데이트하며 아래의 수식을 최대화 하는 방향으로 학습이 진행된다.

![](https://tera.dscloud.me:8080/Images/DataBase/논문_SaliencyMap/1.png)

$S_c(I)$ 는 Image I에 대해서 network가 prediction을 한 class c에 대한 score이다. 따라서 이미지를 업데이트를 진행하는데, target class c라고 모델이 분류하도록 업데이트를 하는 것으로 볼 수 있다. 뒤에 있는 term은 regularization term이다.

![](https://tera.dscloud.me:8080/Images/DataBase/논문_SaliencyMap/2.png)

위 이미지는 Class Model Visualization을 통해서 생성한 이미지이다. 예시로 덤벨에 대한 이미지를 보면, 입력 이미지를 전혀 넣지 않았지만, 덤벨의 형상이 보이는 이미지가 생성된것을 볼 수가 있다. 이러한 이미지들이 학습된 네트워크가 타켓 클래스에 대해서 어떤 이미지를 상상하는가에 대한 답변으로 볼 수 있다.

<br/>
<br/>
<br/>

## Image-specific Class Saliency Visualization

모델을 아주 간단히 경량화하여 생각해보면, output score S_c(I)는 입력 이미지와 weight와 bias가 더해진 형태로 생각할 수 있다.

![](https://tera.dscloud.me:8080/Images/DataBase/논문_SaliencyMap/3.png)

이때, weight는 아래의 식으로 표현할 수 있고,

![](https://tera.dscloud.me:8080/Images/DataBase/논문_SaliencyMap/4.png)

w는 이미지의 각 픽셀이 출력에 미치는 영향으로 볼 수 있다. 따라서 본 연구에서는 w를 saliency map이라고 부르며, 이 saliency map은 입력 이미지의 어떤 부분이 class prediction에 많은 영향을 미쳤는 지를 표현하는 explanation map으로 볼 수 있다.

![](https://tera.dscloud.me:8080/Images/DataBase/논문_SaliencyMap/5.png)
![](https://tera.dscloud.me:8080/Images/DataBase/논문_SaliencyMap/6.png)


위 figure는 입력 이미지에 대한 saliency map의 예시이다. 많이 노이지하지만 object 부분에서 더 높은 값을 가지는 것을 확인할 수 있다. 이러한 saliency map은 weakly supervised segmentation에도 적용할 수 있다.

![](https://tera.dscloud.me:8080/Images/DataBase/논문_SaliencyMap/7.png)

GraphCut colour segmentation saliency map을 masking하면 맨 오른쪽이 결과를 얻게 된다. Saliency map을 통해서 충분히 이용가능한 segmentation map을 얻을 수 있음을 여러 이미지에 대한 실험을 통해서 확인할 수 있다.

<br/>
<br/>
<br/>

## Conclusion

본 논문은 CNN 모델을 분석하기 위한 XAI의 시초가 되는 논문으로, 2개의 saliency map을 제시한다.

한 방법은 이미 학습이 된 모델을 이용하여 target class에 대해 학습한 이미지를 생성하여 모델이 각 클래스에 대해서 어떤 특징들을 학습한 것이지에 대한 분석을 할 수 있고, 다른 방법은 모델의 출력에 대한 입력이미지의 미분맵을 saliency map으로 이용하여, 모델이 prediction을 함에 있어서 어떤 영역에 집중하였는지 를 분석할 수 있으며, 이를 이용하여 weakly supervised segmentation에 적용하여 매우 좋은 성능을 보이는 것을 확인할 수 있다.













