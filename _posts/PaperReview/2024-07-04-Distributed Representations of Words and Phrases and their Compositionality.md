---
title: Skip-gram 논문 리뷰
date: YYYY-MM-DD HH:MM:SS +09:00
categories: [논문 리뷰, NLP]
tags:
  [
    NLP,
    Skip-gram 
  ]
pin: true
math: true
mermaid: true
---



| 태그                 | #NLP                                                                                                       |
| ------------------ | ---------------------------------------------------------------------------------------------------------- |
| 한줄요약               | Skip-gram                                                                                                  |
| Journal/Conference | #NIPS                                                                                                      |
| Link               | [Skip-gram](https://papers.nips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html)         |
| Year(출판년도)         | 2013                                                                                                       |
| 저자                 | Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S. Corrado, Jeff Dean                                        |
| 원문 링크              | [Skip-gram](https://papers.nips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf) |

---

<br/>
<br/>
<br/>

# 핵심요약

- Skip-gram 방법의 word vector 퀄리티를 높이고, 학습 속도를 더 빠르게 하는 기법을 제안한다.
- Negative sampling이라는 방법으로 hierarchical softmax 방식을 대체하고 훨씬 더 빠르게 학습하는 방식을 제안한다.
- Air Canada라는 고유 명사는 “Air”와 “Canada”라는 단어 두 개의 의미로 직접 유추가 힘들다는 점에서 착안하여, 수 백만개의 Phrase에서도 word vector들로 의미를 잘 표현할 수 있다는 점을 설명한다.

<br/>
<br/>
<br/>

## Introduction

최근에 제안된 Skip-gram 모델은 엄청나게 많은 단어를 효율적으로 학습하는 방식을 제안하였다. 
Skip-gram 모델은 이전에 사용한 NNLM들과 다르게 dense matrix multiplication를 수행하지 않기 때문에 엄청 효율적인 학습이 가능하다. single-machine에서 최적화를 통해서 100 billion words를 하루안에 학습 가능하게 만들었다.

![](https://tera.dscloud.me:8080/Images/DataBase/논문_Skip-gram/1.png)

또한 벡터 연산으로 단어 간 관계를 파악할 수 있다는 점이 흥미롭다. 특히나 선형 변환으로 표현이 가능하다는 점이 다소 흥미롭다. 예를 들어 ***vec(”Madrid”) - vec(”Spain”) + vec(”France”)*** 는 다른 word vector들보다 ***vec(”France”)*** 에 가깝다.

저자들은 이번 논문에서 Skip-gram 모델을 몇 가지 확장한 방식을 제안한다. 먼저 자주 등장하는 단어들을 subsampling 하는 것으로 학습 속도는 2배에서 10배 정도 향상시키고, 자주 등장하지 않는 단어들에 대한 정확도도 높인다. 또한 Noise Contrastive Estimation(NCE) 방법을 간단하게 변형시킨 방법으로 훨씬 복잡한 hierarchical softmax에 비해 빈번하게 등장한 단어들에 대한 학습 속도와 정확도를 향상시키는 방식을 제안한다.

이렇게 만들어진 word represenation의 결합으로 관용구의 의미를 표현하지 못한다는 한계를 가지고 있었다. 예를 들면 “Boston Globe”라는 신문사의 이름은 vec(”Boston”)과 vec(”Globe”)의 결합으로 표현이 불가능하다. 그러므로, 관용구에 대한 표현도 Skip-gram 모델로 학습을 하면 관용구의 표현력을 더 향상시킬 수 있었다. 아니면 recursive autoencoder 같은 기법과 word representation을 결합하는 것도 하나의 방법이 될 수 있다.

아예 Skip-gram을 학습 시킬 때 phrase도 하나의 단어처럼 만들어서 학습을 시키면 상대적으로 쉽게 학습이 가능하다. 뻔한 유추 작업을 예시로 들면, vec(”Montreal Canadiens”) - vec(”Montreal”) + vec(”Toronto”)는 vec(”Toronto Maple Leafs”)가 된다.

마지막으로, Skip-gram model의 흥미로운 특징을 하나 소개하려고 한다. 예를 들면 vec(”Russia”) + vec(”River”)는 vec(”Volga River”)와 가깝고, vec(”Germany”) + vec(”Capital”)은 vec(”Berlin”)과 가깝다. 우리가 만든 벡터 연산과 기본적인 수치 연산이 언어를 이해하는데 도움이 될 수 있다.

<br/>
<br/>
<br/>

## The Skip-gram Model

Skip-gram 모델의 objective function은 다음과 같다.

![](https://tera.dscloud.me:8080/Images/DataBase/논문_Skip-gram/2.png)

context window size가 $c$ 고, 가운데 있는 단어를 $w_t$ 라고 할 때, 위의 log probability를 최대로 하는 것을 목표로 한다.

Skip-gram 모델에서 사용되는 $p(w_{t+j} \vert w_t)$ 는 다음과 같이 정의된다.

![](https://tera.dscloud.me:8080/Images/DataBase/논문_Skip-gram/3.png)

![](https://tera.dscloud.me:8080/Images/DataBase/논문_Skip-gram/4.png)
![](https://tera.dscloud.me:8080/Images/DataBase/논문_Skip-gram/5.png)
![](https://tera.dscloud.me:8080/Images/DataBase/논문_Skip-gram/6.png)
![](https://tera.dscloud.me:8080/Images/DataBase/논문_Skip-gram/7.png)
![](https://tera.dscloud.me:8080/Images/DataBase/논문_Skip-gram/8.png)
![](https://tera.dscloud.me:8080/Images/DataBase/논문_Skip-gram/9.png)
![](https://tera.dscloud.me:8080/Images/DataBase/논문_Skip-gram/10.png)
![](https://tera.dscloud.me:8080/Images/DataBase/논문_Skip-gram/11.png)



