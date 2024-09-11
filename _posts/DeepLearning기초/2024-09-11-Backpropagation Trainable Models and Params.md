---
title: 2. Backpropagation with Trainable Models and Params
date: 2024-09-11T10:30:00+09:00
categories: [DeepLearning기초]
tags:
  [
    DeepLearning기초,
    BackPropagation,
    Chain Rule
  ]
pin: true
math: true
mermaid: true
share: true 
comments: true
---

# Trainable Models and Params

<br/>

딥러닝 모델의 핵심은 학습 가능한 파라미터이다. 이 파라미터들은 Backpropagation이라는 알고리즘을 통해 최적화된다.


| J |
|:---:|
|$\uparrow$|
|$L_{CE}$　　　　　　　　*Loss Calculator*| 
|$\uparrow$|
|$L_{Softmax}$　　　　　　　　　*Classifier*|
|$\uparrow$|
|$L^{[5]}_{Dense}$　　　　　　　　　　*Classifier*|
|$\uparrow$|
|$L^{[4]}_{Dense}$　　　　　　　　　　*Classifier*|
|$\uparrow$|
|$L^{[3]}_{Flatten}$　　　　　*Feature Extractor*|
|$\uparrow$|
|$L^{[2]}_{Conv2D}$　　　　　*Feature Extractor*|
|$\uparrow$|
|$L^{[1]}_{Conv2D}$　　　　　*Feature Extractor*|
|$\uparrow$|
|$X$|

<br/>

위 다이어그램은 전형적인 CNN(Convolutional Neural Network) 구조를 보여준다. 입력 $X$부터 최종 손실 $J$까지, 데이터가 어떻게 흐르는지 볼 수 있다.

<br/>
<br/>
<br/>

# Backpropagation

![](https://tera.dscloud.me:8080/Images/DeepLearning기초/2_Backpropagation_Trainable_Models_and_Params/1.png)

Backpropagation은 모델의 파라미터들을 학습시키는 핵심 알고리즘이다. 

<br/>
<br/>

## 딥러닝의 세 가지 주요 요소

### 1. 손실 함수 (Loss Function)

손실 함수는 모델의 현재 성능을 평가한다. 위 다이어그램에서 $L_{CE}$ (Cross-Entropy Loss)가 이 역할을 한다. 이는:

- **방향 제시**: 모델에게 "이렇게 하면 더 좋아질 거야"라고 알려주는것과 같다.
- **목표 정의**: 우리가 달성하고자 하는 최종 목표를 수치화한다.
- **진행 상황 체크**: 현재 모델이 얼마나 잘 하고 있는지 알려준다.

<br/>

### 2. 모델 (Model)

모델은 입력 데이터를 원하는 출력으로 변환하는 함수이다. 위 구조에서는 여러 층의 Conv2D, Dense 레이어들이 이 역할을 한다:

- **지식 저장소**: 학습된 패턴과 특징을 파라미터로 저장한다.
- **변환기**: 원본 이미지 데이터를 고수준의 특징으로 변환한다.
- **가설 공간**: 가능한 모든 분류 함수 중에서 최적의 함수를 찾는다.

<br/>

### 3. 학습 과정 (Learning Process): Backpropagation의 핵심

Backpropagation은 모델을 실제로 개선하는 단계이다. 이는 위 이미지에서 보이는 역방향 화살표로 표현된다:

- **협력적 최적화**: 모든 파라미터들이 협력하여 전체 손실(J)을 줄인다.
- **책임 분배**: 각 파라미터가 전체 오류에 얼마나 기여했는지 계산한다(편미분).
- **점진적 개선**: 작은 스텝으로 파라미터를 조금씩 조정한다(경사 하강법).

<br/>
<br/>

## Backpropagation의 작동 원리

1. **순전파 (Forward Pass)**: 입력 X가 모델의 각 층을 통과하며 최종 출력을 생성한다.
2. **손실 계산**: 모델의 출력과 실제 정답을 비교하여 손실 J를 계산한다.
3. **역전파 (Backward Pass)**: 손실 J에서 시작하여 각 층을 거꾸로 통과하며 그래디언트를 계산한다.
4. **파라미터 업데이트**: 계산된 그래디언트를 사용하여 각 층의 파라미터를 업데이트한다.

이 과정에서 핵심은 **Chain Rule**이다. 이를 통해 복잡한 모델에서도 각 파라미터의 그래디언트를 효율적으로 계산할 수 있다.

<br/>

$$ \frac{\partial J}{\partial w} = \frac{\partial J}{\partial y} \cdot \frac{\partial y}{\partial w} $$

<br/>

여기서 $w$는 특정 층의 가중치, $y$는 그 층의 출력이다.

<br/>
<br/>

## 실제 레이어 업데이트 과정

Chain Rule을 적용할 때 수학적으로는 중간 항들이 상쇄되는 것처럼 보이지만, 실제 구현에서는 각 레이어의 파라미터들이 개별적으로 업데이트된다. 이 과정을 "역방향 누적"이라고 부를 수 있으며, 다음과 같이 진행된다:

<br/>

1. **수학적 표현**:
   n개의 레이어가 있는 네트워크에서 첫 번째 레이어의 가중치 $w_1$에 대한 손실 J의 그래디언트는 다음과 같이 표현된다:

   $\frac{\partial J}{\partial w_1} = \frac{\partial J}{\partial y_n} \cdot \frac{\partial y_n}{\partial y_{n-1}} \cdot ... \cdot \frac{\partial y_2}{\partial y_1} \cdot \frac{\partial y_1}{\partial w_1}$

<br/>

2. **실제 구현**:
   실제로는 이 긴 체인을 한 번에 계산하지 않고, 각 레이어에서 부분적으로 계산한다:

   a. 마지막 레이어부터 시작하여 이전 레이어로 이동하며 계산
   b. 각 레이어에서 두 가지 값을 계산:
      - 현재 레이어 파라미터에 대한 손실의 그래디언트
      - 이전 레이어의 출력에 대한 손실의 그래디언트

<br/>



### 3층 신경망 모델의 상세 계산 예시

먼저, 간단한 3층 신경망 모델의 구조를 정의하고 초기 파라미터 값을 설정한다:

1. 모델 구조:

   - 입력: $x$ (1x2 벡터)
   - 은닉층 1: 　　　$h_1 = \sigma(W_1x + b_1)$ (3 뉴런)
   - 은닉층 2: 　　　$h_2 = \sigma(W_2h_1 + b_2)$ (2 뉴런)
   - 출력: 　　　　　$y_{pred} = W_3h_2 + b_3$ (1 뉴런)
   - 활성화 함수 $\sigma$:　ReLU $(max(0, x))$
   - 손실 함수:　　　$J = \frac{1}{2}(y_{true} - y_{pred})^2$

<br/>

2. 초기 파라미터 (임의의 값으로 설정):
   
   행 : 뉴런 갯수 (다음 층의 뉴런 수)
   
   열 : 입력 데이터 갯수 (현재 층의 입력 특성 수)
   
   
   - $W_1 = \begin{bmatrix} 0.1 & 0.2\\\ 0.3 & 0.4\\\ 0.5 & 0.6 \end{bmatrix}$,　　　　　$b_1 = \begin{bmatrix} 0.1\\\ 0.2\\\ 0.3 \end{bmatrix}$
  
   - $W_2 = \begin{bmatrix} 0.7 & 0.8 & 0.9 \\\ 1.0 & 1.1 & 1.2 \end{bmatrix}$, 　　$b_2 = \begin{bmatrix} 0.4 \\\ 0.5 \end{bmatrix}$
  
   - $W_3 = \begin{bmatrix} 1.3 & 1.4 \end{bmatrix}$, 　　　　　　$b_3 = \begin{bmatrix} 0.6 \end{bmatrix}$

<br/>

1. 입력 데이터:
   - $x = \begin{bmatrix} 1 \\\ 2 \end{bmatrix}$,　　　$y_{true} = 3$

<br/>

이제 순전파와 역전파 과정을 단계별로 계산한다:

<br/>

#### 순전파 (Forward Pass):

1. 은닉층 1:
   $h_1 = \sigma(W_1x + b_1)$
   $= \sigma(\begin{bmatrix} 0.1 & 0.2 \\\ 0.3 & 0.4 \\\ 0.5 & 0.6 \end{bmatrix} \begin{bmatrix} 1 \\\ 2 \end{bmatrix} + \begin{bmatrix} 0.1 \\\ 0.2 \\\ 0.3 \end{bmatrix})$
   $= \sigma(\begin{bmatrix} 0.5 \\\ 1.1 \\\ 1.7 \end{bmatrix}) = \begin{bmatrix} 0.5 \\\ 1.1 \\\ 1.7 \end{bmatrix}$

<br/>

2. 은닉층 2:
   $h_2 = \sigma(W_2h_1 + b_2)$
   $= \sigma(\begin{bmatrix} 0.7 & 0.8 & 0.9 \\\ 1.0 & 1.1 & 1.2 \end{bmatrix} \begin{bmatrix} 0.5 \\\ 1.1 \\\ 1.7 \end{bmatrix} + \begin{bmatrix} 0.4 \\\ 0.5 \end{bmatrix})$
   $= \sigma(\begin{bmatrix} 2.98 \\\ 4.34 \end{bmatrix}) = \begin{bmatrix} 2.98 \\\ 4.34 \end{bmatrix}$

<br/>

3. 출력층:
   $y_{pred} = W_3h_2 + b_3$
   $= \begin{bmatrix} 1.3 & 1.4 \end{bmatrix} \begin{bmatrix} 2.98 \\\ 4.34 \end{bmatrix} + \begin{bmatrix} 0.6 \end{bmatrix} = \begin{bmatrix} 10.446 \end{bmatrix}$

<br/>

4. 손실 계산:
   $J = \frac{1}{2}(y_{true} - y_{pred})^2 = \frac{1}{2}(3 - 10.446)^2 = 27.72$

<br/>

#### 역전파 (Backward Pass):


1. 출력층:
   $\frac{\partial J}{\partial y_{pred}} = -(y_{true} - y_{pred}) = -(3 - 10.446) = 7.446$
   
   <br/>

   $\frac{\partial J}{\partial W_3} = \frac{\partial J}{\partial y_{pred}} \cdot h_2^T = 7.446 \cdot \begin{bmatrix} 2.98 & 4.34 \end{bmatrix} = \begin{bmatrix} 22.189 & 32.316 \end{bmatrix}$
   
   <br/>

   $\frac{\partial J}{\partial b_3} = \frac{\partial J}{\partial y_{pred}} = 7.446$
   
   <br/>

   $\frac{\partial J}{\partial h_2} = W_3^T \cdot \frac{\partial J}{\partial y_{pred}} = \begin{bmatrix} 1.3 \\\ 1.4 \end{bmatrix} \cdot 7.446 = \begin{bmatrix} 9.680 \\\ 10.424 \end{bmatrix}$

<br/>

2. 은닉층 2:
   $\frac{\partial J}{\partial W_2} = \frac{\partial J}{\partial h_2} \cdot h_1^T = \begin{bmatrix} 9.680 \\\ 10.424 \end{bmatrix} \cdot \begin{bmatrix} 0.5 & 1.1 & 1.7 \end{bmatrix} = \begin{bmatrix} 4.840 & 10.648 & 16.456 \\\ 5.212 & 11.466 & 17.721 \end{bmatrix}$
   
   <br/>

   $\frac{\partial J}{\partial b_2} = \frac{\partial J}{\partial h_2} = \begin{bmatrix} 9.680 \\\ 10.424 \end{bmatrix}$
   
   <br/>

   $\frac{\partial J}{\partial h_1} = W_2^T \cdot \frac{\partial J}{\partial h_2} = \begin{bmatrix} 0.7 & 1.0 \\\ 0.8 & 1.1 \\\ 0.9 & 1.2 \end{bmatrix} \cdot \begin{bmatrix} 9.680 \\\ 10.424 \end{bmatrix} = \begin{bmatrix} 17.200 \\\ 19.242 \\\ 21.285 \end{bmatrix}$

<br/>

3. 은닉층 1:
   $\frac{\partial J}{\partial W_1} = \frac{\partial J}{\partial h_1} \cdot x^T = \begin{bmatrix} 17.200 \\\ 19.242 \\\ 21.285 \end{bmatrix} \cdot \begin{bmatrix} 1 & 2 \end{bmatrix} = \begin{bmatrix} 17.200 & 34.400 \\\ 19.242 & 38.484 \\\ 21.285 & 42.570 \end{bmatrix}$
   
   <br/>

   $\frac{\partial J}{\partial b_1} = \frac{\partial J}{\partial h_1} = \begin{bmatrix} 17.200 \\\ 19.242 \\\ 21.285 \end{bmatrix}$

<br/>

#### 파라미터 업데이트:

학습률 $\alpha = 0.01$을 사용하여 파라미터를 업데이트한다:

1. $W_3 := W_3 - \alpha \frac{\partial J}{\partial W_3} = \begin{bmatrix} 1.078 & 1.077 \end{bmatrix}$

<br/>

2. $b_3 := b_3 - \alpha \frac{\partial J}{\partial b_3} = \begin{bmatrix} 0.526 \end{bmatrix}$

<br/>

3. $W_2 := W_2 - \alpha \frac{\partial J}{\partial W_2} = \begin{bmatrix} 0.652 & 0.693 & 0.735 \\\ 0.948 & 0.985 & 1.023 \end{bmatrix}$

<br/>

4. $b_2 := b_2 - \alpha \frac{\partial J}{\partial b_2} = \begin{bmatrix} 0.303 \\\ 0.396 \end{bmatrix}$

<br/>

5. $W_1 := W_1 - \alpha \frac{\partial J}{\partial W_1} = \begin{bmatrix} -0.072 & -0.144 \\\ 0.108 & 0.015 \\\ 0.287 & 0.174 \end{bmatrix}$

<br/>

6. $b_1 := b_1 - \alpha \frac{\partial J}{\partial b_1} = \begin{bmatrix} -0.072 \\\ 0.008 \\\ 0.087 \end{bmatrix}$

<br/>

이렇게 한 번의 역전파와 파라미터 업데이트가 완료된다. 이 과정을 여러 번 반복하면 모델의 파라미터가 점진적으로 최적화되어 손실이 감소하게 된다.





<br/>

### 실제 코드에서의 구현

   대부분의 딥러닝 프레임워크(예: PyTorch, TensorFlow)에서는 이 과정을 자동으로 처리한다. 각 레이어 클래스는 일반적으로 두 가지 주요 메서드를 가진다:
   - `forward()`: 순전파 연산을 수행
   - `backward()`: 그래디언트를 계산하고 파라미터를 업데이트

<br/>
<br/>

이러한 방식으로, 수학적으로는 중간 항들이 상쇄되는 것처럼 보이지만, 실제 구현에서는 이 과정을 단계적으로 수행하여 각 레이어의 파라미터를 효율적으로 업데이트한다. 이는 계산 효율성과 메모리 사용을 최적화하면서 동시에 정확한 그래디언트 계산을 가능하게 한다.






