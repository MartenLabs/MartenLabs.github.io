---
title: 1. Simple RNN (MNIST)
date: 2024-05-01T19:30:00+09:00
categories: [Tensorflow]
tags:
  [
    Tensorflow,
    Simple RNN,
	MNIST
  ]
pin: true
math: true
mermaid: true
share: true 
comments: true
---

# RNN으로 손글씨 이미지 분류

## 데이터 소개
![](https://tera.dscloud.me:8080/Images/Models/1.Simple_RNN.png)
위의 이미지는 다음과 같은 것을 보여줍니다.
(a) 28x28의 배열에서 3이 어떻게 모사되는지
(b) 각 0-9까지의 다양한 그림의 모습들

<br/>
<br/>

## 최종 목표

- 이전에 배웠던 MNIST fully-conntect network과 CNN classificaion외 RNN식 접근을 배워본다.

- 또한, data augmentation 기법의 기초가 될 수 있는 이미지에 noisy와 같은 변형을 줄 수 있는 방법을 알아본다.

- 이미지에 대한 RNN 접근 방법을 배울 수 있습니다.


<br/>
<br/>

## 전처리
``` python
mnist = keras.datasets.mnist
(trian_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

``` python
train_noisy_images = train_images + 
				np.random.normal(0.5, 0.1, train_images.shape)
train_noisy_images[train_noisy_images > 1.0] = 1.0

test_noisy_images = test_images + 
				np.random.normal(0.5, 0.1, test_images.shape)
test_noisy_images[test_noisy_images > 1.0] = 1.0
```


``` python
from keras.utils import to_categorical

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)
```
---

<br/>
<br/>

## Simple RNN Model
``` python
from keras.layers import SimpleRNN
from keras.layers import Dense, Input
from keras.models import Model


inputs = Input(shape = (28, 28))
x1 = SimpleRNN(64, activation = 'tanh')(inputs)
x2 = Dense(10, activation = 'softmax')(x1)
model = Model(inputs, x2)
```


```python
model.summary()
""""
Model: "model" _________________________________________________________________ Layer (type) Output Shape Param # ================================================================= input_1 (InputLayer) [(None, 28, 28)] 0 simple_rnn (SimpleRNN) (None, 64) 5952 dense (Dense) (None, 10) 650 ================================================================= Total params: 6602 (25.79 KB) Trainable params: 6602 (25.79 KB) Non-trainable params: 0 (0.00 Byte) _________________________________________________________________
""""
```


``` python
model.compile(loss = 'categorical_crossentropy', 
			  optimizer = 'adam',
			  metrics = ['accuracy'])
```


``` python
hist = model.fit(
	train_noisy_images,
	train_labels,
	validation_data = (
		test_noisy_images, 
		test_labels
	),
	epochs = 5,
	verbose = 1
)
```

``` python
res = model.predict(test_noisy_images[:1])

loss, acc = model.evaluate(test_noisy_images, test_labels, verbose = 1)
```