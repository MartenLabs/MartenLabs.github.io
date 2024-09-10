---
title: 4. Auto Encoder (노이즈 제거)
date: 2024-05-04T19:30:00+09:00
categories: [Tensorflow]
tags:
  [
    Tensorflow,
    Auto Encoder,
	fashion MNIST
  ]
pin: true
math: true
mermaid: true
---


## 데이터 소개

- fashion MNIST 데이터 사용

![](https://tera.dscloud.me:8080/Images/Models/3.Multi_layer_classification.png)

이미지는 다음과 같은 것을 보여줍니다.
(a) 28 x 28의 배열에서 의류의 모습이 어떻게 모사되는지 
(b) 각 0-9까지의 다양한 의류 그림의 모습들 

<br/>
<br/>

## 최종 목표 
- noise가 있는 fashion MNIST 이미지를 원래대로 복원하기
- 흑백 이미지와 칼라 이미지의 차이
- 이미지에 noise를 추가하는 방법
- 이미지에 대한 오토인코더식 접근 방법
---


<br/>
<br/>

## 전처리 
``` python
"""
데이터 불러오기 
"""

fashion_mnist = keras.datasets.fashion_mnist
datasets = fashion_mnist.load_data()

(train_images, train_labels), (test_images, test_labels) = datasetss
```


``` python
"""
형 변환 및 normalize 
"""

train_images = train_images.astype(np.float64)
test_images = test_images.astype(np.float64)

train_images = train_images / train_images.max()
test_image = test_images / test_images.max()
```


``` python
"""
흑백 이미지를 color 이미지 shape으로 변환
"""

from skimage import color 

np_change = np.stack([train_images[0], 
					  train_images[0], 
					  train_images[0]], axis = -1)
print(np_change.shape) # (28, 28, 3)

"""---------------------------------------------------------"""

train_images = np.array([
			    color.gray2rgb(img) for img in  train_images])
test_images = np.array([
				color.gray2rgb(img) for img in test_images])

print(train_images.shape, test_images.shape) 
# (60000, 28, 28, 3)
# (10000, 28, 28, 3)
```


``` python
"""
이미지에 노이즈 주입
"""

train_noisy_images = train_images + np.random.normal(
			  loc = 0.5, scale = 0.2, size = train_images.shape)
train_noisy_images[train_noisy_images > 1] = 1.0

test_noisy_images = test_images + np.random.normal(
			  loc = 0.5, scale = 0.2, size = test_images.shape)
test_noisy_images[test_noisy_images > 1] = 1.0
```

<br/>
<br/>

## 모델링 
``` python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Reshape
from tensorlfow.keras.layers import Conv2DTranspose

def AutoEncoder():
	inputs = Input(shape = (28, 28, 3))
	x = Conv2D(filters = 32, kernel_size = 3, strides = 2, 
			   padding = 'same', activation = 'relu')(inputs)
	x = Conv2D(filters = 64, kernel_size =3, strides = 2,
			   padding = 'same', activation = 'relu')(x)
	x = Flatten()(x)
	latent_vector = Dense(units = 10)(x)

	x = Dense(7 * 7 * 64)(latent_vector)
	x = Reshape(target_shape = (7, 7, 64))(x)
	x = Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2,
					    padding = 'same', activation = 'relu')(x)
	x = Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2,
						padding = 'same', activation = 'relu')(x)
	outputs = Conv2DTranspose(filters = 3, kernel_size = 3, 
						padding = 'same', activation = 'sigmoid')(x)

	return Model(inputs, outputs)
```


``` python
model = AutoEncoder()
model.summary()

"""
Model: "model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(None, 28, 28, 3)]       0         
                                                                 
 conv2d_2 (Conv2D)           (None, 14, 14, 32)        896       
                                                                 
 conv2d_3 (Conv2D)           (None, 7, 7, 64)          18496     
                                                                 
 flatten_1 (Flatten)         (None, 3136)              0         
                                                                 
 dense_2 (Dense)             (None, 10)                31370     
                                                                 
 dense_3 (Dense)             (None, 3136)              34496     
                                                                 
 reshape_1 (Reshape)         (None, 7, 7, 64)          0         
                                                                 
 conv2d_transpose_3 (Conv2D  (None, 14, 14, 64)        36928     
 Transpose)                                                      
                                                                 
 conv2d_transpose_4 (Conv2D  (None, 28, 28, 32)        18464     
 Transpose)                                                      
                                                                 
 conv2d_transpose_5 (Conv2D  (None, 28, 28, 3)         867       
 Transpose)                                                      
                                                                 
=================================================================
Total params: 141517 (552.80 KB)
Trainable params: 141517 (552.80 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
"""
```


``` python
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

hist = model.fit(
	train_noisy_images,
	train_images,
	validation_data = (
		test_noisy_images,
		test_images
	),
	epochs = 15,
	batch_size = 128,
	verbose = 1
)
```

