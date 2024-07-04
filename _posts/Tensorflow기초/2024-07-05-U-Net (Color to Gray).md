---
title: 6. U-Net (Color to Gray)
date: YYYY-MM-DD HH:MM:SS +09:00
categories: [Tensorflow]
tags:
  [
    Tensorflow,
    U-Net,
	Color to Gray
  ]
pin: true
math: true
mermaid: true
---

## 데이터 소개 

- portrait 데이터로 유명한 PFCN dataset 사용

![](https://tera.dscloud.me:8080/Images/Models/4.UNet_Segmentation.jpg)

이미지는 다음과 같은 것을 보여준다.
- 800 x 600의 사람 portrait 이미지 
- 사람 영역에 대한 흑백 portrait 이미지 
- pfcn_original 
	- 원본 800 x 600 이미지들 
- pfcn_small 
	- colab용 100 x 75 이미지들

<br/>
<br/>

## 최종 목표 
- 작게 줄인 PFCN 데이터를 이용해 사람 영역 추출 
- 코앱에 구글 drive 연동 
- 큰 사진을 작게 줄이기 
- 이미지에 대한 오토인코더식 접근 방법 
- 칼라 사진을 흑백 사진으로 만드는 모델 이해 


<br/>
<br/>

## 전처리
``` python
"""
데이터 불러오기
"""

dataset = np.load('datasets/pfcn_small.npz')
print(list(dataset.keys())) 
# ['train_images', 'test_images', 'train_mattes', 'test_mattes']

train_images = dataset['train_images']
test_images = dataset['test_images']
```


``` python
"""
칼라 이미지를 흑백이미지로 변환하여 학습 데이터 생성
"""

from skimage import color 
train_gray_images = np.array([color.rgb2gray(img).reshape(100, 75, 1) for img in train_images])
test_gray_images = np.array([color.rgb2gray(img).reshape(100, 75, 1) for img in test_images])

print(train_gray_images.shape) # (1700, 100, 75, 1)
print(test_gray_images.shape) # (300, 100, 75, 1)
```

<br/>
<br/>

## 모델링 및 학습 
``` python
from keras.layers import Dense, Input, MaxPool2D, Conv2D, Conv2DTranspose, Flatten, Reshape, Activation
from keras.layers import BatchNormalization, Dropout, Activation, concatenate
from keras.models import Model


def conv2d_block(x, channel):
	x = Conv2D(filters=channel, kernel_size=3, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(filters=channel, kernel_size=3, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
return x

  
def unet_black():
	inputs = Input(shape=(100, 75, 3))
	c1 = conv2d_block(inputs, 16)
	p1 = MaxPool2D(pool_size=(2))(c1)
	p1 = Dropout(0.1)(p1)

	c2 = conv2d_block(p1, 32)
	p2 = MaxPool2D(pool_size=(2))(c2)
	p2 = Dropout(0.1)(p2)

	c3 = conv2d_block(p2, 64)
	p3 = MaxPool2D(pool_size=(2))(c3)
	p3 = Dropout(0.1)(p3)

	c4 = conv2d_block(p3, 128)
	p4 = MaxPool2D(pool_size=(2))(c4)
	p4 = Dropout(0.1)(p4)

	c5 = conv2d_block(p4, 256)

	u6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='valid', output_padding=(0, 1))(c5)
	u6 = concatenate([u6, c4])
	u6 = Dropout(0.1)(u6)
	c6 = conv2d_block(u6, 128)

	u7 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='valid', output_padding=(1, 0))(c6)
	u7 = concatenate([u7, c3])
	u7 = Dropout(0.1)(u7)
	c7 = conv2d_block(u7, 64)

	u8 = Conv2DTranspose(32, kernel_size=2, strides=2, padding='valid', output_padding=(0, 1))(c7)
	u8 = concatenate([u8, c2])
	u8 = Dropout(0.1)(u8)
	c8 = conv2d_block(u8, 32)

	u9 = Conv2DTranspose(16, kernel_size=2, strides=2, padding='valid', output_padding=(0, 1))(c8)
	u9 = concatenate([u9, c1])
	u9 = Dropout(0.1)(u9)
	c9 = conv2d_block(u9, 16)

	outputs = Conv2D(filters = 1, kernel_size = 1, activation = 'sigmoid')(c9)

	model = Model(inputs, outputs)

return model

model = unet_black()

model.compile(loss = 'mae', optimizer = 'adam', metrics=['accuracy'])

hist = model.fit(
	train_images,
	train_gray_images,
	validation_data=(
		test_images,
		test_gray_images
	),
	epochs= 15,
	verbose=1
)
```

<br/>
<br/>

## 매우 간단한 모델 
``` python
from keras import backend as K

def simple_black():
	inputs = Input(shape = (100, 75, 3))
	x = Conv2D(filters = 3, kernel_size = 1, use_bias=False)(inputs) # 30 -> 15 -> 3
	x = Conv2D(filters = 1, kernel_size = 1, use_bias=False)(x)
	x = K.clip(x, 0.0, 1.0) # 0 ~ 1 까지로 보정
	
	return Model(inputs, x)

# 학습은 이전과 동일하게
```

