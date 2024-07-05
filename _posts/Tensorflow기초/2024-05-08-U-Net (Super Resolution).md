---
title: 8. U-Net (Super Resolution)
date: YYYY-MM-DD HH:MM:SS +09:00
categories: [Tensorflow]
tags:
  [
    Tensorflow,
    U-Net,
	Super Resolution
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

<br/>
<br/>

## 전처리 
``` python
"""
데이터 확인 밎 이미지 크기 줄이기
"""

import imageio
import skimage.transform import resize

x = imageio.imread('datasets/pfcn_original/training/00001.png')
x = resize(x, output_shape = (100, 75, 3))
x = resize(x, output_shape = (50, 37, 3))
```

``` python
"""
데이터 로드 
"""
datasets = np.load('datasets/pfcn_small.npz')
print(list(datasets.keys()))

train_big_images, test_big_images = datasets['train_images'], datasets['test_images']
```

``` python
"""
이미지 축소 및 할당
"""

train_small_images = np.array([resize(img, output_shape=(50, 37, 3)) for img in train_big_images])
test_small_images = np.array([resize(img, output_shape=(50, 37, 3)) for img in test_big_images])

print(train_small_images.shape) # (1700, 50, 37, 3)
print(test_small_images.shape)  # (300, 60, 37, 3)

plt.imshow(train_small_images[:5].transpose(1, 0, 2, 3).reshape(50, -1, 3))
plt.colorbar()
plt.show()
```

<br/>
<br/>

## 모델링

``` python
def conv2d_block(x, channel):
	... 

def unet_resolution():
	inputs = Input(shape = (50, 37, 3))
	...

	u10 = Conv2DTranspose(filters = 16, kernel_size = 2, strides = 2, padding = 'valid', output_padding = (0, 1))(c9) # 이미지 4배 크게
	outputs = Conv2D(filters = 3, kernel_size = 1, acitvation = 'sigmoid')(u10)
	
	return Model(inputs, outputs)


model = unet_resolution()

model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

hist = model.fit(
	train_small_images,
	train_big_images,

	validation_data = (
		test_small_images,
		test_big_images
	),
	epochs = 53,
	verbose = 1
)
```


``` python
def srcnn():
	inputs = Input(shape = (100, 75, 3))
	x = Conv2D(filters = 64, kernel_size = 9, activation = 'relu', padding = 'same')(inputs)

	x1 = Conv2D(filters = 32, kernel_size = 1, activation = 'relu', padding = 'same')(x)
	x2 = Conv2D(filters = 32, kernel_size = 3, activation = 'relu', padding = 'same')(x)
	x3 = Conv2D(filters = 32, kernel_size = 6, activation = 'relu', padding = 'same')(x)
	x = Average()([x1, x2, x3])

	outputs = Conv2D(filters = 3, kernel_size = 5, activation = 'relu', padding = 'same')(x)
	model = Model(inputs, outputs)

	model.compile(loss = 'mae', optimizer = 'adam', metrics = ['accuracy'])
	return model 

srcnn_model = srcnn()
srcnn_model.summary()

"""
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_6 (InputLayer)        [(None, 100, 75, 3)]         0         []                            
                                                                                                  
 conv2d_38 (Conv2D)          (None, 100, 75, 64)          15616     ['input_6[0][0]']             
                                                                                                  
 conv2d_39 (Conv2D)          (None, 100, 75, 32)          2080      ['conv2d_38[0][0]']           
                                                                                                  
 conv2d_40 (Conv2D)          (None, 100, 75, 32)          18464     ['conv2d_38[0][0]']           
                                                                                                  
 conv2d_41 (Conv2D)          (None, 100, 75, 32)          51232     ['conv2d_38[0][0]']           
                                                                                                  
 average_4 (Average)         (None, 100, 75, 32)          0         ['conv2d_39[0][0]',           
                                                                     'conv2d_40[0][0]',           
                                                                     'conv2d_41[0][0]']           
                                                                                                  
 conv2d_42 (Conv2D)          (None, 100, 75, 3)           2403      ['average_4[0][0]']           
                                                                                                  
==================================================================================================
Total params: 89795 (350.76 KB)
Trainable params: 89795 (350.76 KB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________
"""


srcnn_hist = srcnn_model.fit(
	train_lr_images,
	train_big_images,
	
	validation_data = (
		test_lr_images,
		test_big_images
	),
	epochs = 50,
	verbose = 1
)
```



