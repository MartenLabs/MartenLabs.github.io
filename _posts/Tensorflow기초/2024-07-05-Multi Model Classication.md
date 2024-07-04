---
title: 2. Multi Model Classication
date: YYYY-MM-DD HH:MM:SS +09:00
categories: [Tensorflow, CNN]
tags:
  [
    Tensorflow,
    Multi Model Classication,
	celeba
  ]
pin: true
math: true
mermaid: true
---

## 데이터 소개

- 얼굴 데이터로 유명한 celeba dataset을 이용
![](https://tera.dscloud.me:8080/Images/Models/2.Multi_Model_Classificaiton.png)

위의 이미지는 다음과 같은 것을 보여줍니다.
- 10,177 개의 신원
- 얼굴 이미지 수 202,599 개
- 5 개의 랜드 마크 위치, 이미지 당 40 개의 바이너리 속성 주석
- 성별
- 큰 코
- 매력적
- 젊음
- 웃음 여부
- 모자 착용 여부
- 안경 착용 여부
- etc

<br/>
<br/>

## 최종 목표
- 작게 줄인 celeba 데이터를 이용하여 웃음, 성별 동시 구분
- 한 모델에서 여러 결과에 대한 분석을 하는 방법
- 큰 사진을 작게 줄이기
---


<br/>
<br/>

## 전처리
``` python 
"""
데이터 추출
"""
import tensorflow_datasets as tfds
from skimage.transform import resize

celeb_a = tfds.load('celeb_a')

celeb_a_trian, celeb_a_test = celeb_a['train'], celeb_a['test']

train_images = []
train_labels = []

for tensor in tftd.as_numpy(celeb_a_train):
	isMale = tensor['attributes']['Male']
	isSmiling = tensor['attributes']['Smiling']

	label = np.array([isMale,
					  isSmiling]).astype(np.int8)
	img = resize(tensor['image'], (190//1.5, 89//1.5))

	train_labels.append(label)
	train_images.append(img)

"""
test dataset도 동일 
"""
```


``` python
"""
데이터량 축소
"""
import random

m_s = [] # 남자, 웃음 
f_s = [] # 여자, 웃음 
m_n = [] # 남자, 안웃음 
f_n = [] # 여자, 안웃음

for a, b in zip(test_images, test_labels):
	if b[0] and b[1]:
		m_s.append((a, b))

	elif not b[0] and b[1]:
		f_s.append((a, b))

	elif b[0] and not b[1]:
		m_n.append((a, b))

	elif not b[0] and not b[1]:
		f_n.append((a, b))


total = m_s[:550] + f_s[:550] + m_n[:550] + f_n[:550]


random.shuffle(total)
trains = total[:2000]
tests = total[2000:]


train_images, train_labels = list(zip(*trains))
test_images, test_labels = list(zip(*tests))

train_images, train_labels = np.array(train_iamges), np.array(train_labels)
test_images, test_labels = np.array(test_images), np.array(test_labels)
```


``` python
"""
각각 onehot encoding
"""

from keras.utils import to_categorical

train_male_labels, train_smile_labels=np.split(train_labels, 2, axis=1)
test_male_labels, test_smile_labels=np.split(test_labels, 2, axis=1)

train_male_labels = to_categorical(train_male_labels)
train_smile_labels = to_categorical(train_smile_labels)
test_male_labels = to_categorical(test_male_labels)
test_smile_labels = to_categorical(test_smile_labels)
```


``` python
"""
onehot encoding 합치기
"""

train_labels_onehot = np.concatenate([
								train_male_labels,
								train_smile_labels], axis=1)

test_labels_onehot = np.concatenate([
								test_male_labels,
								test_smile_labels], axis=1)
```


<br/>
<br/>

## Modeling
``` python
"""
smile, gender 판별 각각 모델링
"""

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Input, Dense, Flatten

def simple_model():
	inputs = Input(shape = (72, 59, 3))
	# shape 구하기
		# n_H' = n_H - f + 1
		# n_H = 72 - 3 + 1, n_W = 59 - 3 + 1 = (70, 57, 32)
	
	# 파라미터 갯수 구하기
		# 파라미터 수 = (필터 크기 * 입력 채널 수 + 1) * 필터 수
		# kernel_size = 3 (3x3 필터를 사용)

	# 입력 채널 수 = 3 (입력 이미지의 RGB 채널)
	# 필터 수는 filters 매개변수로 지정된 3
	
	# 파라미터 갯수 = (3 * 3 * 3 + 1) * 32 = (27 + 1) * 32 = 28 * 32 = 896
	x = Conv2D(filters = 32, kerenl_size = 3, activation = 'relu')(inputs)

	# 70 / 2, 57 / 2 = (35, 28, 32) 일반적으로 내림연산
	x = MaxPool2D(pool_size = 2)(x)

	# (33, 26, 64) (3 * 3 * 32 + 1) * 64 = 18496
	x = Conv2D(filters = 64, kerenl_size = 3, activation = 'relu')(x)	

	# (16, 13, 64)
	x = MaxPool2D(pool_size = 2)(x)

	# (14, 11, 64)
	x = Conv2D(filters = 64, kerenl_size = 3, activation = 'relu')(x)

	# 14 * 11 * 64
	x = Flatten()(x)
	x = Dense(units = 64, activation = 'relu')(x)
	outputs = Dense(units = 2, activation = 'softmax')(x)

	model = Model(inputs , outputs)

	return model
```


``` python
gender_model = simple_model()
smile_model = simple_mode()

gender_model.summary()

"""
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 72, 59, 3)]       0         
                                                                 
 conv2d (Conv2D)             (None, 70, 57, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2  (None, 35, 28, 32)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 33, 26, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 16, 13, 64)        0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 14, 11, 64)        36928     
                                                                 
 flatten (Flatten)           (None, 9856)              0         
                                                                 
 dense (Dense)               (None, 64)                630848    
                                                                 
 dense_1 (Dense)             (None, 2)                 130       
                                                                 
=================================================================
Total params: 687298 (2.62 MB)
Trainable params: 687298 (2.62 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
"""
```


``` python
gender_model.compile(loss = 'categorical_crossentropy', 
					 optimizer = 'adam',
					 metrics = ['accuracy'])

smile_model.compile(loss = 'categorical_crossentropy',
					optimizer = 'adma',
					metrics = ['accuracy'])
```


``` python
gender_hist = gender_model.fit(
	train_images,
	train_male_labels,
	
	validation_data = (
		test_images,
		test_male_labels	
	),

	epochs = 15,
	verbose = 1
)


smile_hist = smile_model.fit(
	train_images,
	train_smile_labels,

	validation_data = (
		test_images,
		test_smile_labels
	),

	epochs = 15,
	verbose = 1
)
```


``` python
gneder_res = gender_model.predict(test_images[1:2])

smile_res = gender_model.predict(test_images[1:2])
```



<br/>
<br/>

## Multioutput Modeling
``` python
def multi_model():
	inputs = Inputs(shape = (72, 59, 3))

	L1 = Conv2D(filters = 32, kernel_size = 3, activation = 'relu')(inputs)
	L2 = MaxPool2D(pool_size = 2)(L1)

	L3 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu')(L2)
	L4 = MaxPool2D(pool_size = 2)(L3)

	L5 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu')(L4)
	L6 = MaxPool2D(pool_size = 2)(L5)
	L7 = Flatten()(L6)

	latent_vector = Dnese(units = 64, activation = 'relu')(L7)

	gender_outputs = Dense(units = 2, activation = 'softmax')(latent_vector)
	smile_outputs = Dense(units = 2, activation = 'softmax')(latent_vector)

	model = Model(inputs, [gender_outputs, smile_outputs])

	# outputs = Concatenate(axis = 1)([gender_outputs, smile_outputs])
	# model = Model(inputs, outputs)

	return model
```


``` python

model = multi_model()
model.summary()

"""
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_3 (InputLayer)        [(None, 72, 59, 3)]          0         []                            
                                                                                                  
 conv2d_6 (Conv2D)           (None, 70, 57, 32)           896       ['input_3[0][0]']             
                                                                                                  
 max_pooling2d_4 (MaxPoolin  (None, 35, 28, 32)           0         ['conv2d_6[0][0]']            
 g2D)                                                                                             
                                                                                                  
 conv2d_7 (Conv2D)           (None, 33, 26, 64)           18496     ['max_pooling2d_4[0][0]']     
                                                                                                  
 max_pooling2d_5 (MaxPoolin  (None, 16, 13, 64)           0         ['conv2d_7[0][0]']            
 g2D)                                                                                             
                                                                                                  
 conv2d_8 (Conv2D)           (None, 14, 11, 64)           36928     ['max_pooling2d_5[0][0]']     
                                                                                                  
 max_pooling2d_6 (MaxPoolin  (None, 7, 5, 64)             0         ['conv2d_8[0][0]']            
 g2D)                                                                                             
                                                                                                  
 flatten_2 (Flatten)         (None, 2240)                 0         ['max_pooling2d_6[0][0]']     
                                                                                                  
 dense_4 (Dense)             (None, 64)                   143424    ['flatten_2[0][0]']           
                                                                                                  
 dense_5 (Dense)             (None, 2)                    130       ['dense_4[0][0]']             
                                                                                                  
 dense_6 (Dense)             (None, 2)                    130       ['dense_4[0][0]']             
                                                                                                  
==================================================================================================
Total params: 200004 (781.27 KB)
Trainable params: 200004 (781.27 KB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________

"""
```


``` python
model.compile(loss = 'categorical_crossentropy',
			  optimizer = 'adam',
			  metrics = ['accuracy']
)
```


``` python
multi_model_hist = model.fit(
	train_images,
	[train_male_labels, train_smile_labels],

	validation_data = (test_images,
					   [test_male_labels, test_smile_labels]),
	epochs = 15,
	verbose = 1
)
```



<br/>
<br/>

## 모델 분리 
``` python
splitted_gender_model = Model(
							inputs = model.input, 
							outputs = model.get_layer('dense_5').output)

splitted_gender_res = splitted_gender_model.predict(test_images[0:1])
splitted_gender_res.argmax()



splitted_smile_model = Model(
							 inputs = model.input,
							 outputs = model.get_layer('dense_6').output)
```


