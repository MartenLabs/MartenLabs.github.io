---
title: 3. Multi layer classification
date: YYYY-MM-DD HH:MM:SS +09:00
categories: [Tensorflow, CNN]
tags:
  [
    Tensorflow,
    Multi layer classification,
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
- fashion MNIST이미지를 classification 하기
- Multi class와 Multi label 구분하기 
- 이미지에 객체 삽입하기 
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

(train_images, train_labels), (test_images, test_labels) = datasets
```


``` python
"""
레이블 설정 

https://www.kaggle.com/datasets/zalando-research/fashionmnist
"""

labels = ["T-shirt/top", # index 0
		  "Trouser",     # index 1
		  "Pullover",    # index 2
		  "Dress",       # index 3
		  "Coat",        # index 4
		  "Sandal",      # index 5
		  "Shirt",       # index 6
		  "Sneaker",     # index 7
		  "Bag",         # index 8
		  "Ankle boot"]  # index 9

def idx2label(idx):
	return labels[idx]
```


``` python
"""
이미지의 값이 가장 큰 idx와 작은 idx 출력
"""

train_images.reshape(shape = (60000, -1)).sum(axis = 1).argmax() # 55023
train_images.reshape(shape = (6000, -1)).sum(axis = 1).argmax()  # 9230

train_images.reshape(shape = (60000, -1)).sum(axis = 1)[55023]
train_images.reshape(shape = (60000, -1)).sum(axis = 1)[9230]
```


``` python
"""
정수형 데이터를 실수형으로 변경
"""

train_images = train_images.astype(np.float64)
test_images = test_images.astype(np.float64)
```

<br/>

$$normalize(x) = \frac{x - 최소값}{최대값 - 최소값}$$
$$normalize(x) = \frac{x}{최대값}$$

<br/>

``` python
"""
데이터 0-1 normalize 수행
"""

train_images = train_images / train_images.max()
test_images = test_images / test_images.max()

def norm(img):
	min_val = data.min()
	max_val = data.max()

	return (data - min_val) / (max_val - min_val)
```


``` python
"""
레이블에 따른 이미지 출력
"""

print(np.argwhere(train_labels == 9)[:5].shape) # (5, 1)

# 아래 셋은 같은 표현
print(np.argwhere(train_labels == 9)[:5].reshape(-1))
print(np.argwhere(train_labels == 9)[:5, 0])
print(np.argwhere(train_labels == 9)[:5, ..., 0])

"""
plt.imshow(train_images
		   [
			   np.argwhere(train_labels == 9)[:5, ..., 0]
		   ].transpose(1, 0, 2).reshape(28, -1))
"""

def filter(label, count=5):
	images = train_images
			[
				np.argwhere(train_labels == label)[:count, 0]
			].transpose(1, 0, 2).reshape(28, -1)
	
	plt.imshow(images)
	plt.show()

filter(0, 3)
```


<br/>
<br/>

## Data augmentation
``` python
"""
이미지 한장의 배셩 크기를 4배로 확대하고, 객체는 4분면 영역중 랜덤으로 한 공간에 넣는 함수
"""

def expand_4times(img):
	bg = np.zeros(img.shape)
	idx = np.random.randint(0, 4)
	
	slots = [bg, bg, bg, bg]
	slots[idx] = img

	expanded_img = np.vstack([
		np.hstack(slots[:2]),
		np.hstack(slots[2:])
	])

	return expanded_img


plt.imshow(expand_4times(train_images[0]))
plt.show()



train_expand_images = np.array([expand_4times(img) for img in train_images])
test_expand_images = np.array([expand_4times(img) for img in test_images])
```


``` python
"""
이미지 한장의 배경 크기를 4배로 확대하고, 객체를 랜덤으로 1 ~ 4개, 랜덤 4분면에 위치 시키는 함수
"""

def expand_4times2(x_data, y_data):
	images = []
	labels = []

	for _ in range(4):
		bg = np.zeros(shape = (28, 28))
		obj_count = np.random.randint(0, 5)
		label = np.zeros((10, ))
		slots = [bg, bg, bg, bg]

		for idx in range(obj_count):
			i = np.random.randint(len(x_data))

			slots[idx] = x_data[i]
			label += tf.keras.utils.to_categorical(y_data[i], 10)

			np.random.shuffle(slots)

	new_img = np.vstack([
		np.hstack(slots[:2]),
		np.hstack(slots[2:])
	])

	images.append(new_img)
	labels.append((label >= 1).astype(np.int8))

	return np.array(images), np.array(labels)



train_multi_images, train_multi_labels = list(zip(*[
		expand_4times2(train_images, train_labels) for _ in train_images]))


test_multi_images, test_multi_labels = list(zip(*[
		expand_4times2(test_images, test_labels) for _ in test_images]))



print(np.array(train_multi_images).shape)      # (60000, 1, 56, 56)
np.array(train_multi_images[:, 0, :, :]).shape # (60000, 56, 56)

print(np.array(train_multi_labels).shape)      # (60000, 1, 10)
np.array(train_multi_labels[:, 0, :]).shape    # (60000, 10)
```


``` python
"""
4차원 데이터 처리
"""

# (60000, 56, 56, 1)
train_multi_images = 
			np.array(train_multi_images)[:, 0, :, :].reshape(-1, 56, 56 ,1)

# (60000, 10)
train_multi_labels = 
			np.array(train_multi_labels)[:, 0, :]


test_multi_images =
			np.array(test_multi_images)[:, 0, :, :].reshape(-1, 56, 56 ,1)
test_multi_labels = 
			np.array(test_multi_labels)[:, 0, :]
```


``` python
"""
의류 갯수에 따라 연속된 그림 보여주는 함수
"""
def filter2(obj_count, count = 5):
	labels = train_multi_labels.sum(axis = 1)

	idx = np.argwhere(labels == obj_count)[:count, 0]

	imgs = train_multi_images[idx][
									..., 
									0].transpose(1, 0, 2).reshape(56, -1)
	plt.imshow(imgs)
	plt.show()


filter2(1, 5)
```

<br/>
<br/>

## 모델링 
``` python
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Dense, Flatten
from keras.models import Model 

def single_fashon_mnist_model():
	inputs = Input(shape = (56, 56, 1))
	
	x = Conv2D(filters = 16, 
			   kernel_size = 2,
			   padding = 'same',
			   activation = 'relu')(inputs)
	x = MaxPool2D(pool_size = 2)(x)
	x = Dropout(0.3)(x)

	x = Conv2D(filters = 32,
			   kernel_size = 2,
			   padding = 'same',
			   activation = 'relu')(x)
	x = MaxPool2D(pool_size = 2)(x)
	x = Dropout(0.3)(x)

	x = Conv2D(filters = 64,
			   kernel_szize = 2,
			   padding = 'same',
			   activation = 'relu')(x)

	x = Flatten()(x)
	"""
	x = GlobalAvgPool2D()(x)
	요즘엔 Flatten 대신 이걸 더 많이 사용한다고 하는데 여기서는 성능이 더 떨어져 Flatten 사용
	"""
	x = Dense(units = 10, activation = 'softmax')(x)

	return (Model(inputs, x))
```

``` python
model = single_fashon_mnist_model()
model.summary()


"""
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 56, 56, 1)]       0         
                                                                 
 conv2d (Conv2D)             (None, 56, 56, 16)        80        
                                                                 
 max_pooling2d (MaxPooling2  (None, 28, 28, 16)        0         
 D)                                                              
                                                                 
 dropout (Dropout)           (None, 28, 28, 16)        0         
                                                                 
 conv2d_1 (Conv2D)           (None, 28, 28, 32)        2080      
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 14, 14, 32)        0         
 g2D)                                                            
                                                                 
 dropout_1 (Dropout)         (None, 14, 14, 32)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 14, 14, 64)        8256      
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 7, 7, 64)          0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 3136)              0         
                                                                 
 dense (Dense)               (None, 10)                31370     
                                                                 
=================================================================
Total params: 41786 (163.23 KB)
Trainable params: 41786 (163.23 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

"""
```


``` python
model.compile(loss = 'categorical_crossentropy',
			  optimizer = 'adam',
			  metrics = ['accuracy'])
```

``` python
hist = model.fit(
	train_expand_images.reshape(-1, 56, 56, 1),
	keras.utils.to_categorical(train_labels, 10),
	
	validation_data = (
		test_expand_images.reshape(-1, 56, 56, 1),
		keras.utils.to_categorical(test_labels, 10)
	),
	epochs = 15,
	verbose = 1
)
```

``` python
res = mode.predict(test_expand_images[1].reshape(1, 56, 56, 1))
print(res.argmax())   # 2
print(test_labels[1]) # 2
```

``` python
plt.bar(range(10), keras.utils_tocategorical(test_labels[1], 10))
plt.bar(np.array(range(10)) + 0.1, res[0])
```

<br/>
<br/>

## Multi label modeling

``` python
"""
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 56, 56, 1)]       0         
                                                                 
 conv2d (Conv2D)             (None, 56, 56, 16)        80        
                                                                 
 max_pooling2d (MaxPooling2  (None, 28, 28, 16)        0         
 D)                                                              
                                                                 
 dropout (Dropout)           (None, 28, 28, 16)        0         
                                                                 
 conv2d_1 (Conv2D)           (None, 28, 28, 32)        2080      
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 14, 14, 32)        0         
 g2D)                                                            
                                                                 
 dropout_1 (Dropout)         (None, 14, 14, 32)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 14, 14, 64)        8256      
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 7, 7, 64)          0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 3136)              0         
                                                                 
 dense (Dense)               (None, 10)                31370     
                                                                 
=================================================================
Total params: 41786 (163.23 KB)
Trainable params: 41786 (163.23 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
"""

def multi_fashion_mnist_model(model:Model) -> Model:
	model.trainable = False                  # 필터 학습 X
	x = model.layers[-2].output              # flatten 부분 인터셉트
	# 각각의 슬롯의 확률을 구하기 위해 sigmoid 사용(softmax는 하나만 부각)
	x = Dense(10, activation = 'sigmoid')(x) 

	return Model(model.input, x)
```


``` python
new_model = multi_fashion_mnist_model(model)
new_model.summary()

"""
Model: "model_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 56, 56, 1)]       0         
                                                                 
 conv2d (Conv2D)             (None, 56, 56, 16)        80        
                                                                 
 max_pooling2d (MaxPooling2  (None, 28, 28, 16)        0         
 D)                                                              
                                                                 
 dropout (Dropout)           (None, 28, 28, 16)        0         
                                                                 
 conv2d_1 (Conv2D)           (None, 28, 28, 32)        2080      
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 14, 14, 32)        0         
 g2D)                                                            
                                                                 
 dropout_1 (Dropout)         (None, 14, 14, 32)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 14, 14, 64)        8256      
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 7, 7, 64)          0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 3136)              0         
                                                                 
 dense_2 (Dense)             (None, 10)                31370     
                                                                 
=================================================================
Total params: 41786 (163.23 KB)
Trainable params: 31370 (122.54 KB)
Non-trainable params: 10416 (40.69 KB)
_________________________________________________________________
"""
```


``` python
new_model.compile(
				  loss = 'binary_crossentropy', 
				  optimizer = 'adam',
				  metrics = ['accuracy'])

new_hist = new_model.fit(
	train_multi_images,
	train_multi_labels,

	validation_data = (
		test_multi_images,
		test_multi_labels
	),
	epochs = 15,
	verbose = 1
)
```

<br/>
<br/>

## epoch 마다 랜덤 데이터 생성하여 학습시키는 법
``` python
epochs = 15
batch_size = 32

new_model.compile(loss = 'binary_crossentropy',
				  optimizer = 'adam',
				  metrics = ['accuracy'])

for epoch in range(epochs):
	for _ in range(0, len(train_images), batch_size):
		batch_x = []
		batch_y = []

		for _ in range(batch_size):
			x, y = expanded_4times2(train_images, train_labels),
			batch_x.append(x)
			batch_y.append(y)

		batch_x = np.array(batch_x).reshape(-1, 56, 56, 1)
		batch_y = np.array(batch_y).reshape(-1, 10)

		new_model.train_on_batch(batch_x, batch_y)
	print(epoch, 'epoch')
```