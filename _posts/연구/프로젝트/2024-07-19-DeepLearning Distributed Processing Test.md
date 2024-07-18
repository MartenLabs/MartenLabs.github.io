---
title: DeepLearning Distributed Processing Test
date: YYYY-MM-DD HH:MM:SS +09:00
categories: [연구, 프로젝트]
tags:
  [
	MCU,
	Distributed Processing
  ]
pin: true
math: true
mermaid: true
---


#### 본 프로젝트는 모델을 세분화하여 여러 저사양 MCU 장치에서 딥러닝 분산처리 테스트를 진행하기 전 증명 과정 단계임.


### 데이터셋 준비 및 전처리

``` python
import keras
import numpy as np
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

  

train_images = train_images + np.random.normal(0.5, 0.1, train_images.shape)
train_images[train_images > 1.0] = 1.0
train_images = train_images / train_images.max()
print(train_images.shape)
train_images = np.expand_dims(train_images, axis=-1)
print(train_images.shape)

print(train_images.max())
print(train_images.min())


test_images = test_images + np.random.normal(0.5, 0.1, test_images.shape)
test_images[test_images > 1.0] = 1.0
test_images = test_images / test_images.max()
print(test_images.shape)
test_images = np.expand_dims(test_images, axis=-1)
print(test_images.shape)


train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

print(train_labels.shape)
train_labels = np.expand_dims(train_labels, axis=1)
print(train_labels.shape)

print(test_labels.shape)
test_labels = np.expand_dims(test_labels, axis=1)
print(test_labels.shape)

"""
(60000, 10) 
(60000, 1, 10) 
(10000, 10) 
(10000, 1, 10)
"""
```

<br/>

### 데이터 확인 
``` python
print(train_labels[2])
plt.imshow(train_images[2])
plt.axis('off')
plt.show()

print(train_images[2].max())
print(train_images[2].min())

"""
[[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
"""
```

<br/>

### 모델 생성 

모델 학습 후 각 함수단위로 모델을 쪼개어 개별 장비에 탑재 

``` python
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Input, Dense, Flatten, Activation, BatchNormalization

  
def Layer_1(inputs): # 장치 1
	x = Conv2D(3, strides=2 ,kernel_size=1, padding='SAME')(inputs)
	x = Conv2D(6, strides=1, kernel_size=3, padding='SAME')(x)
	x = BatchNormalization()(x)
	x = Activation('ReLU')(x)
	x = Conv2D(9, strides=2, kernel_size=1, padding='SAME')(x) # 14
	x = Conv2D(12, strides=1, kernel_size=3, padding='SAME')(x)
	x = BatchNormalization()(x)
	x = Activation('ReLU')(x)
	return x


def Layer_2(inputs): # 장치 2
	x = Conv2D(15, strides=1 ,kernel_size=1, padding='SAME')(inputs)
	x = Conv2D(18, strides=1 ,kernel_size=3, padding='SAME')(x)
	x = BatchNormalization()(x)
	x = Activation('ReLU')(x)
	x = Conv2D(21, strides=2 ,kernel_size=1, padding='SAME')(x) # 7, 7
	x = Conv2D(24, strides=1 ,kernel_size=3, padding='SAME')(x) # 7, 7
	x = BatchNormalization()(x)
	x = Activation('ReLU')(x)
	return x

  
def Layer_3(inputs): # 장치 3
	x = Conv2D(27, strides=2 ,kernel_size=3, padding='SAME')(inputs) # 3
	x = Conv2D(10, strides=2 ,kernel_size=3, padding='SAME')(x) # 1
	return tf.keras.layers.Reshape((-1, 10))(x)

  
  

def just_train_model(): # 학습용 
	inputs = Input(shape=(28, 28, 1))
	layer1_feature = Layer_1(inputs)
	layer2_feature = Layer_2(layer1_feature)
	layer3_output = Layer_3(layer2_feature)
	model = tf.keras.Model(inputs, layer3_output)
	return model


model = just_train_model()
model.trainable = True
model.build(input_shape=(None, 24, 24, 1))
model.summary()


"""
Model: "model_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_5 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                 
 conv2d_20 (Conv2D)          (None, 14, 14, 3)         6         
                                                                 
 conv2d_21 (Conv2D)          (None, 14, 14, 6)         168       
                                                                 
 batch_normalization_8 (Bat  (None, 14, 14, 6)         24        
 chNormalization)                                                
                                                                 
 activation_8 (Activation)   (None, 14, 14, 6)         0         
                                                                 
 conv2d_22 (Conv2D)          (None, 7, 7, 9)           63        
                                                                 
 conv2d_23 (Conv2D)          (None, 7, 7, 12)          984       
                                                                 
 batch_normalization_9 (Bat  (None, 7, 7, 12)          48        
 chNormalization)                                                
                                                                 
 activation_9 (Activation)   (None, 7, 7, 12)          0         
                                                                 
 conv2d_24 (Conv2D)          (None, 7, 7, 15)          195       
                                                                 
 conv2d_25 (Conv2D)          (None, 7, 7, 18)          2448      
                                                                 
 batch_normalization_10 (Ba  (None, 7, 7, 18)          72        
 tchNormalization)                                               
                                                                 
 activation_10 (Activation)  (None, 7, 7, 18)          0         
                                                                 
 conv2d_26 (Conv2D)          (None, 4, 4, 21)          399       
                                                                 
 conv2d_27 (Conv2D)          (None, 4, 4, 24)          4560      
                                                                 
 batch_normalization_11 (Ba  (None, 4, 4, 24)          96        
 tchNormalization)                                               
                                                                 
 activation_11 (Activation)  (None, 4, 4, 24)          0         
                                                                 
 conv2d_28 (Conv2D)          (None, 2, 2, 27)          5859      
                                                                 
 conv2d_29 (Conv2D)          (None, 1, 1, 10)          2440      
                                                                 
 reshape_2 (Reshape)         (None, 1, 10)             0         
                                                                 
=================================================================
Total params: 17362 (67.82 KB)
Trainable params: 17242 (67.35 KB)
Non-trainable params: 120 (480.00 Byte)
_________________________________________________________________

"""
```

<br/>

### 모델 컴파일 및 학습 

``` python
from tensorflow.keras.optimizers import Adam

initial_learning_rate = 0.0001
optimizer = Adam(learning_rate=initial_learning_rate, clipnorm=1.0) #0.25
model.compile(loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
optimizer = optimizer,
metrics = ['accuracy'])

model.fit(
	train_images,
	train_labels,
	validation_data = (
		test_images,
		test_labels
	),
	epochs = 10,
	batch_size = 128,
	verbose = 1
)

"""
Epoch 1/10
2024-07-18 14:14:27.409730: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:442] Loaded cuDNN version 8700
2024-07-18 14:14:28.675514: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f79e536a000 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2024-07-18 14:14:28.675542: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA GeForce RTX 3090, Compute Capability 8.6
2024-07-18 14:14:28.679782: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2024-07-18 14:14:28.821083: I ./tensorflow/compiler/jit/device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
469/469 [==============================] - 15s 18ms/step - loss: 1.4509 - accuracy: 0.5432 - val_loss: 0.7841 - val_accuracy: 0.7676
Epoch 2/10
469/469 [==============================] - 7s 15ms/step - loss: 0.5968 - accuracy: 0.8182 - val_loss: 0.4575 - val_accuracy: 0.8612
Epoch 3/10
469/469 [==============================] - 7s 15ms/step - loss: 0.4114 - accuracy: 0.8716 - val_loss: 0.3508 - val_accuracy: 0.8925
Epoch 4/10
469/469 [==============================] - 7s 15ms/step - loss: 0.3309 - accuracy: 0.8966 - val_loss: 0.2908 - val_accuracy: 0.9099
Epoch 5/10
469/469 [==============================] - 7s 15ms/step - loss: 0.2818 - accuracy: 0.9113 - val_loss: 0.2567 - val_accuracy: 0.9191
Epoch 6/10
469/469 [==============================] - 7s 15ms/step - loss: 0.2488 - accuracy: 0.9220 - val_loss: 0.2290 - val_accuracy: 0.9274
Epoch 7/10
469/469 [==============================] - 7s 15ms/step - loss: 0.2248 - accuracy: 0.9287 - val_loss: 0.2092 - val_accuracy: 0.9337
Epoch 8/10
469/469 [==============================] - 7s 15ms/step - loss: 0.2063 - accuracy: 0.9346 - val_loss: 0.1970 - val_accuracy: 0.9387
Epoch 9/10
469/469 [==============================] - 7s 15ms/step - loss: 0.1915 - accuracy: 0.9389 - val_loss: 0.1844 - val_accuracy: 0.9414
Epoch 10/10
469/469 [==============================] - 7s 15ms/step - loss: 0.1799 - accuracy: 0.9430 - val_loss: 0.1741 - val_accuracy: 0.9431
"""
```

<br/>

### 결과 확인 

``` python
test = model.predict(test_images[2:3])
plt.axis('off')
plt.imshow(test_images[2])
plt.show()
print(np.argmax(test))
```

<br/>

### 모델 분리 

``` python
Layer1 = Model(inputs = model.input,
			   outputs = model.get_layer('activation_1').output)

Layer2 = Model(inputs = model.get_layer('conv2d_4').input, 
			   outputs = model.get_layer('activation_3').output)

Layer3 = Model(inputs = model.get_layer('conv2d_8').input, 
			   outputs = model.get_layer('reshape').output)




Layer1.summary()

"""
Model: "model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                 
 conv2d (Conv2D)             (None, 14, 14, 3)         6         
                                                                 
 conv2d_1 (Conv2D)           (None, 14, 14, 6)         168       
                                                                 
 batch_normalization (Batch  (None, 14, 14, 6)         24        
 Normalization)                                                  
                                                                 
 activation (Activation)     (None, 14, 14, 6)         0         
                                                                 
 conv2d_2 (Conv2D)           (None, 7, 7, 9)           63        
                                                                 
 conv2d_3 (Conv2D)           (None, 7, 7, 12)          984       
                                                                 
 batch_normalization_1 (Bat  (None, 7, 7, 12)          48        
 chNormalization)                                                
                                                                 
 activation_1 (Activation)   (None, 7, 7, 12)          0         
                                                                 
=================================================================
Total params: 1293 (5.05 KB)
Trainable params: 1257 (4.91 KB)
Non-trainable params: 36 (144.00 Byte)
_________________________________________________________________
"""


Layer2.summary()

"""
Model: "model_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(None, 7, 7, 12)]        0         
                                                                 
 conv2d_4 (Conv2D)           (None, 7, 7, 15)          195       
                                                                 
 conv2d_5 (Conv2D)           (None, 7, 7, 18)          2448      
                                                                 
 batch_normalization_2 (Bat  (None, 7, 7, 18)          72        
 chNormalization)                                                
                                                                 
 activation_2 (Activation)   (None, 7, 7, 18)          0         
                                                                 
 conv2d_6 (Conv2D)           (None, 4, 4, 21)          399       
                                                                 
 conv2d_7 (Conv2D)           (None, 4, 4, 24)          4560      
                                                                 
 batch_normalization_3 (Bat  (None, 4, 4, 24)          96        
 chNormalization)                                                
                                                                 
 activation_3 (Activation)   (None, 4, 4, 24)          0         
                                                                 
=================================================================
Total params: 7770 (30.35 KB)
Trainable params: 7686 (30.02 KB)
Non-trainable params: 84 (336.00 Byte)
_________________________________________________________________
"""


Layer3.summary()

"""
Model: "model_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_3 (InputLayer)        [(None, 4, 4, 24)]        0         
                                                                 
 conv2d_8 (Conv2D)           (None, 2, 2, 27)          5859      
                                                                 
 conv2d_9 (Conv2D)           (None, 1, 1, 10)          2440      
                                                                 
 reshape (Reshape)           (None, 1, 10)             0         
                                                                 
=================================================================
Total params: 8299 (32.42 KB)
Trainable params: 8299 (32.42 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
"""
```

<br/>

### 모델 분산 연산 전 Sequential 테스트

``` python
layer1_res = Layer1.predict(test_images[501:502])
layer2_res = Layer2.predict(layer1_res)
layer3_res = Layer3.predict(layer2_res)

print("layer1_res:", layer1_res.shape)
print("layer2_res:", layer2_res.shape)
print("layer3_res:", layer3_res.shape)


plt.axis('off')
plt.imshow(test_images[501])
# plt.savefig('./501', bbox_inches='tight', pad_inches=0)
plt.show()

print(np.argmax(layer3_res))
print(test_images[501].max())
print(test_images[501].min())



"""
1/1 [==============================] - 0s 17ms/step
1/1 [==============================] - 0s 17ms/step
1/1 [==============================] - 0s 28ms/step
layer1_res: (1, 7, 7, 12)
layer2_res: (1, 4, 4, 24)
layer3_res: (1, 1, 10)
"""
```

<br/>

### 각 모델 저장 

``` python
Layer1.save("Layer1.keras")

Layer2.save("Layer2.keras")

Layer3.save("Layer3.keras")
```

<br/>

### 최종 검증 

``` python
from tensorflow.keras.models import load_model

image_path = '2.png'
img = Image.open(image_path).convert('L') 
img = img.resize((28, 28))
img_array = np.array(img).reshape((1, 28, 28, 1)) 

print(img_array.shape)
img_array = img_array.astype('float32') / 255.0 

print(img_array.max())
print(img_array.min())

Layer1 = load_model('Layer1.keras')
Layer2 = load_model('Layer2.keras')
Layer3 = load_model('Layer3.keras')

layer1_res = Layer1.predict(img_array)
layer2_res = Layer2.predict(layer1_res)
layer3_res = Layer3.predict(layer2_res)

print("layer1_res:", layer1_res.shape)
print("layer2_res:", layer2_res.shape)
print("layer3_res:", layer3_res.shape)

print(np.argmax(layer3_res))

plt.axis('off')
plt.imshow(img_array[0])
plt.show()


"""
(1, 28, 28, 1)
0.8509804
0.105882354
1/1 [==============================] - 0s 61ms/step
1/1 [==============================] - 0s 61ms/step
1/1 [==============================] - 0s 42ms/step
layer1_res: (1, 7, 7, 12)
layer2_res: (1, 4, 4, 24)
layer3_res: (1, 1, 10)

2
"""
```

<br/>

### 장치 1 서버 구현(Layer1)

``` python
from flask import Flask, request, jsonify
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import io


app = Flask(__name__)

model = load_model('Layer1.keras')

def predict_image(img_array):
	predictions = model.predict(img_array)
	return predictions


@app.route('/Layer1', methods=['POST'])
def layer1_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_bytes = io.BytesIO(file.read())
        img_array = np.load(img_bytes).reshape((1, 28, 28, 1))
        predictions = predict_image(img_array)
        result = predictions.tolist()  
    except Exception as e:
        return jsonify({'error': str(e)}), 500


    target_ip = 'http://127.0.0.1:7000/Layer2'
    try:
        response = requests.post(target_ip, json={'predictions': result})
        response.raise_for_status()
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        layer2_response = response.json()['predictions']
        print(layer2_response)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to send data to {target_ip}: {str(e)}'}), 500
    return jsonify({'predictions': layer2_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)

```


<br/>

### 장치 2 서버 구현(Layer2)

``` python
from flask import Flask, request, jsonify
import tensorflow as tf
import requests
from tensorflow.keras.models import load_model
import numpy as np
import io

app = Flask(__name__)

model = load_model('Layer2.keras')

def predict_layer2(layer1_data):
    layer1_data = np.array(layer1_data) 
    predictions = model.predict(layer1_data)
    return predictions

@app.route('/Layer2', methods=['POST'])
def layer2_predict():
    data = request.get_json()
    if 'predictions' not in data:
        return jsonify({'error': 'No predictions data'}), 400

    try:
        layer1_data = data['predictions']
        predictions = predict_layer2(layer1_data)
        result = predictions.tolist()
    except Exception as e:
        return jsonify({'error': str(e)}), 500


    target_ip = 'http://127.0.0.1:8000/Layer3'
    try:
        response = requests.post(target_ip, json={'predictions': result})
        response.raise_for_status()
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        layer3_response = response.json()['predictions']
        print(layer3_response)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to send data to {target_ip}: {str(e)}'}), 500
    return jsonify({'predictions': layer3_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)
```


<br/>

### 장치 3 서버 구현(Layer3 - Result)

``` python
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model('Layer3.keras')

def predict_layer3(layer2_data):
    layer2_data = np.array(layer2_data)  
    predictions = model.predict(layer2_data)
    return predictions

@app.route('/Layer3', methods=['POST'])
def layer2_predict():
    data = request.get_json()
    if 'predictions' not in data:
        return jsonify({'error': 'No predictions data'}), 400

    try:
        layer2_data = data['predictions']
        predictions = predict_layer3(layer2_data)
        result = np.argmax(predictions).tolist() 

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    print(result)
    return jsonify({'predictions': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

<br/>
<br/>

## 결과 

유저 이미지 전송 -> 장치 1 (Layer 1 연산 후 결과값 장치 2에 전송 ) -> 장치 2 -> 장치 3 (결과를 장치2 -> 장치1 -> 유저에게 반환)

#### 유저 이미지 전송
![](https://tera.dscloud.me:8080/Images/Project/DeepLearningDistributedProcessingTest/UserSendIMG.png)

#### 장치 1
![](https://tera.dscloud.me:8080/Images/Project/DeepLearningDistributedProcessingTest/Layer1.png)


#### 장치 2
![](https://tera.dscloud.me:8080/Images/Project/DeepLearningDistributedProcessingTest/Layer2.png)


#### 장치 3 
![](https://tera.dscloud.me:8080/Images/Project/DeepLearningDistributedProcessingTest/Layer3.png)


#### 유저 결과값 확인
![](https://tera.dscloud.me:8080/Images/Project/DeepLearningDistributedProcessingTest/UserGetResult.png)