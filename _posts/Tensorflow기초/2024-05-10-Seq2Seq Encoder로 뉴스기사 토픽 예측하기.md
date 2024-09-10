---
title: 10. Seq2Seq Encoder로 뉴스기사 토픽 예측
date: 2024-05-10T19:30:00+09:00
categories: [Tensorflow]
tags:
  [
    Tensorflow,
    Seq2Seq,
  ]
pin: true
math: true
mermaid: true
---

## 데이터 소개 

- reuters  뉴스 기사 Dataset 사용
- reuters dataset은  11,228개의 뉴스기사와 46개의 주제를 가지고 있다.
- 모든 단어는 빈도에 따른 정수로 교체 되어있다. 
- 관례적으로 정수 0은 특정 단어를 나타내는 것으로 사용하지 않는다.
- 학습 데이터와 정답 데이터는, 각각 아래와 같다.

1. train_text
``` python
[
	[1, 20, 432, 12, 312, ...],
	[42, 21, 111, 1213, ...],	  
	...
]
```

2. train_answer
``` python
[
	3, 4, 3, ..., 25, 3
]
```

<br/>
<br/>

## 최종 목표 
- Seq2Seq 기반 모델의 이해
- word embedding 이해 
- 시계열 데이터 학습 이해

<br/>
<br/>

## 전처리
``` python
"""
데이터 불러오기 
"""

reuters = keras.datasets.reuters
(train_text, train_topic), (test_text, test_topic) = reuters.load_data()
```


``` python
"""
train_textm test_text에 등장한 정수 확인
"""

all_words =  set([word for text in train_text for word in text]) 
						| set([word for text in test_text for word in text])

print(sorted(all_words)[:5]) # [1, 2, 4, 5, 6] 
print(sorted(all_words)[-5:]) # [30977, 30978, 30979, 30980, 30981]
```

``` python
"""
train_text와 test_text를 bow형태로 변경
"""

train_bow_text = tf.keras.preprocessing.sequence.pad_sequences(train_text,
																  value = 0)
text_bow_text = tf.keras.preprocessing.sequence.pad_sequences(test_text,
																value = 0)

print(train_bow_text.shape, text_bow_text.shape)
# ((8982, 2376), (2246, 1032))
```

``` python
"""
train 과 test에 나온 word들의 count 측정
"""
import collections

# 딕셔너리의 확장으로, 해시 가능한 객체의 개수를 세는 것을 도와준다. 여기서는 단어의 출현 횟수를 세기 휘해 사용
word_count = collections.Counter()
 
for text in train_text:
	word_count.update(text)
for text in test_text:
	word_count.update(text)

word_count.most_common(10)

"""
[(4, 82723),
 (5, 42393),
 (6, 40350),
 (7, 33157),
 (8, 29978),
 (9, 29956),
 (10, 29581),
 (11, 20141),
 (12, 16668),
 (13, 15224)]
"""
```

``` python
"""
word_count를 이용해 n번 이하로 나온 word를 삭제
"""

def cut_by_count(texts, n):
	return np.array([[word for word in text if word_count[word] >= n] 
										for text in texts], dtype = object)

train_cut_text = cut_by_count(train_text, 20)
test_cut_text = cut_by_count(test_text, 20)


print(train_cut_text.shape) # (8982,)
print(test_cut_text.shape) # (2246,)

```

``` python
"""
train/test_cut_text 를 bow형태로 반환
"""

train_cut_bow_text = 
tf.keras.preprocessing.sequence.pad_sequences(train_cut_text, value = 0)

test_cut_bow_text = 
tf.keras.preprocessing.sequence.pad_sequence(test_cut_text, value = 0)

print(train_cut_bow_text.shape, test_cut_bow_text.shape)
# ((8982, 2266), (2246, 995))
```

``` python
"""
train/test_text에 길이 제한을 줘서 bow 생성
"""

train_cut_bow_text2 = tf.keras.preprocessing.sequence.pad_sequences(train_text,
																	value = 0, 
																	maxlen = 200)

test_cut_bow_text2 = tf.keras.preprocessing.sequence.pad_sequences(test_text,
																  value = 0,
																  maxlen = 200)

print(train_cut_bow_text2.shape, test_cut_bow_text2.shape)
# ((8982, 200), (2246, 200))
```

``` python
"""
topic을 onehot encoding으로 변경
"""
train_onehot_topic = keras.utils.to_categorical(train_topic)
test_onehot_topic = keras.utils.to_categorical(test_topic)

print(train_onehot_topic.shape, test_onehot_topic.shape)
# (8982, 46) (2246, 46)
```

``` python
"""
topic 인덱스를 text label로 변환
"""
raw_labels = ['cocoa','grain','vegoil','earn','acq','wheat','copper','housing','money-supply',
'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin', 'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead']


def topic2label(idx):
	return raw_labels[idx]
```

``` python
"""
reuters.get_word_index()를 이용해 text decoding
"""

index_word = { y : x for x, y in reuters.get_word_index().items()}

def bow2text(bow):
	return " ".join([index_word[idx] for idx in bow])
```


<br/>
<br/>
<br/>
<br/>

## 모델링

``` python
"""
Encoder를 이용해 classification 모델 구축
"""

from keras.layers import Input, GRU, Embedding, Denses
from keras.models improt Model

def seq2seq():
	inputs_x_bow = Input(shape = (200, ))
	x = Embedding(len(index_word) + 1, 120)(inputs_x_bow)
	context_vector = GRU(units = 64)(x)

	y = Dense(units = 46, activation = 'softmax')(context_vector)
	model = Model(inputs_x_bow, y)

	model.compile(loss = 'categorical_crossentropy', 
				  optimizer = 'adam', 
				  metrics = ['accuracy'])

	return model
```

``` python
"""
모델 학습
"""

hist = model.fit(
	train_cut_bow_text2, train_onehot_topic,
	validation_data = (
		test_cut_bow_text2,
		test_onehot_topicm
	),
	epochs = 20,
	verbose = 1
)
```


