```python
import numpy as np
import json
import regex as re
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import utils
import seaborn as sns
import keras
import nltk
import random

from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from sklearn.naive_bayes import MultinomialNB
```


```python
# Load model
model_file = 'models/newsgroup_clean_model'
model = keras.models.load_model(model_file)
```

## Augmentation. Run 04 = Word Swap


```python
df = pd.read_pickle('data/dataframes/newsgroup_cleaned.pkl')
```


```python
df = df.dropna()
```


```python
def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

def swap_rejoin(x):
	if len(x) > 1:
		words = random_swap(x.split(), 1)
		sentence = ' '.join(words)
		return sentence
```


```python
df['subject'] = df['subject'].apply(lambda x: swap_rejoin(x))
print(df.sample(frac=1).reset_index(drop=True).head().to_markdown(tablefmt="grid"))
```


|    | newsgroup   | subject                                          |
|----|-------------|--------------------------------------------------|
|  0 | seller      | complet aix-ps2 repost manual best offer softwar |
|  1 | comp_elec   | summari x11r5 xon                                |
|  2 | sport       | stat al                                          |
|  3 | comp_elec   | confus doe appl give us whi messag               |
|  4 | comp_elec   | vga paradis                                      |




```python
all_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
# We'll use all
target_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
```


```python
df = df.dropna()
```


```python
# container for sentences
X = np.array([s for s in df['subject']])
# container for sentences
y = np.array([n for n in df['newsgroup']])
```


```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(df['newsgroup'])
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.25)

classes = np.unique(y_train)
mapping = dict(zip(classes, target_categories))

len(X_train), len(X_test), classes, mapping
```




    (6167,
     2056,
     array([0, 1, 2, 3, 4, 5, 6]),
     {0: 'sport',
      1: 'autos',
      2: 'religion',
      3: 'comp_elec',
      4: 'sci_med',
      5: 'seller',
      6: 'politics'})




```python
# model parameters
vocab_size = 1200
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
```


```python
# tokenize sentences
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# convert train dataset to sequence and pad sequences
train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

# convert validation dataset to sequence and pad sequences
validation_sequences = tokenizer.texts_to_sequences(X_test)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)
```


```python
# fit model
num_epochs = 20
history = model.fit(train_padded, y_train, 
                    epochs=num_epochs, verbose=1,
                    validation_split=0.3)

# predict values
pred = model.predict(validation_padded)
```

    Epoch 1/20
      1/135 [..............................] - ETA: 0s - loss: 0.9555 - accuracy: 0.6562135/135 [==============================] - 1s 5ms/step - loss: 1.0733 - accuracy: 0.6182 - val_loss: 1.2396 - val_accuracy: 0.5575
    Epoch 2/20
    135/135 [==============================] - 0s 4ms/step - loss: 1.0597 - accuracy: 0.6221 - val_loss: 1.2284 - val_accuracy: 0.5656
    Epoch 3/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.0526 - accuracy: 0.6240 - val_loss: 1.2258 - val_accuracy: 0.5678
    Epoch 4/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.0414 - accuracy: 0.6251 - val_loss: 1.2164 - val_accuracy: 0.5737
    Epoch 5/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.0278 - accuracy: 0.6390 - val_loss: 1.2195 - val_accuracy: 0.5683
    Epoch 6/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.0175 - accuracy: 0.6376 - val_loss: 1.2015 - val_accuracy: 0.5775
    Epoch 7/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.0089 - accuracy: 0.6453 - val_loss: 1.2028 - val_accuracy: 0.5797
    Epoch 8/20
    135/135 [==============================] - 0s 3ms/step - loss: 0.9992 - accuracy: 0.6460 - val_loss: 1.1910 - val_accuracy: 0.5808
    Epoch 9/20
    135/135 [==============================] - 1s 5ms/step - loss: 0.9893 - accuracy: 0.6499 - val_loss: 1.1969 - val_accuracy: 0.5856
    Epoch 10/20
    135/135 [==============================] - 0s 3ms/step - loss: 0.9791 - accuracy: 0.6564 - val_loss: 1.1823 - val_accuracy: 0.5824
    Epoch 11/20
    135/135 [==============================] - 0s 3ms/step - loss: 0.9689 - accuracy: 0.6573 - val_loss: 1.1840 - val_accuracy: 0.5754
    Epoch 12/20
    135/135 [==============================] - 1s 4ms/step - loss: 0.9625 - accuracy: 0.6657 - val_loss: 1.1715 - val_accuracy: 0.5883
    Epoch 13/20
    135/135 [==============================] - 0s 3ms/step - loss: 0.9531 - accuracy: 0.6610 - val_loss: 1.1651 - val_accuracy: 0.5883
    Epoch 14/20
    135/135 [==============================] - 1s 4ms/step - loss: 0.9421 - accuracy: 0.6687 - val_loss: 1.1594 - val_accuracy: 0.5905
    Epoch 15/20
    135/135 [==============================] - 0s 4ms/step - loss: 0.9349 - accuracy: 0.6701 - val_loss: 1.1574 - val_accuracy: 0.5894
    Epoch 16/20
    135/135 [==============================] - 0s 3ms/step - loss: 0.9287 - accuracy: 0.6719 - val_loss: 1.1502 - val_accuracy: 0.5916
    Epoch 17/20
    135/135 [==============================] - 1s 4ms/step - loss: 0.9184 - accuracy: 0.6791 - val_loss: 1.1650 - val_accuracy: 0.5943
    Epoch 18/20
    135/135 [==============================] - 1s 4ms/step - loss: 0.9108 - accuracy: 0.6786 - val_loss: 1.1563 - val_accuracy: 0.5948
    Epoch 19/20
    135/135 [==============================] - 1s 4ms/step - loss: 0.9070 - accuracy: 0.6905 - val_loss: 1.1600 - val_accuracy: 0.5975
    Epoch 20/20
    135/135 [==============================] - 1s 5ms/step - loss: 0.8985 - accuracy: 0.6884 - val_loss: 1.1430 - val_accuracy: 0.6035
    65/65 [==============================] - 0s 2ms/step



```python
import os

file_name = 'run_04'
plot_type = 'history'
model_name = 'newsgroup_clean'
#####
os.makedirs(f"images/{plot_type}", exist_ok=True)
os.makedirs(f"images/{plot_type}/{model_name}", exist_ok=True)
save_path = f'images/{plot_type}/{model_name}/{file_name}.png' 

utils.plot_history_and_save(history, save_path)
```


![png](/images/news/clean_run_04.png)


It does seem to be performing slightly better than our previous run, but this is still certainly nothing to write home about. Now, let's work with some real data and incorporate the body of these message into our next runs. We will save the model file incase we want to use it later in any form.


```python
# TensorFlow SavedModel format => .keras
model_file = 'models/newsgroup_clean_model'
model.save(model_file)
```


