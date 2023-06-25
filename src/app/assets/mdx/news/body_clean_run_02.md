# Newsgroup 20 Dataset with Body - Clean Run 02

---

After seeing some pretty lackluster results from the subject only dataset, lets now go ahead and use the body for the text that we will train on. This should offer us a lot more data, but it was a struggle wrangling it all into the dataframes. I chose to go ahead and start from the text files themselves, and I will incode some snippets of the transformations below.

---

```python
import numpy as np
import regex as re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import nltk
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
```


```python
# Load model
model_file = 'models/newsgroup_body_clean_model'
model = keras.models.load_model(model_file)
```

## We will augment out data now. Run 02 = Synonym Replacement


```python
df = pd.read_pickle('../data/dataframes/newsgroup_body_cleaned_exploded.pkl')
```


```python
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
df['exploded_body'] = df['exploded_body'].apply(lambda x: utils.replace_rejoin(x))
print(df.sample(frac=1).reset_index(drop=True).loc[:,['newsgroup', 'exploded_body']].head().to_markdown())
```


<style>.container { width:100% !important; }</style>


|    | newsgroup   | exploded_body                                                                                                       |
|---:|:------------|:--------------------------------------------------------------------------------------------------------------------|
|  0 | religion    | apr god promis chronicl fail fill asa said unto pick up ye asa judah benjamin godhead ye ye seek found ye forsak fo |
|  1 | comp_elec   | page setup notepad previou articl joel jachhawaiiedu joel aycock drop a line struggl margin problem age well final  |
|  2 | politics    | convict ye survey present accord mr cramer valu call median- one use thi make us believ manly plu sex partner manly |
|  3 | sci_med     | plato sinc whole enterpris philosophi wa essenti defin although got hi suffice wrong definit identifi import intee  |
|  4 | comp_elec   | dx eisa bu s unused speed local bu -- ani idea andrea dist institut fuer computersystem eth zuerich electronic ma   |



```python
all_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
# We'll use all
target_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
```


```python
# container for sentences
X = np.array([s for s in df['exploded_body']])
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




    (51882,
     17295,
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

    2023-06-21 00:29:07.052575: W tensorflow/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 17432160 exceeds 10% of free system memory.


    Epoch 1/20
    1135/1135 [==============================] - 5s 3ms/step - loss: 2.1807 - accuracy: 0.4020 - val_loss: 1.3669 - val_accuracy: 0.5232
    Epoch 2/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 1.1810 - accuracy: 0.5877 - val_loss: 0.9958 - val_accuracy: 0.6583
    Epoch 3/20
    1135/1135 [==============================] - 5s 4ms/step - loss: 0.8793 - accuracy: 0.6990 - val_loss: 0.7855 - val_accuracy: 0.7329
    Epoch 4/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.7055 - accuracy: 0.7607 - val_loss: 0.6652 - val_accuracy: 0.7784
    Epoch 5/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.6038 - accuracy: 0.7972 - val_loss: 0.6038 - val_accuracy: 0.7987
    Epoch 6/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.5404 - accuracy: 0.8177 - val_loss: 0.5597 - val_accuracy: 0.8115
    Epoch 7/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.4974 - accuracy: 0.8321 - val_loss: 0.5313 - val_accuracy: 0.8202
    Epoch 8/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.4664 - accuracy: 0.8427 - val_loss: 0.5147 - val_accuracy: 0.8280
    Epoch 9/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.4438 - accuracy: 0.8496 - val_loss: 0.5014 - val_accuracy: 0.8297
    Epoch 10/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.4244 - accuracy: 0.8560 - val_loss: 0.4904 - val_accuracy: 0.8353
    Epoch 11/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.4088 - accuracy: 0.8627 - val_loss: 0.4824 - val_accuracy: 0.8359
    Epoch 12/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.3961 - accuracy: 0.8648 - val_loss: 0.4782 - val_accuracy: 0.8382
    Epoch 13/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.3851 - accuracy: 0.8675 - val_loss: 0.4762 - val_accuracy: 0.8373
    Epoch 14/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3761 - accuracy: 0.8711 - val_loss: 0.4723 - val_accuracy: 0.8421
    Epoch 15/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3679 - accuracy: 0.8736 - val_loss: 0.4753 - val_accuracy: 0.8421
    Epoch 16/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3606 - accuracy: 0.8758 - val_loss: 0.4741 - val_accuracy: 0.8422
    Epoch 17/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3546 - accuracy: 0.8789 - val_loss: 0.4749 - val_accuracy: 0.8427
    Epoch 18/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.3496 - accuracy: 0.8804 - val_loss: 0.4749 - val_accuracy: 0.8424
    Epoch 19/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.3430 - accuracy: 0.8823 - val_loss: 0.4765 - val_accuracy: 0.8430
    Epoch 20/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3397 - accuracy: 0.8823 - val_loss: 0.4821 - val_accuracy: 0.8353
    541/541 [==============================] - 1s 1ms/step



```python
import os

file_name = 'run_02'
plot_type = 'history'
model_name = 'newsgroup_body_clean'
#####
os.makedirs(f"images/{plot_type}", exist_ok=True)
os.makedirs(f"images/{plot_type}/{model_name}", exist_ok=True)
save_path = f'images/{plot_type}/{model_name}/{file_name}.png' 

utils.plot_history_and_save(history, save_path)
```


![png](clean_run_02_files/clean_run_02_12_0.png)



```python
# reviews on which we need to predict
sentence = ["son of genuine vinyl records 4sale"]

# convert to a sequence
sequences = tokenizer.texts_to_sequences(sentence)

# pad the sequence
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# preict the label
print(model.predict(padded))
```

    1/1 [==============================] - 0s 19ms/step
    [[0.36069816 0.63863254 0.38042638 0.34472513 0.38515127 0.27881363
      0.4187051 ]]



```python
# TensorFlow SavedModel format => .keras
model_file = 'models/newsgroup_body_clean_model'
model.save(model_file)
```

