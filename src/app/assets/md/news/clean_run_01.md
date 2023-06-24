```python
import numpy as np
import regex as re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
```

```python
import os
dirpath = 'data/pkl/to_use/'
directory = os.fsencode(dirpath)
df = pd.DataFrame(columns=['newsgroup', 'subject'])
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     name, ext = filename.split('.')
     if filename.endswith(".pkl"): 
         df_ = pd.read_pickle(dirpath + filename) 
         df = pd.merge(
            df, df_, how="outer"
        )
         continue
     else:
         continue
```


```python
df.shape
```




    (37660, 2)




```python
df['subject'] = [utils.clean_text(s) for s in df['subject']]
```


```python
# Drop dups
df.drop_duplicates(inplace = True)
```


```python
df.shape
```




    (8228, 2)




```python
df.to_pickle('data/dataframes/newsgroup_cleaned.pkl')
```


```python
all_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
# We'll use all
target_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
```


```python
# container for sentences
X = np.array([s for s in df['subject']])
# container for sentences
y = np.array([n for n in df['newsgroup']])
```


```python
encoder = LabelEncoder()
y = encoder.fit_transform(df['newsgroup'])
```


```python
y.shape
```




    (8228,)




```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.25)

classes = np.unique(y_train)
mapping = dict(zip(classes, target_categories))

len(X_train), len(X_test), classes, mapping
```

    (6171,
     2057,
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
### Final layer must be same as y.shape output -> # categories
```


```python
# model initialization
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(7, activation='sigmoid')
])
model._name = "NewsgroupCleanBOW"
# compile model
# categorical-cross-entropy requires labels one-hot-encoded. sparse = as ints. binary = t/f
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model summary
print(model.summary())
```

    Model: "NewsgroupCleanBOW"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding_1 (Embedding)     (None, 120, 16)           19200     
                                                                     
     global_average_pooling1d_1   (None, 16)               0         
     (GlobalAveragePooling1D)                                        
                                                                     
     dense_2 (Dense)             (None, 24)                408       
                                                                     
     dense_3 (Dense)             (None, 7)                 175       
                                                                     
    =================================================================
    Total params: 19,783
    Trainable params: 19,783
    Non-trainable params: 0
    _________________________________________________________________
    None



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
    135/135 [==============================] - 2s 5ms/step - loss: 1.8095 - accuracy: 0.4105 - val_loss: 1.6940 - val_accuracy: 0.4368
    Epoch 2/20
    135/135 [==============================] - 1s 5ms/step - loss: 1.7105 - accuracy: 0.4239 - val_loss: 1.6889 - val_accuracy: 0.4368
    Epoch 3/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.7062 - accuracy: 0.4239 - val_loss: 1.6866 - val_accuracy: 0.4368
    Epoch 4/20
    135/135 [==============================] - 0s 4ms/step - loss: 1.7027 - accuracy: 0.4239 - val_loss: 1.6832 - val_accuracy: 0.4368
    Epoch 5/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.6986 - accuracy: 0.4239 - val_loss: 1.6779 - val_accuracy: 0.4368
    Epoch 6/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.6916 - accuracy: 0.4239 - val_loss: 1.6716 - val_accuracy: 0.4368
    Epoch 7/20
    135/135 [==============================] - 1s 6ms/step - loss: 1.6792 - accuracy: 0.4239 - val_loss: 1.6581 - val_accuracy: 0.4368
    Epoch 8/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.6572 - accuracy: 0.4239 - val_loss: 1.6333 - val_accuracy: 0.4368
    Epoch 9/20
    135/135 [==============================] - 0s 4ms/step - loss: 1.6175 - accuracy: 0.4239 - val_loss: 1.5935 - val_accuracy: 0.4368
    Epoch 10/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.5682 - accuracy: 0.4263 - val_loss: 1.5511 - val_accuracy: 0.4411
    Epoch 11/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.5194 - accuracy: 0.4429 - val_loss: 1.5110 - val_accuracy: 0.4503
    Epoch 12/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.4755 - accuracy: 0.4612 - val_loss: 1.4785 - val_accuracy: 0.4692
    Epoch 13/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.4356 - accuracy: 0.4742 - val_loss: 1.4502 - val_accuracy: 0.4719
    Epoch 14/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.4041 - accuracy: 0.4804 - val_loss: 1.4345 - val_accuracy: 0.4789
    Epoch 15/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.3806 - accuracy: 0.4844 - val_loss: 1.4073 - val_accuracy: 0.4768
    Epoch 16/20
    135/135 [==============================] - 1s 6ms/step - loss: 1.3598 - accuracy: 0.4876 - val_loss: 1.3954 - val_accuracy: 0.4822
    Epoch 17/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.3384 - accuracy: 0.4890 - val_loss: 1.3817 - val_accuracy: 0.4811
    Epoch 18/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.3221 - accuracy: 0.4890 - val_loss: 1.3721 - val_accuracy: 0.4827
    Epoch 19/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.3126 - accuracy: 0.4929 - val_loss: 1.3611 - val_accuracy: 0.4897
    Epoch 20/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.2951 - accuracy: 0.5020 - val_loss: 1.3528 - val_accuracy: 0.4957
    65/65 [==============================] - 0s 1ms/step



```python
import os

file_name = 'run_01'
plot_type = 'history'
model_name = 'newsgroup_clean'
#####
os.makedirs(f"images/{plot_type}", exist_ok=True)
os.makedirs(f"images/{plot_type}/{model_name}", exist_ok=True)
save_path = f'images/{plot_type}/{model_name}/{file_name}.png' 

utils.plot_history_and_save(history, save_path)
```


![png](/images/news/clean_run_01.png)



```python
# TensorFlow SavedModel format => .keras
model_file = 'models/newsgroup_clean_model'
model.save(model_file)
```

```python
# need a sentence to predict on
sentence = ["son of genuine vinyl records 4sale"]
# convert to sequence
sequences = tokenizer.texts_to_sequences(sentence)
# pad sequence
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# preict label
print(model.predict(padded))
```

    1/1 [==============================] - 0s 23ms/step
    [[0.42751697 0.44417906 0.43653548 0.3826555  0.4249938  0.48741317
      0.43762988]]

