```python
import numpy as np
import json
import regex as re
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import nltk
import utils


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

## We will augment out data now. Run 02 = Synonym Replacement


```python
df = pd.read_pickle('data/dataframes/newsgroup_cleaned.pkl')
```


```python
df['subject'] = df['subject'].apply(lambda x: utils.replace_rejoin(x))
print(df.sample(frac=1).reset_index(drop=True).head().to_markdown(tablefmt="grid"))
```

    +----+-------------+------------------------------------+
    |    | newsgroup   | subject                            |
    +====+=============+====================================+
    |  0 | comp_elec   | w4w publish envelop dj550c 500c    |
    +----+-------------+------------------------------------+
    |  1 | sport       | opinion eli denni render           |
    +----+-------------+------------------------------------+
    |  2 | autos       | point helmet law point megahertz b |
    +----+-------------+------------------------------------+
    |  3 | comp_elec   | centris610 trouble                 |
    +----+-------------+------------------------------------+
    |  4 | sport       | cub gritty april 6th               |
    +----+-------------+------------------------------------+


---



|    | newsgroup   | subject                            |
|----|-------------|------------------------------------|
|  0 | comp_elec   | w4w publish envelop dj550c 500c    |
|  1 | sport       | opinion eli denni render           |
|  2 | autos       | point helmet law point megahertz b |
|  3 | comp_elec   | centris610 trouble                 |
|  4 | sport       | cub gritty april 6th               |


---


| Item         | Price     | # In stock |
|--------------|-----------|------------|
| Juicy Apples | 1.99      | *7*        |
| Bananas      | **1.89**  | 5234       |



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
# fit model
num_epochs = 20
history = model.fit(train_padded, y_train, 
                    epochs=num_epochs, verbose=1,
                    validation_split=0.3)

# predict values
pred = model.predict(validation_padded)
```

    Epoch 1/20
    135/135 [==============================] - 2s 7ms/step - loss: 1.8397 - accuracy: 0.4063 - val_loss: 1.8383 - val_accuracy: 0.4228
    Epoch 2/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.7779 - accuracy: 0.4184 - val_loss: 1.7834 - val_accuracy: 0.4303
    Epoch 3/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.7320 - accuracy: 0.4316 - val_loss: 1.7465 - val_accuracy: 0.4357
    Epoch 4/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.7004 - accuracy: 0.4367 - val_loss: 1.7193 - val_accuracy: 0.4368
    Epoch 5/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.6749 - accuracy: 0.4390 - val_loss: 1.6947 - val_accuracy: 0.4482
    Epoch 6/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.6523 - accuracy: 0.4473 - val_loss: 1.6811 - val_accuracy: 0.4395
    Epoch 7/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.6345 - accuracy: 0.4503 - val_loss: 1.6640 - val_accuracy: 0.4471
    Epoch 8/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.6184 - accuracy: 0.4533 - val_loss: 1.6462 - val_accuracy: 0.4509
    Epoch 9/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.6045 - accuracy: 0.4543 - val_loss: 1.6334 - val_accuracy: 0.4579
    Epoch 10/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.5903 - accuracy: 0.4568 - val_loss: 1.6230 - val_accuracy: 0.4654
    Epoch 11/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.5772 - accuracy: 0.4587 - val_loss: 1.6107 - val_accuracy: 0.4584
    Epoch 12/20
    135/135 [==============================] - 1s 6ms/step - loss: 1.5707 - accuracy: 0.4605 - val_loss: 1.6066 - val_accuracy: 0.4552
    Epoch 13/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.5534 - accuracy: 0.4628 - val_loss: 1.5983 - val_accuracy: 0.4509
    Epoch 14/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.5410 - accuracy: 0.4654 - val_loss: 1.5805 - val_accuracy: 0.4746
    Epoch 15/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.5291 - accuracy: 0.4693 - val_loss: 1.5743 - val_accuracy: 0.4627
    Epoch 16/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.5170 - accuracy: 0.4705 - val_loss: 1.5635 - val_accuracy: 0.4746
    Epoch 17/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.5050 - accuracy: 0.4774 - val_loss: 1.5585 - val_accuracy: 0.4746
    Epoch 18/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.4916 - accuracy: 0.4774 - val_loss: 1.5420 - val_accuracy: 0.4741
    Epoch 19/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.4766 - accuracy: 0.4807 - val_loss: 1.5353 - val_accuracy: 0.4741
    Epoch 20/20
    135/135 [==============================] - 1s 5ms/step - loss: 1.4636 - accuracy: 0.4853 - val_loss: 1.5238 - val_accuracy: 0.4779
    65/65 [==============================] - 0s 2ms/step



```python
import os

file_name = 'run_02'
plot_type = 'history'
model_name = 'newsgroup_clean'
#####
os.makedirs(f"images/{plot_type}", exist_ok=True)
os.makedirs(f"images/{plot_type}/{model_name}", exist_ok=True)
save_path = f'images/{plot_type}/{model_name}/{file_name}.png' 

utils.plot_history_and_save(history, save_path)
```


![png](/images/news/clean_run_02.png)



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
model_file = 'models/newsgroup_clean_model'
model.save(model_file)
```


