```python
import numpy as np
import json
import regex as re
import pandas as pd
```


```python
# making one large function from it to call above

def txt_to_pickle(filename):
    orig_file = f'../data/orig/{filename}'
    new_file = orig_file.replace('../data/orig/', '../data/clean_proc/')
    newsgroup_name = orig_file.replace('../data/orig/', '').replace('.txt', '')

    with open(new_file, 'w') as modified:
        with open(orig_file, 'r+', encoding="utf8", errors='ignore') as original:
            lines = original.readlines()
            for line in lines:
                # Remove all carets now. We are going to use this as a delimiter soon
                line = line.replace('^', '')
                if line.startswith("Newsgroup:"):
                    line = line.replace('Newsgroup: ', '')
                    line = line.replace('\n', '^')
                    modified.write(line)
                elif not line.startswith("document_id:") and not line.startswith("From:"):
                    line = line.replace('\n', '')
                    modified.write(line + " ")

    # Define column headers. Create blank file. Add col headers + original data to new file to get DF
    col_headers = "newsgroup^body"

    nm_idx = new_file.find('.txt')
    final_file_name = new_file[:nm_idx] + '_final' + new_file[nm_idx:]
    with open(final_file_name, 'w') as final_mod:
        with open(new_file, 'r+', encoding="utf8", errors='ignore') as original:
            lines = original.readlines()
            for line in lines:
                if line.startswith(newsgroup_name):
                    line = line.replace('Subject: ', '')
                    line = line.replace('Re: ', '')
                    line = line.replace('Re^2:', '')
                    line = line.replace('re: ', '')
                    final_mod.write(line)

    nm_idx = final_file_name.find('.txt')
    ultimate_file_name = final_file_name[:nm_idx] + '_ultimate' + final_file_name[nm_idx:]
    with open(ultimate_file_name, 'w') as ultimate_mod:
        with open(final_file_name, 'r+', encoding="utf8", errors='ignore') as original:
            ultimate_mod.write(col_headers + "\n")
            lines = original.readlines()
            for line in lines:
                line = line.replace(newsgroup_name, "\n" + newsgroup_name)
                ultimate_mod.write(line)

    df = pd.read_csv(ultimate_file_name, sep='^')
    df.to_pickle('../data/clean_pkl/' + newsgroup_name + '.pkl')

```


```python
import os
dirpath = '../data/orig/'
directory = os.fsencode(dirpath)
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     txt_to_pickle(filename)
```


```python
df = pd.read_pickle('../data/clean_pkl/talk.religion.misc.pkl')
print(df.sample(frac=1).reset_index(drop=True).head().to_markdown(tablefmt="grid"))
```



```python
# Filter out those incorrect newsgroup rows
final_df = final_df[final_df.newsgroup == 'politics']
print(final_df['newsgroup'].value_counts().to_markdown(tablefmt="grid"))
```


|          |   newsgroup |
|----------|-------------|
| politics |        5250 |




```python
final_df.to_pickle('../data/clean_pkl/to_use/politics.pkl')
```


```python
## Just simple for misc.forsale
df_misc = pd.read_pickle('../data/clean_pkl/misc.forsale.pkl')
df_misc['newsgroup'] = df_misc['newsgroup'].replace(['misc.forsale'], ['seller'])
```


```python
final_df_misc = df_misc[df_misc.newsgroup == 'seller']
```


```python
final_df_misc.to_pickle('../data/clean_pkl/to_use/seller.pkl')
```

### There were several additional sets that were needed to get the data into the correct form. Here are some that were in the run 01 but I copied over just for clean sake.


```python
from string import digits
df['body'] = df['body'].apply(lambda x: x.replace('documentId',''))
df['body'] = df['body'].apply(lambda x: x.replace('documentid',''))
# We're just going to remove the digits
df['body'] = df['body'].apply(lambda x: x.translate({ord(k): None for k in digits}))
```
