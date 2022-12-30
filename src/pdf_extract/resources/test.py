import pdfplumber, os
import os.path as osp
from pdf_extract.config import global_config as glob
from importlib import reload
from pdf_extract.resources import preprocessor as preproc

reload(preproc)


# Preprocess corpus:
cleaner = preproc.clean_text(language='english', lemma = False, stem = False)

#X_cl = cleaner.fit_transform(X)

combine_fct = lambda x: '__label__{}'.format(x.label)

train_set['fasttext'] = train_set.apply(combine_fct, axis = 1)

X = train_set[['fasttext', 'text']].reset_index(drop=True)
X.head()
