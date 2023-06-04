import re
from itertools import islice

import pandas as pd
import spacy
from imblearn import FunctionSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
import nltk


class CleanSubs(BaseEstimator, TransformerMixin):
    HTML = r'<.*?>'
    TAG = r'{.*?}'
    COMMENTS = r'[\(\[][A-Z ]+[\)\]]'
    LETTERS = r'[^a-zA-Z\']'
    SPACES = r'([ ])\1+'
    DOTS = r'[\.]+'

    @classmethod
    def clean_subs(cls, subs):
        subs = re.sub(cls.HTML, ' ', subs)  # html тэги меняем на пробел
        subs = re.sub(cls.TAG, ' ', subs)  # тэги меняем на пробел
        subs = re.sub(cls.COMMENTS, ' ', subs)  # комменты меняем на пробел
        subs = re.sub(cls.LETTERS, ' ', subs)  # все что не буквы меняем на пробел
        subs = re.sub(cls.SPACES, r'\1', subs)  # повторяющиеся пробелы меняем на один пробел
        subs = re.sub(cls.DOTS, r'.', subs)  # многоточие меняем на точку
        subs = subs.encode('ascii', 'ignore').decode()  # удаляем все что не ascii символы
        # subs = ".".join(subs.lower().split('.')[1:-1]) #удаляем первый и последний субтитр (обычно это реклама)
        return subs.lower()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['subs'] = X['subs'].parallel_apply(self.clean_subs)
        return X


class LemmatizeSub(BaseEstimator, TransformerMixin):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'tok2vec'])

    @classmethod
    def lemmatize(cls, subs):
        doc = cls.nlp(subs)
        tokens = nltk.word_tokenize(subs)
        return ' '.join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['subs'] = X['subs'].parallel_apply(self.lemmatize)
        if y is None:
            return X
        else:
            return X, y


class DropSubs(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X = X.drop(columns=['subs'])
        if y is None:
            return X
        else:
            return X, y


class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=1500, min_df=7, max_df=0.7, PCA=None):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.PCA = PCA

    def fit(self, X, y=None):
        tfidfvectorizer = TfidfVectorizer(max_features=self.max_features, min_df=self.min_df, max_df=self.max_df,
                                          stop_words=stopwords.words('english'))
        tfidfvectorizer.fit(X['subs'])
        self.tfidfvectorizer = tfidfvectorizer
        if self.PCA:
            tfidf_wm = self.tfidfvectorizer.transform(X['subs'])
            tfidf_tokens = self.tfidfvectorizer.get_feature_names_out()
            tfidf_tokens = ['tfidf_' + token for token in tfidf_tokens]
            df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens, index=X.index)
            svd = TruncatedSVD(n_components=self.PCA, random_state=12345)
            svd.set_output(transform='pandas')
            svd.fit(df_tfidfvect)
            self.svd = svd
        return self

    def transform(self, X, y=None):
        tfidf_wm = self.tfidfvectorizer.transform(X['subs'])
        tfidf_tokens = self.tfidfvectorizer.get_feature_names_out()
        tfidf_tokens = ['tfidf_' + token for token in tfidf_tokens]
        df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens, index=X.index)
        if self.PCA:
            df_tfidfvect = self.svd.transform(df_tfidfvect)
        X = X.merge(df_tfidfvect, left_index=True, right_index=True)
        if y is None:
            return X
        else:
            return X, y


def batched(iterable, chunk_size):
    iterator = iter(iterable)
    while chunk := tuple(islice(iterator, chunk_size)):
        yield chunk


def split_subs(X, y, n_splits=10):
    X_split = []
    y_split = []
    for index, row in X.iterrows():
        subs = row['subs'].split()
        split_subs = np.array_split(subs, n_splits)
        for split in split_subs:
            X_split.append(' '.join(split))
            y_split.append(y.loc[index])

    X_split = pd.DataFrame({'subs': X_split})
    y_split = pd.Series(y_split)

    return shuffle(X_split, y_split, random_state=12345)


subs_sampler_x2 = FunctionSampler(func=split_subs, kw_args={'n_splits': 2}, accept_sparse=False, validate=False)
