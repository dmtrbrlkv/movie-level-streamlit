import re
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class CleanSubs(BaseEstimator, TransformerMixin):
    HTML = r'<.*?>'
    TAG = r'{.*?}'
    COMMENTS = r'[\(\[][A-Z ]+[\)\]]'
    LETTERS = r'[^a-zA-Z]'
    SPACES = r'([ ])\1+'
    DOTS = r'[\.]+'

    @classmethod
    def clean_subs(cls, subs):
        subs = re.sub(cls.HTML, ' ', subs) #html тэги меняем на пробел
        subs = re.sub(cls.TAG, ' ', subs) #тэги меняем на пробел
        subs = re.sub(cls.COMMENTS, ' ', subs) #комменты меняем на пробел
        subs = re.sub(cls.LETTERS, ' ', subs) #все что не буквы меняем на пробел
        subs = re.sub(cls.SPACES, r'\1', subs) #повторяющиеся пробелы меняем на один пробел
        subs = re.sub(cls.DOTS, r'.', subs) #многоточие меняем на точку
        subs = subs.encode('ascii', 'ignore').decode() #удаляем все что не ascii символы
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
        # res = Parallel(n_jobs=8)(delayed(lambda x: x.lemma_)(token) for token in doc)
        tokens = [token.lemma_ for token in doc]
        return " ".join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['subs'] = X['subs'].parallel_apply(self.lemmatize)
        if y is None:
            return X
        else:
            return X, y


class WordsLevel(BaseEstimator, TransformerMixin):

    @staticmethod
    def load_vocabulary(vocabulary_path):
        with open(vocabulary_path) as f:
            full = f.readlines()

        full = [line.split()[0].lower() for line in full]

        a1_index = full.index('a1')
        a2_index = full.index('a2')
        b1_index = full.index('b1')
        b2_index = full.index('b2')
        c1_index = full.index('c1')

        a1 = set(full[a1_index + 1: a2_index])
        a2 = set(full[a2_index + 1: b1_index])
        b1 = set(full[b1_index + 1: b2_index])
        b2 = set(full[b2_index + 1: c1_index])
        c1 = set(full[c1_index + 1:])

        return a1, a2, b1, b2, c1

    def __init__(self):
        vocabulary_path='../data/vocabulary/american/full.txt'
        self.a1, self.a2, self.b1, self.b2, self.c1 = self.load_vocabulary(vocabulary_path)
        super().__init__()

    def count_word_level(self, text):
        words = text.split()
        a1_count = 0
        a2_count = 0
        b1_count = 0
        b2_count = 0
        c1_count = 0
        other_count = 0

        total = len(words)

        for word in words:
            if word in self.a1:
                a1_count += 1
            elif word in self.a2:
                a2_count += 1
            elif word in self.b1:
                b1_count += 1
            elif word in self.b2:
                b2_count += 1
            elif word in self.c1:
                c1_count += 1
            else:
                other_count += 1
        return a1_count, a2_count, b1_count, b2_count, c1_count, other_count, total

    @staticmethod
    def _expand_array(row):
        return pd.Series(row)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        stats = X['subs'].parallel_apply(self.count_word_level)
        stats = stats.parallel_apply(self._expand_array)
        stats.columns = ['a1_count', 'a2_count', 'b1_count', 'b2_count', 'c1_count', 'other_count', 'total']
        X = X.merge(stats, left_index=True, right_index=True)
        X['a1_coef'] = X['a1_count'] / X['total']
        X['a2_coef'] = X['a2_count'] / X['total']
        X['b1_coef'] = X['b1_count'] / X['total']
        X['b2_coef'] = X['b2_count'] / X['total']
        X['c1_coef'] = X['c1_count'] / X['total']
        X = X.drop(columns=['a1_count', 'a2_count', 'b1_count', 'b2_count', 'c1_count', 'total'])
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


from sklearn.decomposition import TruncatedSVD


class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=1500, min_df=5, max_df=0.7, PCA=None):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.PCA = PCA

    def fit(self, X, y=None):
        # self.max_features = max_features
        # self.min_df = min_df
        # self.max_df = max_df
        # self.PCA = PCA
        tfidfvectorizer = TfidfVectorizer(max_features=self.max_features, min_df=self.min_df, max_df=self.max_df, stop_words=stopwords.words('english'))
        # tfidfvectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
        # tfidfvectorizer = TfidfVectorizer(max_features=self.max_features, min_df=self.min_df, max_df=self.max_df)
        tfidfvectorizer.fit(X['subs'])
        self.tfidfvectorizer = tfidfvectorizer
        if self.PCA:
            tfidf_wm = self.tfidfvectorizer.transform(X['subs'])
            tfidf_tokens = self.tfidfvectorizer.get_feature_names_out()
            tfidf_tokens = ['tfidf_' + token for token in tfidf_tokens]
            df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(), columns = tfidf_tokens, index=X.index)
            svd = TruncatedSVD(n_components=self.PCA, random_state=12345)
            svd.set_output(transform='pandas')
            svd.fit(df_tfidfvect)
            self.svd = svd
        return self

    def transform(self, X, y=None):
        tfidf_wm = self.tfidfvectorizer.transform(X['subs'])
        tfidf_tokens = self.tfidfvectorizer.get_feature_names_out()
        tfidf_tokens = ['tfidf_' + token for token in tfidf_tokens]
        df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(), columns = tfidf_tokens, index=X.index)
        if self.PCA:
            df_tfidfvect = self.svd.transform(df_tfidfvect)
        X = X.merge(df_tfidfvect, left_index=True, right_index=True)
        if y is None:
            return X
        else:
            return X, y

from imblearn import FunctionSampler
from itertools import islice
from sklearn.utils import shuffle

def batched(iterable, chunk_size):
            iterator = iter(iterable)
            while chunk := tuple(islice(iterator, chunk_size)):
                yield chunk

def split_subs(X, y, n_splits=10):
    X_split  = []
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

split_subs_sampler = FunctionSampler(func=split_subs, kw_args={'n_splits': 10}, accept_sparse=False, validate=False)
