import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
from catboost import CatBoostClassifier

from srt_processing import save_srt, subs_text

from tqdm import tqdm

from pandarallel import pandarallel
import matplotlib.pyplot as plt


@st.cache_resource(show_spinner=False)
def init_parallel():
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)


@st.cache_resource(show_spinner=False)
def load_spacy_model():
    from spacy.cli import download as spacy_download
    spacy_download('en_core_web_sm')


@st.cache_resource(show_spinner=False)
def load_pipeline():
    from transformers import CleanSubs, LemmatizeSub, WordsLevel, Vectorizer, DropSubs, split_subs
    globals_ = globals()
    globals_['CleanSubs'] = CleanSubs
    globals_['LemmatizeSub'] = LemmatizeSub
    globals_['WordsLevel'] = WordsLevel
    globals_['Vectorizer'] = Vectorizer
    globals_['DropSubs'] = DropSubs
    globals_['split_subs'] = split_subs

    pipeline = joblib.load('best_pipe.ppl')
    return pipeline


# @st.cache_resource(show_spinner=False)
# def load_model():
#     model = CatBoostClassifier().load_model('best_model.cbm')
#     return model


init_parallel()
load_spacy_model()
pipeline = load_pipeline()
# model = load_model()


def get_subs_text(srt_file):
    if srt_file is None:
        st.warning('Загрузите файл с субтитрами')
        exit(0)

    try:
        srt_file_path = save_srt(srt_file.name, srt_file.getvalue())
        subs = subs_text(srt_file_path)
    except:
        subs = None

    return subs


def make_predict_proba(subs_text):
    df = pd.DataFrame({'subs': [subs]}, index=[srt_file.name])
    res = {}
    predict_proba = pipeline.predict_proba(df)
    for i in range(len(pipeline.classes_)):
        res[pipeline.classes_[i]] = predict_proba[0][i]
    return res


st.set_page_config(page_title='Уровень английского в фильме')
st.title('Уровень английского в фильме')

st.header('Файл с субтитрами')
srt_files = st.file_uploader('Файл с субтитрами', 'srt', label_visibility='hidden', accept_multiple_files=True)

predict_btn = st.button('Узнать уровень английского', use_container_width=True)

if predict_btn:
    for srt_file in srt_files:
        st.header(srt_file.name)
        subs = get_subs_text(srt_file)
        if subs is None:
            st.warning('Не удалось прочитать файл с субтитрами')
        else:
            with st.spinner(''):
                predict_proba = make_predict_proba(subs)
                predict_proba = pd.DataFrame({'proba': predict_proba.values()}, index=predict_proba.keys())
                predict = predict_proba['proba'].idxmax()
                fig = plt.figure(figsize=(1, 1))
                ax = sns.barplot(x=predict_proba.index, y=predict_proba['proba'])
                ax.set(title='Вероятность уровня')
                ax.set(ylabel='')

            st.success(f'Уровень английского: {predict}')
            st.pyplot(fig)
