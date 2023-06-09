import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from pandarallel import pandarallel
from tqdm import tqdm

from srt_processing import subs_text, save_srt


@st.cache_resource(show_spinner=False)
def init_parallel():
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)


@st.cache_resource(show_spinner=False)
def load_nlp_models():
    from spacy.cli import download as spacy_download
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')


@st.cache_resource(show_spinner=False)
def load_pipeline():
    from transformers import CleanSubs, LemmatizeSub, Vectorizer, DropSubs, subs_sampler_x2, split_subs
    globals_ = globals()
    globals_['CleanSubs'] = CleanSubs
    globals_['LemmatizeSub'] = LemmatizeSub
    globals_['Vectorizer'] = Vectorizer
    globals_['DropSubs'] = DropSubs
    globals_['split_subs'] = split_subs
    globals_['subs_sampler_x2'] = subs_sampler_x2

    pipeline = joblib.load('best_pipeline.ppl')
    return pipeline


init_parallel()
load_nlp_models()
pipeline = load_pipeline()


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
    df = pd.DataFrame({'subs': [subs_text]})
    res = {}
    predict_proba = pipeline.predict_proba(df)
    for i in range(len(pipeline.classes_)):
        res[pipeline.classes_[i]] = predict_proba[0][i]
    return res


def process_text(subs_text):
    with st.spinner(''):
        predict_proba = make_predict_proba(subs_text)
        predict_proba = pd.DataFrame({'proba': predict_proba.values()}, index=predict_proba.keys())
        predict = predict_proba['proba'].idxmax()
        fig = plt.figure(figsize=(1, 1))
        ax = sns.barplot(x=predict_proba.index, y=predict_proba['proba'])
        ax.set(title='Вероятность уровня')
        ax.set(ylabel='')

    st.success(f'Уровень английского: {predict}')
    st.pyplot(fig)

st.set_page_config(page_title='Уровень английского в фильме')
st.title('Уровень английского в фильме')

types = ['Текст', 'Файл субтитров']
st.header('Входящие данные')
input_type = st.radio('', types, horizontal=True)

st.header(input_type)
if input_type == types[0]:
    text = st.text_area('')
else:
    srt_files = st.file_uploader('Файл с субтитрами', 'srt', label_visibility='hidden', accept_multiple_files=True)

predict_btn = st.button('Узнать уровень английского', use_container_width=True)

if predict_btn:
    if input_type == types[0]:
        process_text(text)
    else:
        for srt_file in srt_files:
            st.header(srt_file.name)
            subs = get_subs_text(srt_file)
            if subs is None:
                st.warning('Не удалось прочитать файл с субтитрами')
            else:
                process_text(subs)
