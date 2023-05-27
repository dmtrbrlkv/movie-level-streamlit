import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier

from srt_processing import save_srt, subs_text

from tqdm import tqdm

from pandarallel import pandarallel


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
    from transformers import CleanSubs, LemmatizeSub, WordsLevel, Vectorizer, DropSubs
    globals_ = globals()
    globals_['CleanSubs'] = CleanSubs
    globals_['LemmatizeSub'] = LemmatizeSub
    globals_['WordsLevel'] = WordsLevel
    globals_['Vectorizer'] = Vectorizer
    globals_['DropSubs'] = DropSubs

    pipeline = joblib.load('best_pipe.ppl')
    return pipeline


@st.cache_resource(show_spinner=False)
def load_model():
    model = CatBoostClassifier().load_model('best_model.cbm')
    return model


init_parallel()
load_spacy_model()
pipeline = load_pipeline()
model = load_model()


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

def make_predict(subs_text):
    df = pd.DataFrame({'subs': [subs]}, index=[srt_file.name])
    df_transformed = pipeline.transform(df[['subs']])
    predict = model.predict(df_transformed)
    return predict[0, 0]

st.set_page_config(page_title='Уровень фильма')
st.title('Уровень фильма')

st.header('Файл с субтитрами')
srt_file = st.file_uploader('Файл с субтитрами', 'srt', label_visibility='hidden')

predict_btn = st.button('Узнать уровень фильма', use_container_width=True)

if predict_btn:
    subs = get_subs_text(srt_file)
    if subs is None:
        st.warning('Не удалось прочитать файл с субтитрами')
    else:
        with st.spinner(''):
            predict = make_predict(subs)
        st.success(f'Уровень фильма: {predict}')
