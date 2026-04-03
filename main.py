import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import re
import string
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk import ngrams
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import lightgbm as lgb

# --- إصلاح 1: تحميل موارد NLTK ---
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# --- المتغيرات العامة ---
EN_STOPWORDS = set(stopwords.words('english'))
AR_STOPWORDS = set(stopwords.words('arabic'))
ALL_STOPWORDS = EN_STOPWORDS.union(AR_STOPWORDS)
TEXT_LEN_THRESHOLD = 50
UNIQUE_RATIO_THRESHOLD = 0.8
FONT_PATH = None 

# --- إصلاح 2: إدارة الـ Session State لمنع ضياع الداتا ---
if "df" not in st.session_state:
    st.session_state.df = None
if 'suggested_models_list' not in st.session_state:
    st.session_state.suggested_models_list = []

# --- دالة تحميل الملفات (مع إصلاح Data.seek) ---
def load_file(Data):
    name = Data.name.lower()
    Data.seek(0)
    content = Data.read()
    if not content.strip():
        raise ValueError("Uploaded file is empty!")
    Data.seek(0)

    if name.endswith(".csv"):
        encodings = ['utf-8', 'utf-8-sig', 'cp1256']
        separators = [',', ';', '\t']
        for enc in encodings:
            for sep in separators:
                try:
                    Data.seek(0)
                    df_res = pd.read_csv(Data, encoding=enc, sep=sep)
                    if df_res.shape[1] > 0: return df_res
                except: continue
        raise ValueError("Failed to read CSV.")
    elif name.endswith(".xlsx"):
        return pd.read_excel(Data, engine='openpyxl')
    elif name.endswith(".xls"):
        return pd.read_excel(Data)
    elif name.endswith(".json"):
        try: return pd.read_json(Data, lines=True)
        except: return pd.read_json(Data)
    elif name.endswith(".txt"):
        text = content.decode("utf-8", errors='ignore')
        return pd.DataFrame({"text": text.splitlines()})
    return None

# --- الدوال الخاصة بك (NLP) ---
def normalize_arabic(text):
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[ؤئ]', 'و', text)
    text = re.sub(r'ة', 'ه', text)
    return re.sub(r'ـ+', '', text)

def universal_strong_cleaner(text):
    if pd.isna(text) or not isinstance(text, str): return ""
    text = text.strip().lower()
    text = re.sub(r'http\S+|www\S+|<.*?>|\S+@\S+', '', text)
    text = normalize_arabic(text)
    text = re.sub(r'\d+', '', text)
    punctuation = string.punctuation + "«»،؛؟ـ…"
    text = text.translate(str.maketrans('', '', punctuation))
    tokens = text.split()
    lang = "ar" if re.search(r'[\u0600-\u06FF]', text) else "en"
    stop_words = AR_STOPWORDS if lang == "ar" else EN_STOPWORDS
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

# --- Sidebar ---
with st.sidebar:
    st.title("Auto AI System")
    st.image("loggo.png")
    Data = st.file_uploader("Upload CSV, Excel, JSON or TXT", type=["csv","xls","xlsx","json","txt"])
    
    if Data:
        try:
            st.session_state.df = load_file(Data)
            st.success("Data loaded")
        except Exception as e:
            st.error(f"Error: {e}")

    df = st.session_state.df
    model_choice_auto = ""

    if Data and df is not None:
        name = Data.name
        obj_cols = df.select_dtypes(include=['object']).columns
        if len(obj_cols) > 0:
            avg_len = df[obj_cols].astype(str).apply(lambda x: x.str.len().mean()).mean()
            unique_ratio = df[obj_cols].nunique().sum() / (len(df) * len(obj_cols))
            if name.endswith(".txt") or avg_len > TEXT_LEN_THRESHOLD or unique_ratio > UNIQUE_RATIO_THRESHOLD:
                model_choice_auto = "NLP"
            else:
                model_choice_auto = "ML"
        st.success(f"Detected: {model_choice_auto}")

    choise = st.selectbox("Service", [" ", "Data Analysis", "Auto AI system", "Evaluation Visualization"])

# --- معالجة الداتا ---
if df is not None:
    df = df.loc[:, df.isna().mean() < 0.8]
    st.session_state.df = df
    num_numeric = len(df.select_dtypes(include="number").columns)
    num_categorical = len(df.select_dtypes(exclude="number").columns)

# --- Main Page Logic ---
if choise == "Data Analysis" and df is not None:
    if model_choice_auto == "ML":
        st.title("Data Analysis Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Numeric", num_numeric)
        col4.metric("Categorical", num_categorical)
        st.dataframe(df.head(100))
        # (بقية كود الـ Heatmap والـ Distributions الخاصة بك تضاف هنا)

    elif model_choice_auto == "NLP":
        st.subheader("NLP Text Analysis")
        # (منطق الـ WordCloud والـ n-grams الخاص بك يضاف هنا)

elif choise == "Auto AI system" and df is not None:
    model_choice = st.selectbox("Choose AI Model", [" ", "ML", "NLP", "auto choise"])
    
    current_mode = model_choice
    if model_choice == "auto choise":
        current_mode = model_choice_auto
    
    if current_mode == "ML":
        superviseML = st.selectbox("Is the data supervised?", [" ", "supervise", "unsupervise", "auto choise"])
        # (بقية منطق الـ Classification/Regression Scores المعقد الخاص بك تضاف هنا)
        st.info(f"System Mode: {current_mode}")

elif df is None and choise != " ":
    st.warning("Please upload a file to start.")
