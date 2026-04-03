import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
# --- البيانات ---
import numpy as np
import pandas as pd

# --- Streamlit (لو تستخدم واجهة) ---
import streamlit as st

# --- موديلات ML ---
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix , classification_report
from sklearn.pipeline import Pipeline
# --- Hyperparameter Tuning ---
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# --- تقييم النماذج ---
from sklearn.metrics import accuracy_score, f1_score, precision_score

# --- Boosting متقدم (اختياري) ---
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
#pip install -r requirements.txt
#.\venv\Scripts\Activate
#streamlit run steamlit.py


#nlp lib
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

import streamlit as st
from streamlit_lottie import st_lottie
import requests

# دالة محسنة للتحميل مع معالجة الأخطاء
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5) # إضافة وقت انتظار لضمان عدم تعليق الكود
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# استخدم هذا الرابط (رابط مباشر ومستقر حالياً لذكاء اصطناعي)
lottie_url = "https://lottie.host/8863f699-4d2b-4573-8321-72f1076b1070/fXm2FqV246.json"
lottie_ai_animation = load_lottieurl(lottie_url)

# التحقق: لا تعرض الأنيميشن إلا إذا نجح التحميل
with st.sidebar:
    if lottie_ai_animation:
        st_lottie(lottie_ai_animation, height=200, key="ai_sidebar")
    else:
        st.write("🤖 **Auto AI System**") # نص بديل في حال فشل التحميل


st.markdown("""
    <style>
    /* تغيير الخلفية للون كحلي داكن فخم */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
        color: #ffffff;
    }
    
    /* تجميل البطاقات (Cards) لتظهر بلمسة زجاجية */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* تغيير لون النصوص داخل الـ Metrics */
    div[data-testid="stMetricValue"] {
        color: #00d4ff !important;
    }

    /* تحسين شكل السايدبار (Sidebar) */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)


df = None
file_name = ""
model_choice_auto = ""
learning_type = ""
type_of_task = ""
type_Oftask_ai = ""
suggested_model = ""
potential_targets = []
list_of_unique_ratio = []
num_classes = 0
num_numeric = 0
num_categorical = 0
target=None
classification_model_name=""
Done=0
name=None
unique_count =None
model_choice_auto = None
TEXT_LEN_THRESHOLD = 50
UNIQUE_RATIO_THRESHOLD = 0.8
# -----------
# ----- Sidebar ----------------
if "df" not in st.session_state:
    st.session_state.df = None

# --- دالة تحميل الملفات ---

def load_file(Data):
    """
    Load uploaded file (CSV, Excel, TXT, JSON) safely, with Arabic support.
    Handles different encodings and separators.
    """
    name = Data.name.lower()

    # --- تحقق من الملف فارغ ---
    Data.seek(0)
    content = Data.read()
    if not content.strip():
        raise ValueError("Uploaded file is empty!")
    Data.seek(0)

    # --- CSV بالعربي وفواصل مختلفة ---
    if name.endswith(".csv"):
        encodings = ['utf-8', 'utf-8-sig', 'cp1256']
        separators = [',', ';', '\t']
        for enc in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(Data, encoding=enc, sep=sep)
                    if df.shape[1] > 0:  # تأكد من وجود أعمدة
                        return df
                except Exception:
                    pass
                Data.seek(0)  # إعادة المؤشر للبداية
        raise ValueError("Failed to read CSV. Could not detect delimiter or encoding.")

    # --- Excel .xlsx ---
    elif name.endswith(".xlsx"):
        try:
            return pd.read_excel(Data, engine='openpyxl')
        except Exception as e:
            raise ValueError(f"Error reading .xlsx file: {e}")

    # --- Excel .xls ---
    elif name.endswith(".xls"):
        try:
            return pd.read_excel(Data, engine='xlrd')
        except Exception:
            # fallback: ربما الملف ليس Excel حقيقي بل CSV مخفي بالامتداد
            return load_file_as_csv(Data)

    # --- JSON ---
    elif name.endswith(".json"):
        try:
            return pd.read_json(Data, lines=True)
        except ValueError:
            return pd.read_json(Data)

    # --- TXT ---
    elif name.endswith(".txt"):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("utf-8-sig")
        df = pd.DataFrame({"text": text.splitlines()})
        return df

    else:
        raise ValueError("Unsupported file format. Please upload CSV, Excel, JSON or TXT.")


def load_file_as_csv(Data):
    """
    Helper function to read a mis-labeled .xls as CSV.
    """
    Data.seek(0)
    encodings = ['utf-8', 'utf-8-sig', 'cp1256']
    separators = [',', ';', '\t']
    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(Data, encoding=enc, sep=sep)
                if df.shape[1] > 0:
                    return df
            except Exception:
                pass
            Data.seek(0)
    raise ValueError("Failed to read mislabeled .xls as CSV.")
# --- Sidebar ---
with st.sidebar:
    st.title("Auto AI System")
    st.image("loggo.png")
    Data = st.file_uploader("Upload CSV, Excel, JSON or TXT", type=["csv","xls","xlsx","json","txt"])
    if Data:
        st.session_state.df = load_file(Data)
        st.success("Data loaded ")
        df = st.session_state.get("df", None)

    if Data and df is not None:
        name = Data.name
        
        # استخراج الأعمدة النصية
        obj_cols = df.select_dtypes(include=['object']).columns
        
        # عدادات التصويت
        text_votes = 0
        cat_votes = 0
        
        if len(obj_cols) > 0:
            for col in obj_cols:
                # تنظيف البيانات وأخذ عينة
                sample = df[col].dropna().astype(str).head(100)
                if sample.empty: continue
                
                # حساب متوسط الكلمات ونسبة التفرد
                avg_words = sample.str.split().str.len().mean()
                unique_ratio = sample.nunique() / len(sample)
                has_spaces = sample.str.contains(' ').mean()

                # منطق التصنيف لكل عمود
                if avg_words > 10 and (unique_ratio > 0.5 or has_spaces > 0.8):
                    text_votes += 1
                else:
                    cat_votes += 1
            
            # اتخاذ القرار النهائي
            if name.endswith(".txt"):
                model_choice_auto = "NLP"
            elif text_votes >= 1 and len(df.columns) <= 3:
                model_choice_auto = "NLP"
            elif text_votes > cat_votes:
                model_choice_auto = "NLP"
            else:
                model_choice_auto = "ML"
        else:
            model_choice_auto = "ML"

        st.success(f"🤖 AI Analysis: This dataset is best handled as **{model_choice_auto}**")


    choise = st.selectbox("Service", ["  ", "Data Analysis", "Auto AI system", "Evaluation Visualization"])
    st.markdown("---")
    
    # --- تحميل الملف وتخزينه ---
    
     
# --- جلب الداتا أولاً ---
       

# --- تنظيف الأعمدة التي أغلبها NaN ---
    if df is not None:
        threshold = 0.8
        df = df.loc[:, df.isna().mean() < threshold]

    # حفظ النسخة المنظفة
        st.session_state.df = df

# --- عرض معلومات الداتا ---
    if df is not None:
        num_numeric = len(df.select_dtypes(include="number").columns)
        num_categorical = len(df.select_dtypes(exclude="number").columns)
          





    
# --- استخدام الداتا في Main Page ---
df = st.session_state.df
if choise == "Data Analysis":
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head(1000))
        st.write("Shape:", df.shape)
    else:
        st.warning("Please upload data first")
    if df is not None:
        num_numeric = df.select_dtypes(include='number').shape[1]
        num_categorical = df.select_dtypes(include='object').shape[1]

# ---------------- Initialize Session State ----------------
if 'suggested_models_list' not in st.session_state:
    st.session_state.suggested_models_list = []











# ---------------- Data Analysis ----------------

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
# تحميل stopwords مرة واحدة (أسرع)
EN_STOPWORDS = set(stopwords.words('english'))
AR_STOPWORDS = set(stopwords.words('arabic'))
ALL_STOPWORDS = EN_STOPWORDS.union(AR_STOPWORDS)





def normalize_arabic(text):
    text = re.sub(r'[\u064B-\u065F]', '', text)      # إزالة التشكيل
    text = re.sub(r'[أإآ]', 'ا', text)               # توحيد الألف
    text = re.sub(r'ى', 'ي', text)                   # توحيد الياء
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ـ+', '', text)                   # إزالة التطويل
    return text
def remove_emojis(text):
    return ''.join(c for c in text if not unicodedata.category(c).startswith('So'))


def universal_strong_cleaner(text):

    if pd.isna(text) or not isinstance(text, str):
        return ""

    original_text = text
    text = text.strip().lower()

    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    text = remove_emojis(text)
    text = normalize_arabic(text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    punctuation = string.punctuation + "«»،؛؟ـ…"
    text = text.translate(str.maketrans('', '', punctuation))
    text = re.sub(r'[^\u0600-\u06FFa-zA-Z\s]', '', text)

    try:
        if re.search(r'[\u0600-\u06FF]', text):
           tokens = text.split()          # عربي
        else:
            tokens = word_tokenize(text)   # إنجليزي
    except:
        return ""

    lang = "ar" if re.search(r'[\u0600-\u06FF]', text) else "en"
    stopwords = AR_STOPWORDS if lang == "ar" else EN_STOPWORDS

    tokens = [
        w for w in tokens
        if w not in stopwords and len(w) > 1
    ]

    # حماية ذكية
    if not tokens:
        return text

    return " ".join(tokens)

# 4️⃣ معالجة كل الأعمدة النصية
# -----------------------------------
def is_real_text_column(series, threshold=0.7):

    sample = series.dropna().astype(str).head(300)

    if len(sample) == 0:
        return False

    # استبعاد الأعمدة التاريخية
    try:
        pd.to_datetime(sample, errors='raise')
        return False
    except:
        pass

    # نسبة وجود حروف
    letter_ratio = sample.apply(
        lambda x: bool(re.search(r'[a-zA-Z\u0600-\u06FF]', x))
    ).mean()

    # متوسط طول النص
    avg_length = sample.apply(len).mean()

    # نسبة القيم الرقمية فقط
    numeric_ratio = sample.apply(lambda x: x.isdigit()).mean()

    if numeric_ratio > 0.5:
        return False

    if avg_length < 20:
        return False

    return letter_ratio >= threshold

def process_all_text_columns(df):

    original_df = df.copy()
    processed_columns_df = pd.DataFrame()

    # detect object columns
    candidate_columns = original_df.select_dtypes(include=['object']).columns.tolist()
    print(f"Detected candidate text columns: {candidate_columns}")

    real_text_columns = []

    for col in candidate_columns:

        if col.lower() in ['label', 'class', 'target']:
            continue

        if is_real_text_column(original_df[col]):
            real_text_columns.append(col)

    if not real_text_columns:
        print("No real text columns detected")
        return original_df, processed_columns_df

    print(f"Real text columns: {real_text_columns}")

    # process and store separately
    for col in real_text_columns:
        print(f"Processing column: {col}")

        processed_series = original_df[col].apply(universal_strong_cleaner)

        # store each processed column separately
        processed_columns_df[f"{col}_processed"] = processed_series

    print("Processing completed successfully")

    return original_df, processed_columns_df
# -----------------------------------
# 5️⃣ تحميل CSV بأمان
# -----------------------------------

FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def load_data_safely(file_path):

    try:
        df = pd.read_csv(
            file_path,
            encoding='utf-8-sig',   # أفضل خيار أولي
            engine='c',
            on_bad_lines='skip'
        )

        print("✅ تم تحميل الملف بنجاح باستخدام الترميز: utf-8-sig")
        print(f"عدد الصفوف: {len(df)}")
        return df

    except Exception as e:
        print("❌ فشل التحميل:", e)
        return None


#----------------
def get_ngrams(text, n=1):
    try:
        tokens = str(text).split()
        return list(ngrams(tokens, n))
    except:
        return []

def count_ngrams(series, n=1):
    all_ngrams = [
        ' '.join(item)
        for sublist in series.fillna("").apply(lambda x: get_ngrams(x, n))
        for item in sublist
    ]
    return Counter(all_ngrams)

def process_text_for_display(text):
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except:
        return text
def generate_multiclass_wordcloud(ngrams_by_label, title):

    labels = list(ngrams_by_label.keys())

    # Strong modern palette
    custom_palette = [
        "#E63946",  # red
        "#1D3557",  # navy
        "#2A9D8F",  # teal
        "#F4A261",  # orange
        "#6A4C93",  # purple
        "#118AB2",  # blue
        "#EF476F",  # pink
        "#073B4C"   # dark cyan
    ]

    label_colors = {
        label: custom_palette[i % len(custom_palette)]
        for i, label in enumerate(labels)
    }

    combined_freq = {}
    word_label_map = {}

    for label, counter in ngrams_by_label.items():
        for word, freq in counter.most_common(50):

            fixed_word = process_text_for_display(word)
            combined_freq[fixed_word] = combined_freq.get(fixed_word, 0) + freq

            if word not in word_label_map or freq > word_label_map[word][1]:
                word_label_map[fixed_word] = (label, freq)
    wc = WordCloud(
        font_path=FONT_PATH,
        width=1600,
        height=800,
        background_color='black',
        max_words=150,
        collocations=False,
        regexp=r"\w+",
        prefer_horizontal=0.9
    ).generate_from_frequencies(combined_freq)

    def color_func(word, **kwargs):
        label = word_label_map.get(word, (None, 0))[0]
        return label_colors.get(label, "#777777")

    wc.recolor(color_func=color_func)

    # Plot
    plt.figure(figsize=(20, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(process_text_for_display(title), fontsize=26, pad=30)

    legend_elements = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=str(label),
            markerfacecolor=label_colors[label],
            markersize=14
        )
        for label in labels
    ]

    plt.legend(
        handles=legend_elements,
        title="Class",
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(4, len(labels)),
        fontsize=13,
        title_fontsize=15,
        frameon=False
    )

    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
def analyze_and_visualize(df_original, df_processed, text_column, label_column, n=2):

    if label_column not in df_original.columns:
        st.write("Label column not found in original dataframe")
        return

    if text_column not in df_processed.columns:
        st.write("Processed text column not found")
        return

    # Merge label correctly
    analysis_df = df_processed.copy()
    analysis_df[label_column] = df_original[label_column]

    unique_labels = analysis_df[label_column].unique()
    ngrams_by_label = {}

    for label in unique_labels:
        subset = analysis_df[analysis_df[label_column] == label]
        counts = count_ngrams(subset[text_column], n)
        ngrams_by_label[label] = counts

        #st.write(f"\nTop 10 {n}-grams for class: {label}")
        #st.write(counts.most_common(10))

    generate_multiclass_wordcloud(
        ngrams_by_label,
        f"Most frequent {n}-grams by class")

if choise == "Data Analysis" and df is not None and model_choice_auto == "NLP":
    original_df, processed_df = process_all_text_columns(df)

    st.write("Original DataFrame:")
    st.write(original_df.head())

    st.write("\nProcessed Columns DataFrame:")
    st.write(processed_df.head())

    txcol=st.selectbox("text", processed_df.columns)
    lbcol=st.selectbox("label", df.columns)

    analyze_and_visualize(
    original_df,
    processed_df,
    text_column=txcol,  # change if needed
    label_column=lbcol,
    n=3  # 1=unigram, 2=bigram, 3=trigram
)

     





if choise == "Data Analysis" and df is not None and model_choice_auto == "ML":
    st.title("Data Analysis Dashboard")

    # --- Top Metrics Cards ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Numeric Columns", num_numeric)
    col4.metric("Categorical Columns", num_categorical)

    # --- Tabs ---
    tab_summary, tab_outliers, tab_categorical, tab_numeric, tab_corr, tab_target = st.tabs([
        "Summary Table", "Outliers", "Categorical Imbalance", "Numeric Distributions", "Correlation Heatmap", "Suggested Target"
    ])

    # ---------------- Summary Table ----------------
    with tab_summary:
        missing_threshold = 0.05
        outlier_threshold = 0.05
        summary_data = []
        num_cols = df.select_dtypes(include="number").columns

        # Calculate outliers ratio
        outlier_report = {}
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(df[col].dropna())>0:
                outlier_report[col] = len(outliers)/len(df[col].dropna())
            else:
                pass

        for col in df.columns:
            col_type = df[col].dtype
            missing_count = df[col].isna().sum()
            missing_ratio = df[col].isna().mean()
            unique_count = df[col].nunique()
            top_value = df[col].mode()[0] if unique_count > 0 else None
            top_freq = df[col].value_counts().iloc[0] if unique_count > 0 else None
            mean_val = df[col].mean() if col_type in ["int64", "float64"] else None
            std_val = df[col].std() if col_type in ["int64", "float64"] else None
            min_val = df[col].min() if col_type in ["int64", "float64"] else None
            max_val = df[col].max() if col_type in ["int64", "float64"] else None
            skewness = df[col].dropna().skew() if col_type in ["int64", "float64"] else None
            kurtosis = df[col].dropna().kurt() if col_type in ["int64", "float64"] else None
            has_outliers = outlier_report.get(col, 0) > outlier_threshold
            has_missing = missing_ratio > missing_threshold
            summary_data.append({
                "Column": col,
                "Type": col_type,
                "Missing Count": missing_count,
                "Missing %": round(missing_ratio*100,2),
                "Unique Values": unique_count,
                "Top Value": top_value,
                "Top Freq": top_freq,
                "Mean": mean_val,
                "Std": std_val,
                "Min": min_val,
                "Max": max_val,
                "Skewness": round(skewness,2) if skewness else None,
                "Kurtosis": round(kurtosis,2) if kurtosis else None,
                "Has Outliers": has_outliers,
                "Has Significant Missing": has_missing
            })
        summary_df = pd.DataFrame(summary_data)

        def highlight_issues(val, col_name):
            if col_name == "Has Significant Missing" and val:
                return "background-color: red; color: white"
            if col_name == "Has Outliers" and val == True:
                return "background-color: red; color: white"
            return ""

        styled_df = summary_df.style.map(lambda val: highlight_issues(val, "Has Outliers"), subset=["Has Outliers"]) \
                                    .map(lambda val: highlight_issues(val, "Has Significant Missing"), subset=["Has Significant Missing"])
        st.dataframe(styled_df, use_container_width=True)

    # ---------------- Outliers ----------------
    with tab_outliers:
        st.subheader("Outlier Counts")
        outlier_counts = {}
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            outlier_counts[col] = len(outliers)
        st.dataframe(pd.DataFrame.from_dict(outlier_counts, orient="index", columns=["Outlier Count"]))

    # ---------------- Categorical Imbalance ----------------
    with tab_categorical:
        cat_cols = df.select_dtypes(include="object").columns
        if len(cat_cols) == 0:
            st.info("No categorical columns detected")
        else:
            for col in cat_cols:
                with st.expander(f"Column: {col}"):
                    counts = df[col].value_counts(normalize=True)
                    max_ratio = counts.max()
                    st.dataframe(counts)
                    fig = px.bar(counts, title=f"Value Counts for {col}", labels={"index":"Value","y":"Proportion"})
                    st.plotly_chart(fig, use_container_width=True)
                    if max_ratio > 0.75:
                        st.warning("Imbalanced Column detected")
                    else:
                        st.success("Balanced Column")

    # ---------------- Numeric Distributions ----------------
    with tab_numeric:
        for col in num_cols:
            with st.expander(f"{col} Distribution"):
                fig = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}")
                st.plotly_chart(fig, use_container_width=True)
                fig_box = px.box(df, y=col, title=f"Boxplot of {col}")
                st.plotly_chart(fig_box, use_container_width=True)
                st.write(f"Skewness: {df[col].dropna().skew():.2f}, Kurtosis: {df[col].dropna().kurt():.2f}")

    # ---------------- Correlation Heatmap ----------------
    with tab_corr:
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            plt.close(fig)
            high_corr_pairs = [(i,j) for i in corr.columns for j in corr.columns if i!=j and abs(corr.loc[i,j])>0.85]
            if high_corr_pairs:
                st.warning(f"High correlation detected in pairs: {high_corr_pairs}")

    # ---------------- Suggested Target ----------------
    with tab_target:
        potential_targets = []
        list_of_unique_ratio = []
        for col in df.columns:
            nunique_ratio = df[col].nunique()/df.shape[0]
            if nunique_ratio < 0.1 and df[col].dtype != "object" and  df[col] is not None:
                potential_targets.append(col)
                list_of_unique_ratio.append(nunique_ratio)
        if potential_targets:
            target_index = list_of_unique_ratio.index(min(list_of_unique_ratio))
            target = potential_targets[target_index]
            st.write(f"Suggested target column: {target}")
            missing_ratio = df[target].isna().mean()
            issupervise = "Supervised Learning" if missing_ratio < 0.25 else "UnSupervised Learning"
            st.write(f"Learning type: {issupervise}")
            if df[target].dtype == "object":
                fig = px.bar(df[target].value_counts(), title=f"Target Value Counts for {target}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.histogram(df, x=target, nbins=30, title=f"Histogram of Target: {target}")
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No suitable target column found automatically")
if st.button("Run Full Analysis"):
    # ... كود التحليل هنا ...
    st.balloons()
    st.success("Analysis Completed Successfully!")

# ---------------- Auto AI System (Original) ----------------
model_choise=None
Donn=0
if choise == "Auto AI system":
    model_choice = st.selectbox(
        "Choose your AI model", [" ", "ML", "NLP", "auto choise"], key="main_model_select"
    )
    if model_choice== "ML" or model_choice=="NLP":
         model_choice_auto= model_choice
    # --- Auto detect system ---
    if model_choice == "auto choise" and df is not None:
         st.success(model_choice_auto)
         model_choice=model_choice_auto


    # --- ML Scope ---
    if model_choice== "ML"  and df is not None: 
        superviseML = st.selectbox(
            "Is the data supervised?", [" ", "supervise", "unsupervise", "auto choise"]
        )
       
        if superviseML in ["supervise", "unsupervise"]:
            learning_type = superviseML
        elif superviseML == "auto choise":
            
            for col in df.columns:
                unique_count = df[col].nunique()
                unique_ratio = unique_count / len(df)
                missing_ratio = df[col].isna().mean()
                if unique_count < 30 and unique_ratio < 0.5 and missing_ratio < 0.3:
                    learning_type = "supervise"
                    
                else :
                     learning_type="unsupervise"
            
        if learning_type:
             st.success(learning_type)

        if learning_type == "supervise":
            target = st.selectbox("Choose your label column", df.columns, key="target_col_select")

        # --- Task Type ---
        if learning_type=="supervise":

            type_of_task = st.selectbox(
           "Choose your type of task",
           ["", "Classification", "Regression","auto choise"],
              key="task_type_select"
             )
        if type_of_task in ["Classification", "Regression"]:
             type_Oftask_ai=type_of_task
        if type_of_task == "auto choise" and target is not None:
            if target is not None:
                unique_count = df[target].nunique()
                if df[target].dtype in ["int64", "float64"]:
                    type_Oftask_ai = "Regression" if unique_count > 20 else "Classification"
                else:
                    type_Oftask_ai = "Classification"

        if type_Oftask_ai:
             st.success(type_Oftask_ai)



    classification_models = [
        "",
        "Logistic Regression",
        "SVM",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "LightGBM",
        "XGBoost",
        "all of them",
        "Auto Chiose"
    ]

   
##__________________ML_Classification_____________

if type_of_task == "Classification" or type_Oftask_ai == "Classification":

	classification_model_name = st.selectbox(
		"Choose your classification model",
		classification_models,
		key="class_model_box"
	)

	if classification_model_name in ["Logistic Regression","SVM","Decision Tree","Random Forest","Gradient Boosting","LightGBM","XGBoost"]:
		suggested_model = classification_model_name


	if classification_model_name == "Auto Chiose":

		if "suggested_models_list" not in st.session_state:
			st.session_state.suggested_models_list = []

		scores = {
			"Logistic Regression": 0,
			"SVM": 0,
			"Random Forest": 0,
			"XGBoost": 0,
			"LightGBM": 0,
			"Gradient Boosting": 0,
			"Naive Bayes": 0
		}

		n_samples, n_features = df.shape
		num_numeric = len(df.select_dtypes(include=[np.number]).columns)
		num_categorical = len(df.select_dtypes(exclude=[np.number]).columns)

		missing_ratio = df.isnull().sum().sum() / (n_samples * n_features)

		if target in df.columns:
			class_counts = df[target].value_counts()
			imbalance_ratio = class_counts.min() / class_counts.max() if class_counts.max() > 0 else 1
			n_classes = df[target].nunique()
		else:
			imbalance_ratio = 1
			n_classes = 2

		high_dimensional = n_features > n_samples

		if n_samples < 1000:
			scores["Logistic Regression"] += 2
			scores["SVM"] += 2

		if 1000 <= n_samples <= 50000:
			scores["Random Forest"] += 2
			scores["Gradient Boosting"] += 2

		if n_samples > 50000:
			scores["LightGBM"] += 3
			scores["XGBoost"] += 2

		if high_dimensional:
			scores["SVM"] += 4
			scores["Naive Bayes"] += 2
			scores["Logistic Regression"] += 1

		if imbalance_ratio < 0.2:
			scores["XGBoost"] += 5
			scores["LightGBM"] += 4
			scores["Random Forest"] += 3
			scores["Logistic Regression"] += 1

		if missing_ratio > 0.1:
			scores["Random Forest"] += 3
			scores["LightGBM"] += 3
			scores["XGBoost"] += 3

		if num_categorical > num_numeric:
			scores["Random Forest"] += 2
			scores["LightGBM"] += 2

		if num_numeric > num_categorical:
			scores["SVM"] += 2
			scores["Logistic Regression"] += 2

		if n_classes > 5:
			scores["LightGBM"] += 3
			scores["XGBoost"] += 3
			scores["Random Forest"] += 2

		if n_samples < 2000 and n_features > 100:
			scores["SVM"] += 3
			scores["Naive Bayes"] += 2

		suggested_model = max(scores, key=scores.get)

		st.session_state.suggested_models_list.append(suggested_model)
		Done = 1
		st.success(f"AI Recommendation: **{suggested_model}**")
##__________________ML_Regression_____________





regression_models = [
          "",
          "Linear Regression",        
          "Ridge Regression",         
          "Random Forest Regressor",  
          "Gradient Boosting Regressor", 
          "XGBoost Regressor",        
          "Huber Regressor",         
          "Auto Chiose" ,
             "all them"           
              ]



if type_of_task == "Regression" or type_Oftask_ai == "Regression":

          Regression_model_name = st.selectbox(
        "Choose your Regression model",
        regression_models,
        key="reg_model_box"
           )

          if Regression_model_name == "Auto Chiose":

        # تهيئة session_state
            if "suggested_models_list" not in st.session_state:
                 st.session_state.suggested_models_list = []

        # تهيئة scores
            scores = {
              "Linear Regression": 0,
             "Ridge Regression": 0,
             "Random Forest Regressor": 0,
             "Gradient Boosting Regressor": 0,
             "XGBoost Regressor": 0,
             "Huber Regressor": 0
             }

        # ======= تحليلات الداتا =======
            n_samples, n_features = df.shape
            num_numeric = len(df.select_dtypes(include=[np.number]).columns)
            num_categorical = len(df.select_dtypes(exclude=[np.number]).columns)

            missing_ratio = df.isnull().sum().sum() / (n_samples * n_features)
            variance = df.select_dtypes(include=[np.number]).var().mean()

     
        # Dataset Size Logic
        
            if n_samples < 1000:
                scores["Linear Regression"] += 2
                scores["Ridge Regression"] += 2
                scores["Huber Regressor"] += 1

            if 1000 <= n_samples <= 50000:
              scores["Random Forest Regressor"] += 2
              scores["Gradient Boosting Regressor"] += 2

            if n_samples > 50000:
                scores["XGBoost Regressor"] += 3
                scores["Gradient Boosting Regressor"] += 2

       
        # High Dimensional Logic
       
            if n_features > n_samples:
                scores["Ridge Regression"] += 3
                scores["Linear Regression"] += 2

       
        # Missing Data Logic
        
            if missing_ratio > 0.1:
                scores["Random Forest Regressor"] += 2
                scores["Gradient Boosting Regressor"] += 1

    
        # Low Variance / Outliers
        
            if variance < 0.5:
                scores["Huber Regressor"] += 3
                scores["Ridge Regression"] += 1

     
        # Feature Type Logic
  
            if num_categorical > num_numeric:
                scores["Random Forest Regressor"] += 2
                scores["Gradient Boosting Regressor"] += 1

            if num_numeric > num_categorical:
                scores["Linear Regression"] += 1
                scores["Ridge Regression"] += 1

        # Final Decision
        
            suggested_model = max(scores, key=scores.get)
            st.session_state.suggested_models_list.append(suggested_model)

            st.success(f"AI Recommendation: **{suggested_model}**")
            #st.write("Regression Model Scores:", scores)
   

###"Logistic Regression",
        #"SVM",
        #"Decision Tree",
        #"Random Forest",
        #"Gradient Boosting",
        #"LightGBM",
        #"XGBoost",
        #"all of them",
        #"let AI choose"

for key in ["X_train", "X_test", "y_train", "y_test",
            "numeric_cols", "categorical_cols",
            "best_model", "best_acc"]:
    if key not in st.session_state:
        st.session_state[key] = None

bot_sep = None
split = "no"

# =================== تقسيم البيانات ===================
if st.session_state.df is not None and suggested_model:
    st.subheader("Data Splitting Options")
    test_size = st.slider("Test set size (%)", 10, 50, 20, 5)
    random_state = st.slider("Random state", 0, 100, 42, 1)
    stratify_option = st.checkbox("Use stratify (keep class distribution)")
    bot_sep = st.button("TRAIN")

    if bot_sep:
        X = st.session_state.df.drop(columns=[target])
        y = st.session_state.df[target]

        stratify = y if stratify_option and y.value_counts().min() > 1 else None
        if stratify_option and stratify is None:
            st.warning("Stratify disabled because some classes have <2 samples.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state, stratify=stratify
        )

        # تحويل الأعمدة إذا كان عمود واحد فقط
        if len(X_train.shape) == 1:
            X_train = X_train.to_frame()
            X_test = X_test.to_frame()

        numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
        categorical_cols = X_train.select_dtypes(exclude="number").columns.tolist()

        # حفظ في session_state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.numeric_cols = numeric_cols
        st.session_state.categorical_cols = categorical_cols

        st.success("Data split successfully!")
        st.write("X_train shape:", X_train.shape)
        st.write("X_test shape:", X_test.shape)
        split = "yes"

# =================== Training & Evaluation ===================
if split == "yes":

    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    numeric_cols = st.session_state.numeric_cols
    categorical_cols = st.session_state.categorical_cols

    # ===== قائمة الموديلات =====
    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "SVM": SVC(kernel='rbf', C=2, gamma='scale', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9,
                                 eval_metric='logloss', use_label_encoder=False),
        "LightGBM": lgb.LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05),
        "Naive Bayes": GaussianNB()
    }

    # فقط تدريب الموديل الذي اقترحه AI
    clf = models_dict.get(suggested_model, None)
    if clf is None:
        st.error("Model not supported")
    else:
        transformers = []

        # ===== تجهيز preprocessing =====
        if numeric_cols:
            if suggested_model in ["Logistic Regression", "SVM"]:
                transformers.append(('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numeric_cols))
            else:  # Tree-based
                transformers.append(('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median'))
                ]), numeric_cols))

        if categorical_cols:
            transformers.append(('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ]), categorical_cols))

        preprocessor = ColumnTransformer(transformers)

        model = Pipeline([
            ('prep', preprocessor),
            ('clf', clf)
        ])

        # ===== Encode target إذا كان نصي واحتاجه XGB/LightGBM =====
        y_train_fit = y_train
        y_test_fit = y_test
        label_encoder = None

        if y_train.dtype == 'object' and suggested_model in ["XGBoost", "LightGBM"]:
            label_encoder = LabelEncoder()
            y_train_fit = label_encoder.fit_transform(y_train)
            y_test_fit  = label_encoder.transform(y_test)

        # ===== تدريب الموديل =====
        model.fit(X_train, y_train_fit)
        preds = model.predict(X_test)

        # ===== إعادة تسمية predictions إذا تم الترميز =====
        if label_encoder is not None:
            preds = label_encoder.inverse_transform(preds)

        # ===== عرض النتائج =====
        st.write(f"## Model Used: {suggested_model}")
        st.write(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
        st.text(classification_report(y_test, preds))
        st.dataframe(pd.DataFrame(confusion_matrix(y_test, preds),
                                  columns=[f"Pred {c}" for c in sorted(set(y_test))],
                                  index=[f"Actual {c}" for c in sorted(set(y_test))]))

        st.session_state.best_model = suggested_model
        st.session_state.best_acc = accuracy_score(y_test, preds)
