# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import easyocr
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud
import gc

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Certificados Médicos Perú",
    page_icon="🏥",
    layout="wide"
)

plt.style.use("ggplot")

# =========================================================
# SESSION STATE
# =========================================================
if "df_final" not in st.session_state:
    st.session_state.df_final = None
if "tokens" not in st.session_state:
    st.session_state.tokens = None
if "freq" not in st.session_state:
    st.session_state.freq = None

# =========================================================
# NLTK SAFE INIT
# =========================================================
@st.cache_resource
def init_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

init_nltk()

STOPWORDS = set(stopwords.words("spanish"))

def tokenizar(texto):
    try:
        return word_tokenize(texto, language="spanish")
    except:
        return texto.split()

# =========================================================
# OCR
# =========================================================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["es"], gpu=False)

reader = load_ocr()

# =========================================================
# IMAGEN PREPROCESS
# =========================================================
def preprocesar(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

# =========================================================
# OCR ORDENADO (IMPORTANTE)
# =========================================================
def ocr_lineas(gray):
    result = reader.readtext(gray, detail=1, paragraph=False)
    result = sorted(result, key=lambda x: x[0][0][1])
    return [r[1] for r in result]

# =========================================================
# NLP PIPELINE COMPLETO
# =========================================================
def pipeline_nlp(texto):

    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúñ0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    tokens = tokenizar(texto)
    tokens_limpios = [
        t for t in tokens
        if t not in STOPWORDS and len(t) > 2
    ]

    return {
        "texto": " ".join(tokens_limpios),
        "tokens": tokens_limpios,
        "n_tokens": len(tokens),
        "n_limpios": len(tokens_limpios)
    }

# =========================================================
# 🔥 EXTRAER NOMBRE (CORREGIDO CLÍNICA vs PACIENTE)
# =========================================================
def extraer_nombre(lineas):

    texto = " | ".join(lineas).upper()

    patrones = [
        r"PACIENTE\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ ]{6,})",
        r"NOMBRE\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ ]{6,})",
        r"SR(?:A)?\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ ]{6,})"
    ]

    for p in patrones:
        m = re.search(p, texto)
        if m:
            nombre = m.group(1)

            # cortar si aparece institución
            for k in ["CLINICA", "HOSPITAL", "ESSALUD"]:
                nombre = nombre.split(k)[0]

            return nombre.strip()

    return "NO IDENTIFICADO"

# =========================================================
# EXTRACCIONES
# =========================================================
def extraer_dni(texto):
    m = re.search(r"\b\d{8}\b", texto)
    return m.group() if m else None

def extraer_edad(texto):
    m = re.search(r"(\d{1,3})\s*años", texto)
    return m.group(1) if m else None

def extraer_fechas(texto):
    f = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", texto)
    return " | ".join(f) if f else None

# =========================================================
# CLASIFICACIÓN SIMPLE
# =========================================================
def clasificar(texto):
    t = texto.upper()

    if "TRAUMA" in t:
        return "Traumatología"
    if "CARDIO" in t:
        return "Cardiología"
    if "NEURO" in t:
        return "Neurología"
    if "PEDIATR" in t:
        return "Pediatría"
    return "Medicina General"

# =========================================================
# UI
# =========================================================
st.title("🏥 Certificados Médicos Perú - OCR + NLP")

files = st.file_uploader(
    "Sube certificados médicos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =========================================================
# PREVIEW
# =========================================================
if files:
    cols = st.columns(3)
    for i, f in enumerate(files[:6]):
        with cols[i % 3]:
            st.image(Image.open(f), use_container_width=True)

# =========================================================
# PROCESAMIENTO
# =========================================================
if files and st.button("EJECUTAR OCR + NLP"):

    data = []
    progress = st.progress(0)

    all_tokens = []

    for i, f in enumerate(files):

        img = Image.open(f)
        gray = preprocesar(img)

        lineas = ocr_lineas(gray)
        texto = " ".join(lineas)

        nlp = pipeline_nlp(texto)
        categoria = clasificar(texto)

        all_tokens.extend(nlp["tokens"])

        data.append({
            "archivo": f.name,
            "nombre": extraer_nombre(lineas),
            "dni": extraer_dni(texto),
            "edad": extraer_edad(texto),
            "fechas": extraer_fechas(texto),
            "categoria": categoria,
            "texto_ocr": texto,
            "texto_limpio": nlp["texto"],
            "tokens": nlp["n_tokens"],
            "tokens_limpios": nlp["n_limpios"]
        })

        progress.progress((i + 1) / len(files))
        gc.collect()

    df = pd.DataFrame(data)

    st.session_state.df_final = df
    st.session_state.tokens = all_tokens
    st.session_state.freq = Counter(all_tokens)

    st.success("Procesamiento completo")

# =========================================================
# DASHBOARD
# =========================================================
if st.session_state.df_final is not None:

    df = st.session_state.df_final

    st.divider()
    st.subheader("📊 Resultado Completo")

    doc = st.selectbox("Seleccionar documento", df["archivo"])
    row = df[df["archivo"] == doc].iloc[0]

    # =====================================================
    # FICHA ORDENADA (COMO PEDISTE)
    # =====================================================
    st.markdown(f"""
    ## Subir la foto:
    **{row['archivo']}**

    **Nombre:** {row['nombre']}  
    **DNI:** {row['dni']}  
    **Edad:** {row['edad']}  
    **Fechas:** {row['fechas']}  
    **Servicio:** Medicina General  
    **Categoría:** {row['categoria']}  
    """)

    st.markdown("## Información Extraída")

    st.markdown(f"""
    <div style="
        background:#fff;
        padding:20px;
        border-radius:15px;
        color:#111;
        box-shadow:0 3px 10px rgba(0,0,0,0.1);
    ">
    <b>Nombre:</b> {row['nombre']}<br>
    <b>DNI:</b> {row['dni']}<br>
    <b>Edad:</b> {row['edad']}<br>
    <b>Fechas:</b> {row['fechas']}<br>
    <b>Categoría:</b> {row['categoria']}<br>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # MÉTRICAS
    # =====================================================
    st.subheader("📈 Métricas")

    c1, c2, c3 = st.columns(3)
    c1.metric("Documentos", len(df))
    c2.metric("Tokens OCR", int(df["tokens"].mean()))
    c3.metric("Tokens Limpios", int(df["tokens_limpios"].mean()))

    # =====================================================
    # TABS NLP
    # =====================================================
    tab1, tab2, tab3 = st.tabs(["Frecuencia", "WordCloud", "Clasificación"])

    with tab1:
        freq = st.session_state.freq.most_common(20)
        if freq:
            p, v = zip(*freq)
            fig, ax = plt.subplots()
            ax.barh(p, v)
            st.pyplot(fig)

    with tab2:
        wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(st.session_state.tokens))
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

    with tab3:
        st.bar_chart(df["categoria"].value_counts())
