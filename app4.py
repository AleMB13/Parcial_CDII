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
from io import BytesIO
from wordcloud import WordCloud
from textblob import TextBlob
import gc
from rapidfuzz import fuzz
from datetime import datetime
from itertools import chain

# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(
    page_title="Certificado Médico Perú",
    page_icon="🏥",
    layout="wide"
)

plt.style.use('ggplot')

# =========================================================
# SESSION STATE (CLAVE STREAMLIT)
# =========================================================
if "df_final" not in st.session_state:
    st.session_state.df_final = None
if "frecuencias" not in st.session_state:
    st.session_state.frecuencias = None
if "palabras" not in st.session_state:
    st.session_state.palabras = None

# =========================================================
# NLTK FIX DEFINITIVO
# =========================================================
@st.cache_resource
def descargar_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

descargar_nltk()

# fallback seguro (evita LookupError)
def tokenizar(texto):
    try:
        return word_tokenize(texto, language='spanish')
    except:
        return texto.split()

STOPWORDS_ES = set(stopwords.words('spanish'))

# =========================================================
# OCR
# =========================================================
@st.cache_resource
def cargar_ocr():
    return easyocr.Reader(['es'], gpu=False)

reader = cargar_ocr()

# =========================================================
# NLP LIMPIEZA
# =========================================================
def pipeline_limpieza(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    tokens = tokenizar(texto)
    tokens_limpios = [t for t in tokens if t not in STOPWORDS_ES and len(t) > 2]

    return {
        "texto_preprocesado": " ".join(tokens_limpios),
        "tokens": tokens_limpios,
        "n_tokens_original": len(tokens),
        "n_tokens_limpios": len(tokens_limpios)
    }

# =========================================================
# CLASIFICACIÓN
# =========================================================
def clasificar_documento(texto):
    texto = texto.upper()

    if "CARDIO" in texto:
        return "Cardiología"
    if "NEURO" in texto:
        return "Neurología"
    if "TRAUMA" in texto:
        return "Traumatología"
    if "PEDI" in texto:
        return "Pediatría"

    return "Medicina General"

# =========================================================
# EXTRACCIONES BÁSICAS
# =========================================================
def extraer_dni(texto):
    m = re.search(r'\b\d{8}\b', texto)
    return m.group() if m else None

def extraer_edad(texto):
    m = re.search(r'(\d{1,3})\s*años', texto)
    return m.group(1) if m else None

# =========================================================
# PREPROCESAMIENTO IMAGEN
# =========================================================
def preprocesar_imagen(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brillo = np.mean(gray)
    contraste = np.std(gray)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    return {
        "gray": gray,
        "metricas": {
            "brillo": round(brillo, 2),
            "contraste": round(contraste, 2),
            "blur": round(blur, 2),
            "estado_blur": "Nítido" if blur > 100 else "Borroso"
        }
    }

# =========================================================
# UI HEADER
# =========================================================
st.title("🏥 Certificado Médico Perú - OCR + NLP")

uploaded_files = st.file_uploader(
    "Subir certificados médicos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =========================================================
# PREVIEW
# =========================================================
if uploaded_files:
    st.subheader("Vista previa")

    cols = st.columns(3)
    for i, file in enumerate(uploaded_files[:9]):
        with cols[i % 3]:
            st.image(Image.open(file), use_container_width=True)

# =========================================================
# PROCESAMIENTO OCR + NLP
# =========================================================
if uploaded_files:

    if st.button("EJECUTAR OCR + NLP"):

        resultados = []
        progress = st.progress(0)

        for i, file in enumerate(uploaded_files):

            img = Image.open(file)
            pre = preprocesar_imagen(img)

            ocr = reader.readtext(pre["gray"], detail=0)
            texto = " ".join(ocr)

            nlp = pipeline_limpieza(texto)

            resultados.append({
                "archivo": file.name,
                "texto_ocr": texto,
                "texto_preprocesado": nlp["texto_preprocesado"],
                "tokens_originales": nlp["n_tokens_original"],
                "tokens_limpios": nlp["n_tokens_limpios"],
                "dni": extraer_dni(texto),
                "edad": extraer_edad(texto),
                "categoria": clasificar_documento(texto),
                "brillo": pre["metricas"]["brillo"],
                "contraste": pre["metricas"]["contraste"],
                "blur": pre["metricas"]["blur"],
                "estado_blur": pre["metricas"]["estado_blur"]
            })

            progress.progress((i + 1) / len(uploaded_files))
            gc.collect()

        df = pd.DataFrame(resultados)

        st.session_state.df_final = df
        st.session_state.palabras = " ".join(df["texto_preprocesado"]).split()
        st.session_state.frecuencias = Counter(st.session_state.palabras)

        st.success("Procesamiento completado")

# =========================================================
# DASHBOARD
# =========================================================
if st.session_state.df_final is not None:

    df_final = st.session_state.df_final
    frecuencias = st.session_state.frecuencias
    palabras = st.session_state.palabras

    st.divider()
    st.subheader("📊 Resultados")

    st.dataframe(df_final)

    # =====================================================
    # MÉTRICAS
    # =====================================================
    c1, c2, c3 = st.columns(3)

    c1.metric("Documentos", len(df_final))
    c2.metric("Tokens prom", int(df_final["tokens_originales"].mean()))
    c3.metric("Tokens limpios", int(df_final["tokens_limpios"].mean()))

    # =====================================================
    # TABS
    # =====================================================
    tab1, tab2, tab3 = st.tabs(["Frecuencia", "WordCloud", "Clasificación"])

    with tab1:
        top = frecuencias.most_common(20)
        if top:
            p, v = zip(*top)
            fig, ax = plt.subplots()
            ax.barh(p, v)
            st.pyplot(fig)

    with tab2:
        if palabras:
            wc = WordCloud(width=800, height=400).generate(" ".join(palabras))
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

    with tab3:
        st.bar_chart(df_final["categoria"].value_counts())

    # =====================================================
    # EXPORT
    # =====================================================
    csv = df_final.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Descargar CSV",
        csv,
        "certificados_medicos.csv",
        "text/csv"
    )
