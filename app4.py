# =========================================================
# IMPORTS LIMPIOS
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import re
import gc
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import easyocr
from io import BytesIO
from wordcloud import WordCloud
from textblob import TextBlob
from rapidfuzz import fuzz

# =========================================================
# CONFIG STREAMLIT
# =========================================================
st.set_page_config(
    page_title="Receta Médica Perú",
    page_icon="🏥",
    layout="wide"
)

# =========================================================
# NLTK (CORRECTO Y CACHEADO)
# =========================================================
@st.cache_resource
def asegurar_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

asegurar_nltk()

STOPWORDS_ES = set(stopwords.words('spanish'))

# =========================================================
# OCR (ESTO ES LO MÁS IMPORTANTE)
# =========================================================
@st.cache_resource(show_spinner=True)
def cargar_ocr():
    import easyocr
    return easyocr.Reader(['es', 'en'], gpu=False)

reader = cargar_ocr()

# =========================================================
# PREPROCESAMIENTO IMAGEN
# =========================================================
def preprocesar_imagen(pil_image):
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return {
        "original": img,
        "gray": gray
    }

# =========================================================
# NLP SIMPLE
# =========================================================
def limpiar_texto(texto):

    if texto is None:
        return ""

    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    tokens = word_tokenize(texto, language='spanish')

    tokens = [
        t for t in tokens
        if t not in STOPWORDS_ES and len(t) > 2
    ]

    return " ".join(tokens), len(tokens)

# =========================================================
# UI - UPLOADER
# =========================================================
uploaded_files = st.file_uploader(
    "Subir Certificados Médicos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =========================================================
# PREVIEW
# =========================================================
if uploaded_files:

    st.markdown("## Vista previa")
    cols = st.columns(3)

    for i, file in enumerate(uploaded_files[:9]):
        img = Image.open(file)

        with cols[i % 3]:
            st.image(img)

# =========================================================
# PROCESAMIENTO PRINCIPAL
# =========================================================
if uploaded_files:

    if st.button("EJECUTAR OCR + NLP"):

        resultados = []
        progress = st.progress(0)

        for i, file in enumerate(uploaded_files):

            image = Image.open(file)
            pre = preprocesar_imagen(image)

            # =========================
            # OCR (NO SE RECARGA MODELO)
            # =========================
            texto_ocr = "\n".join(
                reader.readtext(
                    pre["gray"],
                    detail=0,
                    paragraph=True
                )
            )

            texto_limpio, n_tokens = limpiar_texto(texto_ocr)

            resultados.append({
                "archivo": file.name,
                "texto_ocr": texto_ocr,
                "texto_limpio": texto_limpio,
                "tokens": n_tokens
            })

            progress.progress((i + 1) / len(uploaded_files))
            gc.collect()

        df_final = pd.DataFrame(resultados)

        st.success("PROCESO TERMINADO")

        st.dataframe(df_final)

        # =========================
        # FRECUENCIAS
        # =========================
        palabras = " ".join(df_final["texto_limpio"]).split()
        freq = Counter(palabras)

        top20 = freq.most_common(20)

        if top20:

            palabras_, valores = zip(*top20)

            fig, ax = plt.subplots(figsize=(10,5))

            ax.barh(
                list(reversed(palabras_)),
                list(reversed(valores))
            )

            st.pyplot(fig)

        # =========================
        # WORDCLOUD
        # =========================
        if palabras:

            wc = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate(" ".join(palabras))

            fig2, ax2 = plt.subplots()

            ax2.imshow(wc, interpolation="bilinear")
            ax2.axis("off")

            st.pyplot(fig2)
