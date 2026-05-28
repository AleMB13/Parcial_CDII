# =========================================================
# IMPORTS LIMPIOS
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
import seaborn as sns
from rapidfuzz import fuzz
from datetime import datetime

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Receta Médica Perú",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.style.use('ggplot')

# =========================================================
# NLTK (CACHE CORRECTO)
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

STOPWORDS_DOMINIO = {
    'sr', 'sra', 'sres', 'srta',
    'dr', 'dra',
    'certificado', 'certifica',
    'medico', 'médico',
    'fecha', 'dias', 'día',
    'lima', 'piura',
    'trujillo', 'arequipa',
    'escaneado', 'camscanner',
    'paciente', 'doctor',
    'doctora'
}

STOPWORDS_COMPLETO = STOPWORDS_ES | STOPWORDS_DOMINIO

# =========================================================
# OCR (CACHE REAL - FIX PRINCIPAL)
# =========================================================
@st.cache_resource
def cargar_ocr():
    import easyocr
    return easyocr.Reader(['es', 'en'], gpu=False)

reader = cargar_ocr()

# =========================================================
# PREPROCESAMIENTO IMAGEN
# =========================================================
def corregir_rotacion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 100:
        return img

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 1:
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

def analizar_calidad_imagen(gray):
    brillo = np.mean(gray)
    contraste = np.std(gray)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    blancos = np.sum(gray > 240)
    porcentaje_blanco = blancos / gray.size * 100

    estado_brillo = (
        "Oscura" if brillo < 80
        else "Muy clara" if brillo > 180
        else "Normal"
    )

    estado_blur = "Borroso" if blur < 100 else "Nítido"

    return {
        "brillo": round(brillo, 2),
        "contraste": round(contraste, 2),
        "blur": round(blur, 2),
        "porcentaje_blanco": round(porcentaje_blanco, 2),
        "estado_brillo": estado_brillo,
        "estado_blur": estado_blur
    }

def preprocesar_imagen(pil_image):
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = corregir_rotacion(img)

    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return {
        'original': img,
        'gray': gray,
        'metricas': analizar_calidad_imagen(gray)
    }

# =========================================================
# NLP
# =========================================================
def pipeline_limpieza(texto):
    if pd.isna(texto):
        texto = ""

    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    tokens = word_tokenize(texto, language='spanish')

    tokens_limpios = [
        t for t in tokens
        if t not in STOPWORDS_COMPLETO and len(t) > 2
    ]

    return {
        "texto_preprocesado": " ".join(tokens_limpios),
        "n_tokens_original": len(tokens),
        "n_tokens_limpios": len(tokens_limpios)
    }

# =========================================================
# UI
# =========================================================
uploaded_files = st.file_uploader(
    "Subir Certificados Médicos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:

    st.markdown("## Vista Previa")
    cols = st.columns(3)

    for i, file in enumerate(uploaded_files[:9]):
        img = Image.open(file)
        with cols[i % 3]:
            st.image(img)   # FIX AQUÍ

# =========================================================
# PROCESAMIENTO
# =========================================================
if uploaded_files:

    if st.button("EJECUTAR OCR + NLP"):

        resultados = []
        progress = st.progress(0)

        for i, file in enumerate(uploaded_files):

            image = Image.open(file)
            pre = preprocesar_imagen(image)

            texto_ocr = "\n".join(
                reader.readtext(pre['gray'], detail=0, paragraph=True)
            )

            nlp = pipeline_limpieza(texto_ocr)

            resultados.append({
                "archivo": file.name,
                "texto_ocr": texto_ocr,
                "texto_preprocesado": nlp["texto_preprocesado"],
                "tokens_originales": nlp["n_tokens_original"],
                "tokens_limpios": nlp["n_tokens_limpios"],
                "brillo": pre["metricas"]["brillo"],
                "blur": pre["metricas"]["blur"]
            })

            progress.progress((i + 1) / len(uploaded_files))
            gc.collect()

        df_final = pd.DataFrame(resultados)

        st.success("OCR + NLP FINALIZADO")

        st.dataframe(df_final)
