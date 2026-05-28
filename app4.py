# =========================================================
# IMPORTS LIVIANOS (OPTIMIZADO)
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import re
import gc
from collections import Counter
from io import BytesIO
from PIL import Image
import cv2

# =========================================================
# CONFIGURACIÓN STREAMLIT
# =========================================================
st.set_page_config(
    page_title="Receta Médica Perú",
    page_icon="🏥",
    layout="wide"
)

# =========================================================
# NLTK (CACHE)
# =========================================================
@st.cache_resource
def init_nltk():
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    return nltk

nltk = init_nltk()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

STOPWORDS_ES = set(stopwords.words('spanish'))

# =========================================================
# SPACY (LIGERO + CACHE)
# =========================================================
@st.cache_resource
def cargar_spacy():
    import spacy

    nlp = spacy.blank("es")
    ruler = nlp.add_pipe("entity_ruler")

    patterns = [
        {"label": "CLINICA", "pattern": "Clínica San Pablo"},
        {"label": "CLINICA", "pattern": "Clínica Ricardo Palma"},
        {"label": "CLINICA", "pattern": "Hospital Rebagliati"},
        {"label": "SERVICIO", "pattern": "Traumatología"},
        {"label": "SERVICIO", "pattern": "Neurología"},
        {"label": "SERVICIO", "pattern": "Cardiología"},
    ]

    ruler.add_patterns(patterns)
    return nlp

nlp = cargar_spacy()

# =========================================================
# OCR (VERSIÓN PRO: TESSERACT)
# =========================================================
def ocr_tesseract(image):
    import pytesseract

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    texto = pytesseract.image_to_string(gray, lang="spa")
    return texto

# =========================================================
# PREPROCESAMIENTO IMAGEN
# =========================================================
def preprocesar_imagen(image):

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray

# =========================================================
# NLP SIMPLE Y ESTABLE
# =========================================================
def pipeline_limpieza(texto):

    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    tokens = texto.split()

    tokens_limpios = [
        t for t in tokens
        if len(t) > 2 and t not in STOPWORDS_ES
    ]

    return {
        "texto_preprocesado": " ".join(tokens_limpios),
        "tokens": tokens,
        "n_tokens": len(tokens),
        "n_tokens_limpios": len(tokens_limpios)
    }

# =========================================================
# CLASIFICACIÓN SIMPLE
# =========================================================
def clasificar(texto):

    texto = texto.upper()

    if "TRAUMA" in texto:
        return "Traumatología"
    if "NEURO" in texto:
        return "Neurología"
    if "CARDIO" in texto:
        return "Cardiología"
    if "NIÑO" in texto:
        return "Pediatría"

    return "Medicina General"

# =========================================================
# EXTRAER DNI SIMPLE
# =========================================================
def extraer_dni(texto):
    match = re.search(r'\b\d{8}\b', texto)
    return match.group() if match else None

# =========================================================
# UI
# =========================================================
st.title("🏥 Receta Médica Perú - OCR + NLP Optimizado")

uploaded_files = st.file_uploader(
    "Subir imágenes",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =========================================================
# PROCESAMIENTO
# =========================================================
if uploaded_files:

    if st.button("Ejecutar OCR + NLP"):

        resultados = []

        progress = st.progress(0)

        for i, file in enumerate(uploaded_files):

            image = Image.open(file)

            # OCR
            texto_ocr = ocr_tesseract(image)

            # NLP
            nlp_result = pipeline_limpieza(texto_ocr)

            # Clasificación
            categoria = clasificar(texto_ocr)

            resultados.append({
                "archivo": file.name,
                "texto_ocr": texto_ocr,
                "texto_limpio": nlp_result["texto_preprocesado"],
                "tokens": nlp_result["n_tokens"],
                "tokens_limpios": nlp_result["n_tokens_limpios"],
                "categoria": categoria,
                "dni": extraer_dni(texto_ocr)
            })

            progress.progress((i + 1) / len(uploaded_files))

            gc.collect()

        df = pd.DataFrame(resultados)

        st.success("Procesamiento completado")

        # =====================================================
        # RESULTADOS
        # =====================================================
        st.dataframe(df)

        # =====================================================
        # MÉTRICAS
        # =====================================================
        st.subheader("📊 Métricas")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Documentos", len(df))

        with c2:
            st.metric("Tokens promedio", int(df["tokens"].mean()))

        with c3:
            st.metric("Tokens limpios promedio", int(df["tokens_limpios"].mean()))

        # =====================================================
        # DESCARGA
        # =====================================================
        csv = df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "Descargar CSV",
            csv,
            "resultados_ocr.csv",
            "text/csv"
        )
