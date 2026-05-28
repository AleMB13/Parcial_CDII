# =========================================================
# IMPORTS
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
from nltk.corpus import stopwords

import easyocr
from wordcloud import WordCloud
from textblob import TextBlob

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Receta Médica Perú",
    page_icon="🏥",
    layout="wide"
)

# =========================================================
# NLTK SAFE
# =========================================================
@st.cache_resource
def asegurar_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

asegurar_nltk()

STOPWORDS_ES = set(stopwords.words("spanish"))

# =========================================================
# OCR CACHE
# =========================================================
@st.cache_resource
def cargar_ocr():
    import easyocr
    return easyocr.Reader(['es', 'en'], gpu=False)

reader = cargar_ocr()

# =========================================================
# CALIDAD DE IMAGEN (RESTO ORIGINAL)
# =========================================================
def calidad_imagen(gray):

    brillo = np.mean(gray)
    contraste = np.std(gray)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    estado_brillo = (
        "Oscura" if brillo < 80
        else "Muy clara" if brillo > 180
        else "Normal"
    )

    estado_blur = "Borroso" if blur < 100 else "Nítido"

    return {
        "brillo": round(brillo,2),
        "contraste": round(contraste,2),
        "blur": round(blur,2),
        "estado_brillo": estado_brillo,
        "estado_blur": estado_blur
    }

# =========================================================
# PREPROCESAMIENTO
# =========================================================
def preprocesar(img):

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return {
        "gray": gray,
        "metricas": calidad_imagen(gray)
    }

# =========================================================
# NLP SIMPLE
# =========================================================
def limpiar(texto):

    if texto is None:
        return "", 0

    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    tokens = texto.split()

    tokens_limpios = [
        t for t in tokens
        if t not in STOPWORDS_ES and len(t) > 2
    ]

    return " ".join(tokens_limpios), len(tokens_limpios)

# =========================================================
# EXTRACCIONES (RESTO ORIGINAL)
# =========================================================
def extraer_dni(t):
    m = re.search(r'\b\d{8}\b', t)
    return m.group() if m else None

def extraer_edad(t):
    m = re.search(r'(?:edad)\s*[:\-]?\s*(\d{1,3})', t, re.I)
    return m.group(1) if m else None

def extraer_cmp(t):
    m = re.search(r'(?:cmp)\s*[:\-]?\s*(\d+)', t, re.I)
    return m.group(1) if m else None

def extraer_nombre(t):
    m = re.search(r'(?:paciente|nombre)\s*[:\-]?\s*([a-zA-ZÁÉÍÓÚÑ\s]{5,})', t, re.I)
    return m.group(1).strip() if m else None

def extraer_sexo(t):
    if re.search(r'masculino| m\b', t, re.I):
        return "Masculino"
    if re.search(r'femenino| f\b', t, re.I):
        return "Femenino"
    return None

def extraer_fechas(t):
    fechas = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', t)
    return " | ".join(fechas) if fechas else None

def extraer_servicio(t):
    servicios = ["TRAUMATOLOGIA","NEUROLOGIA","CARDIOLOGIA","PEDIATRIA"]
    for s in servicios:
        if s in t.upper():
            return s.title()
    return "Medicina General"

def clasificar(t):
    t = t.upper()
    if "FRACTURA" in t or "TRAUMA" in t:
        return "Traumatología"
    if "NEURO" in t:
        return "Neurología"
    return "Medicina General"

def sentimiento(t):
    p = TextBlob(t).sentiment.polarity
    if p > 0:
        return "Positivo"
    if p < 0:
        return "Negativo"
    return "Neutral"

# =========================================================
# UI
# =========================================================
files = st.file_uploader(
    "Subir Certificados Médicos",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

# =========================================================
# PREVIEW
# =========================================================
if files:
    st.markdown("## Vista previa")
    cols = st.columns(3)

    for i,f in enumerate(files):
        with cols[i%3]:
            st.image(Image.open(f))

# =========================================================
# PROCESAMIENTO
# =========================================================
if files:

    if st.button("EJECUTAR OCR + NLP"):

        resultados = []
        progress = st.progress(0)

        for i,f in enumerate(files):

            img = Image.open(f)
            pre = preprocesar(img)

            texto = "\n".join(
                reader.readtext(pre["gray"], detail=0, paragraph=True)
            )

            limpio,_ = limpiar(texto)

            resultados.append({

                "archivo": f.name,

                # EXTRACCIONES
                "nombre": extraer_nombre(texto),
                "sexo": extraer_sexo(texto),
                "dni": extraer_dni(texto),
                "edad": extraer_edad(texto),
                "cmp": extraer_cmp(texto),
                "fechas": extraer_fechas(texto),
                "servicio": extraer_servicio(texto),

                # NLP
                "categoria": clasificar(texto),
                "sentimiento": sentimiento(texto),
                "texto": texto,
                "texto_limpio": limpio,

                # CALIDAD IMAGEN
                "brillo": pre["metricas"]["brillo"],
                "contraste": pre["metricas"]["contraste"],
                "blur": pre["metricas"]["blur"],
                "estado_brillo": pre["metricas"]["estado_brillo"],
                "estado_blur": pre["metricas"]["estado_blur"]
            })

            progress.progress((i+1)/len(files))
            gc.collect()

        df = pd.DataFrame(resultados)

        st.success("OCR + NLP COMPLETADO")

        # =====================================================
        # 📌 ESTA ES TU ESTRUCTURA ORIGINAL RESTAURADA
        # =====================================================
        st.markdown("## Información Extraída")

        for _, row in df.iterrows():

            st.markdown(f"""
            <div style="
                background:white;
                padding:20px;
                border-radius:15px;
                margin-bottom:20px;
                box-shadow:0 4px 10px rgba(0,0,0,0.1);
            ">

            <h3>{row['archivo']}</h3>

            <b>Nombre:</b> {row['nombre']}<br>
            <b>Sexo:</b> {row['sexo']}<br>
            <b>DNI:</b> {row['dni']}<br>
            <b>Edad:</b> {row['edad']}<br>
            <b>CMP:</b> {row['cmp']}<br>
            <b>Fechas:</b> {row['fechas']}<br>
            <b>Servicio:</b> {row['servicio']}<br>
            <b>Categoría:</b> {row['categoria']}<br>
            <b>Sentimiento:</b> {row['sentimiento']}<br>

            <hr>

            <b>Brillo:</b> {row['brillo']} ({row['estado_brillo']})<br>
            <b>Contraste:</b> {row['contraste']}<br>
            <b>Blur:</b> {row['blur']} ({row['estado_blur']})<br>

            </div>
            """, unsafe_allow_html=True)

        st.dataframe(df)
