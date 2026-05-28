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
import gc

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Certificados Médicos Perú",
    page_icon="🏥",
    layout="wide"
)

# =========================================================
# NLTK SAFE
# =========================================================
@st.cache_resource
def init_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

init_nltk()

STOPWORDS = set(stopwords.words("spanish"))

# =========================================================
# OCR
# =========================================================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['es'], gpu=False)

reader = load_ocr()

# =========================================================
# IMAGEN PREPROCESS
# =========================================================
def preprocesar(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

# =========================================================
# 🔥 OCR MEJORADO (IMPORTANTE)
# =========================================================
def ocr_mejor(gray):
    result = reader.readtext(gray, detail=1, paragraph=False)

    # ordenar por posición (muy importante)
    result = sorted(result, key=lambda x: x[0][0][1])

    lineas = [x[1] for x in result]

    return lineas

# =========================================================
# LIMPIEZA NLP
# =========================================================
def limpiar(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ0-9\s]', ' ', texto)
    tokens = texto.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

# =========================================================
# 🔥 NOMBRE CORREGIDO (CLAVE)
# =========================================================
def extraer_nombre(lineas):

    texto = " | ".join(lineas).upper()

    # patrón fuerte primero
    patrones = [
        r'PACIENTE\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ ]{6,})',
        r'NOMBRE\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ ]{6,})'
    ]

    for p in patrones:
        m = re.search(p, texto)
        if m:
            nombre = m.group(1)

            # cortar si aparece clínica
            nombre = nombre.split("CLINICA")[0]
            nombre = nombre.split("HOSPITAL")[0]

            return nombre.strip()

    return "NO IDENTIFICADO"

# =========================================================
# DNI / FECHAS / EDAD
# =========================================================
def extraer_dni(texto):
    m = re.search(r'\b\d{8}\b', texto)
    return m.group() if m else None

def extraer_edad(texto):
    m = re.search(r'(\d{1,3})\s*años', texto.lower())
    return m.group(1) if m else None

def extraer_fechas(texto):
    fechas = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', texto)
    return " | ".join(fechas) if fechas else None

# =========================================================
# UI
# =========================================================
st.title("🏥 Certificados Médicos - OCR Inteligente")

files = st.file_uploader(
    "Sube certificados",
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
# PROCESS
# =========================================================
if files and st.button("EJECUTAR OCR"):

    data = []
    progress = st.progress(0)

    for i, f in enumerate(files):

        img = Image.open(f)
        gray = preprocesar(img)

        lineas = ocr_mejor(gray)
        texto = " ".join(lineas)

        data.append({
            "archivo": f.name,
            "nombre": extraer_nombre(lineas),
            "dni": extraer_dni(texto),
            "edad": extraer_edad(texto),
            "fechas": extraer_fechas(texto),
            "texto": texto
        })

        progress.progress((i + 1) / len(files))
        gc.collect()

    df = pd.DataFrame(data)

    st.session_state.df = df
    st.success("Listo")

# =========================================================
# RESULTADOS
# =========================================================
if "df" in st.session_state:

    df = st.session_state.df

    st.subheader("📊 Resultado Completo")

    doc = st.selectbox("Documento", df["archivo"])
    row = df[df["archivo"] == doc].iloc[0]

    st.markdown(f"""
    ## {row['archivo']}

    **Nombre:** {row['nombre']}  
    **DNI:** {row['dni']}  
    **Edad:** {row['edad']}  
    **Fechas:** {row['fechas']}  

    ---
    ### Texto OCR
    {row['texto']}
    """)
