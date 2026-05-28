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
    page_title="Certificado Médico Perú",
    page_icon="🏥",
    layout="wide"
)

plt.style.use('ggplot')

# =========================================================
# SESSION STATE
# =========================================================
if "df_final" not in st.session_state:
    st.session_state.df_final = None
if "frecuencias" not in st.session_state:
    st.session_state.frecuencias = None
if "palabras" not in st.session_state:
    st.session_state.palabras = None

# =========================================================
# NLTK
# =========================================================
@st.cache_resource
def descargar_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

descargar_nltk()

STOPWORDS_ES = set(stopwords.words('spanish'))

def tokenizar(texto):
    try:
        return word_tokenize(texto, language='spanish')
    except:
        return texto.split()

# =========================================================
# OCR
# =========================================================
@st.cache_resource
def cargar_ocr():
    return easyocr.Reader(['es'], gpu=False)

reader = cargar_ocr()

# =========================================================
# NLP CLEAN
# =========================================================
def pipeline_limpieza(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    tokens = tokenizar(texto)
    tokens_limpios = [
        t for t in tokens
        if t not in STOPWORDS_ES and len(t) > 2
    ]

    return {
        "texto_preprocesado": " ".join(tokens_limpios),
        "tokens": tokens_limpios,
        "n_tokens_original": len(tokens),
        "n_tokens_limpios": len(tokens_limpios)
    }

# =========================================================
# EXTRACCIONES BÁSICAS
# =========================================================
def extraer_dni(texto):
    m = re.search(r'\b\d{8}\b', texto)
    return m.group() if m else None

def extraer_edad(texto):
    m = re.search(r'(\d{1,3})\s*años', texto.lower())
    return m.group(1) if m else None

def extraer_cmp(texto):
    m = re.search(r'cmp\s*[:\-]?\s*(\d+)', texto.lower())
    return m.group(1) if m else None

def extraer_fechas(texto):
    fechas = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', texto)
    return " | ".join(fechas) if fechas else None

# =========================================================
# 🔥 CORREGIDO: NOMBRE PACIENTE (SIN CLÍNICAS)
# =========================================================
CLINICAS = [
    "CLINICA", "HOSPITAL", "SAN PABLO", "RICARDO PALMA",
    "REBAGLIATI", "ESSALUD", "CENTRO", "MEDICO"
]

def extraer_nombre(texto):

    texto_up = re.sub(r'\s+', ' ', texto.upper())

    patrones = [
        r'NOMBRE\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ ]{8,})',
        r'PACIENTE\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ ]{8,})',
        r'SR(?:A)?\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ ]{8,})'
    ]

    # 1. búsqueda por patrones directos
    for p in patrones:
        m = re.search(p, texto_up)
        if m:
            nombre = m.group(1).strip()
            if not any(c in nombre for c in CLINICAS):
                return nombre

    # 2. fallback inteligente (evitando clínicas)
    palabras = texto_up.split()

    candidatos = []
    for w in palabras:
        if (
            w.isalpha()
            and len(w) > 3
            and w not in CLINICAS
            and w not in STOPWORDS_ES
        ):
            candidatos.append(w)

    # unir solo bloques tipo nombre (2–4 palabras)
    if len(candidatos) >= 3:
        return " ".join(candidatos[:4])

    return "NO IDENTIFICADO"

# =========================================================
# IMAGEN PREPROCESS
# =========================================================
def preprocesar_imagen(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return {
        "gray": gray,
        "metricas": {
            "brillo": float(np.mean(gray)),
            "contraste": float(np.std(gray)),
            "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "estado_blur": "Nítido" if cv2.Laplacian(gray, cv2.CV_64F).var() > 100 else "Borroso"
        }
    }

# =========================================================
# UI
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
    cols = st.columns(3)
    for i, file in enumerate(uploaded_files[:9]):
        with cols[i % 3]:
            st.image(Image.open(file), use_container_width=True)

# =========================================================
# PROCESAMIENTO
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
                "nombre": extraer_nombre(texto),
                "sexo": "Masculino",
                "dni": extraer_dni(texto),
                "edad": extraer_edad(texto),
                "cmp": extraer_cmp(texto),
                "fechas": extraer_fechas(texto),
                "categoria": "Traumatología",
                "sentimiento": "Neutral",
                "texto_ocr": texto,
                "texto_preprocesado": nlp["texto_preprocesado"],
                "tokens_originales": nlp["n_tokens_original"],
                "tokens_limpios": nlp["n_tokens_limpios"],
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

    df = st.session_state.df_final

    st.divider()
    st.subheader("📊 Resultados")

    doc = st.selectbox("Seleccionar documento", df["archivo"].tolist())
    fila = df[df["archivo"] == doc].iloc[0]

    st.markdown("## Información Extraída")

    st.markdown(f"""
    <div style="
        background:#fff;
        padding:20px;
        border-radius:15px;
        box-shadow:0 3px 10px rgba(0,0,0,0.1);
        color:#111827;
    ">

    <h3 style="color:#2563eb;">{fila['archivo']}</h3>

    <b>Nombre (Paciente):</b> {fila['nombre']}<br>
    <b>DNI:</b> {fila['dni']}<br>
    <b>Edad:</b> {fila['edad']}<br>
    <b>Fechas:</b> {fila['fechas']}<br>

    </div>
    """, unsafe_allow_html=True)

    st.subheader("📈 Métricas")
    st.metric("Documentos", len(df))
