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
# PREPROCESAMIENTO IMAGEN (MEJORADO)
# =========================================================
def preprocesar_imagen(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # mejora contraste
    gray = cv2.equalizeHist(gray)

    # reducción ruido
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    return gray

# =========================================================
# OCR SAFE
# =========================================================
def hacer_ocr(gray):
    try:
        texto = reader.readtext(gray, detail=0)
        return " ".join(texto)
    except:
        return ""

# =========================================================
# EXTRAER DNI (MEJORADO)
# =========================================================
def extraer_dni(texto):
    matches = re.findall(r'\d{8}', texto)
    return matches[0] if matches else "NO ENCONTRADO"

# =========================================================
# FECHAS
# =========================================================
def extraer_fechas(texto):
    fechas = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', texto)
    return " | ".join(fechas) if fechas else "SIN FECHAS"

# =========================================================
# EDAD
# =========================================================
def extraer_edad(texto):
    m = re.findall(r'(\d{1,3})\s*años', texto.lower())
    return m[0] if m else "NO ENCONTRADO"

# =========================================================
# CMP
# =========================================================
def extraer_cmp(texto):
    m = re.findall(r'cmp\D*(\d+)', texto.lower())
    return m[0] if m else "NO ENCONTRADO"

# =========================================================
# NOMBRE (MEJORADO - EVITA CLÍNICAS)
# =========================================================
def extraer_nombre(texto):

    texto_u = texto.upper()

    # quitar palabras de clínica
    texto_u = re.sub(r'CLINICA|HOSPITAL|CENTRO|MEDICO|SALUD', ' ', texto_u)

    patrones = [
        r'NOMBRE\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ ]{8,})',
        r'PACIENTE\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ ]{8,})',
        r'SR(?:A)?\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ ]{8,})'
    ]

    for p in patrones:
        m = re.search(p, texto_u)
        if m:
            return m.group(1).strip()

    # fallback: busca nombre probable (evita números y palabras cortas)
    palabras = [w for w in texto_u.split() if w.isalpha() and len(w) > 4]

    if len(palabras) >= 4:
        return " ".join(palabras[:4])

    return "NO IDENTIFICADO"

# =========================================================
# UI
# =========================================================
st.title("🏥 Certificado Médico Perú - OCR Mejorado")

uploaded_files = st.file_uploader(
    "Subir imágenes",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

# =========================================================
# PREVIEW
# =========================================================
if uploaded_files:
    st.subheader("Vista previa")
    cols = st.columns(3)
    for i, f in enumerate(uploaded_files):
        with cols[i % 3]:
            st.image(Image.open(f), use_container_width=True)

# =========================================================
# PROCESO
# =========================================================
if uploaded_files:

    if st.button("EJECUTAR OCR"):

        resultados = []
        progress = st.progress(0)

        for i, file in enumerate(uploaded_files):

            img = Image.open(file)
            gray = preprocesar_imagen(img)

            texto = hacer_ocr(gray)

            if texto.strip() == "":
                texto = "OCR VACÍO"

            resultados.append({
                "archivo": file.name,
                "nombre": extraer_nombre(texto),
                "dni": extraer_dni(texto),
                "edad": extraer_edad(texto),
                "cmp": extraer_cmp(texto),
                "fechas": extraer_fechas(texto),
                "texto_ocr": texto
            })

            progress.progress((i+1)/len(uploaded_files))
            gc.collect()

        st.session_state.df_final = pd.DataFrame(resultados)
        st.success("Listo")

# =========================================================
# RESULTADOS
# =========================================================
if st.session_state.df_final is not None:

    df = st.session_state.df_final

    st.subheader("📊 RESULTADOS")

    doc = st.selectbox("Documento", df["archivo"])
    fila = df[df["archivo"] == doc].iloc[0]

    st.markdown("## Resultado Completo")

    st.write("Nombre:", fila["nombre"])
    st.write("DNI:", fila["dni"])
    st.write("Edad:", fila["edad"])
    st.write("CMP:", fila["cmp"])
    st.write("Fechas:", fila["fechas"])

    with st.expander("Ver OCR completo"):
        st.write(fila["texto_ocr"])

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV", csv, "resultado.csv")
