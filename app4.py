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
    page_title="Certificado Médico Perú - PRO",
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
# NLTK SAFE INIT
# =========================================================
@st.cache_resource
def init_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

init_nltk()

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
# NORMALIZADOR (CLAVE PRO)
# =========================================================
def normalizar_texto(texto):
    texto = texto.upper()
    texto = re.sub(r'\s+', ' ', texto)
    texto = texto.replace(":", " : ")
    return texto

# =========================================================
# NLP
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
# EXTRACCIÓN ROBUSTA
# =========================================================

def extraer_dni(texto):
    texto = normalizar_texto(texto)

    m = re.search(r'DNI\s*[:\-]?\s*(\d{8})', texto)
    if m:
        return m.group(1)

    m = re.search(r'\b\d{8}\b', texto)
    if m:
        return m.group(0)

    return None


def extraer_edad(texto):
    m = re.search(r'(\d{1,3})\s*AÑOS', texto.upper())
    return m.group(1) if m else None


def extraer_cmp(texto):
    m = re.search(r'CMP\s*[:\-]?\s*(\d+)', texto.upper())
    return m.group(1) if m else None


def extraer_fechas(texto):
    fechas = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', texto)
    fechas = list(dict.fromkeys(fechas))
    return " | ".join(fechas) if fechas else None


# =========================================================
# 🔥 NOMBRE (VERSIÓN PRO REAL)
# =========================================================
def extraer_nombre(texto):

    texto = normalizar_texto(texto)

    # CASO 1: PACIENTE (prioridad máxima)
    m = re.search(r'PACIENTE\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ ]{6,})', texto)
    if m:
        nombre = m.group(1)
        nombre = re.split(r'DNI|CMP|EDAD|CLINICA|HOSPITAL', nombre)[0]
        return nombre.strip()

    # CASO 2: NOMBRE
    m = re.search(r'NOMBRE\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ ]{6,})', texto)
    if m:
        nombre = m.group(1)
        nombre = re.split(r'DNI|CMP|CLINICA|HOSPITAL', nombre)[0]
        return nombre.strip()

    # CASO 3: fallback inteligente
    palabras = texto.split()
    candidatos = [w for w in palabras if w.isalpha() and len(w) > 4]

    if len(candidatos) >= 3:
        return " ".join(candidatos[:4])

    return "NO IDENTIFICADO"

# =========================================================
# PREPROCESAMIENTO IMAGEN
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
            "estado_blur": "NÍTIDO" if cv2.Laplacian(gray, cv2.CV_64F).var() > 100 else "BORROSO"
        }
    }

# =========================================================
# UI
# =========================================================
st.title("🏥 Certificado Médico Perú - OCR PRO")

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

    for i, f in enumerate(uploaded_files[:9]):
        with cols[i % 3]:
            st.image(Image.open(f), use_container_width=True)

# =========================================================
# PROCESSING
# =========================================================
if uploaded_files:

    if st.button("EJECUTAR OCR + NLP PRO"):

        resultados = []
        progress = st.progress(0)

        for i, file in enumerate(uploaded_files):

            img = Image.open(file)
            pre = preprocesar_imagen(img)

            # 🔥 OCR PRO (con estructura)
            ocr = reader.readtext(pre["gray"], detail=1, paragraph=True)
            texto = " ".join([r[1] for r in ocr])
            texto = normalizar_texto(texto)

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

        st.success("✔ Procesamiento PRO completado")

# =========================================================
# DASHBOARD
# =========================================================
if st.session_state.df_final is not None:

    df = st.session_state.df_final
    freq = st.session_state.frecuencias
    words = st.session_state.palabras

    st.subheader("📊 Resultados")

    doc = st.selectbox("Seleccionar documento", df["archivo"])
    row = df[df["archivo"] == doc].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.info(f"Sexo: {row['sexo']}")
    c2.info(f"Edad: {row['edad']}")
    c3.info(f"DNI: {row['dni']}")
    c4.info(f"Blur: {row['estado_blur']}")

    st.markdown("## Información Extraída")

    st.markdown(f"""
    <div style="
        background:white;
        padding:25px;
        border-radius:15px;
        color:#111;
        box-shadow:0 4px 12px rgba(0,0,0,0.1);
    ">

    <h3>{row['archivo']}</h3>

    <b>Nombre:</b> {row['nombre']}<br>
    <b>DNI:</b> {row['dni']}<br>
    <b>Edad:</b> {row['edad']}<br>
    <b>CMP:</b> {row['cmp']}<br>
    <b>Fechas:</b> {row['fechas']}<br>

    </div>
    """, unsafe_allow_html=True)

    st.subheader("📈 Métricas")
    c1, c2, c3 = st.columns(3)
    c1.metric("Docs", len(df))
    c2.metric("Tokens OCR", int(df["tokens_originales"].mean()))
    c3.metric("Tokens Limpios", int(df["tokens_limpios"].mean()))

    tab1, tab2, tab3 = st.tabs(["Frecuencia", "WordCloud", "Clasificación"])

    with tab1:
        top = freq.most_common(20)
        if top:
            p, v = zip(*top)
            fig, ax = plt.subplots()
            ax.barh(p, v)
            st.pyplot(fig)

    with tab2:
        wc = WordCloud(width=800, height=400).generate(" ".join(words))
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

    with tab3:
        st.bar_chart(df["categoria"].value_counts())

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Descargar CSV",
        csv,
        "certificados_pro.csv",
        "text/csv"
    )
