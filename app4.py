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
# CALIDAD DE IMAGEN (RESTAURADO)
# =========================================================
def analizar_calidad(gray):

    brillo = np.mean(gray)
    contraste = np.std(gray)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    blancos = np.sum(gray > 240)
    porcentaje_blanco = (blancos / gray.size) * 100

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

# =========================================================
# PREPROCESAMIENTO
# =========================================================
def preprocesar_imagen(pil_image):

    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    metricas = analizar_calidad(gray)

    return {
        "original": img,
        "gray": gray,
        "metricas": metricas
    }

# =========================================================
# NLP SIMPLE
# =========================================================
def limpiar_texto(texto):

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
# UPLOADER
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
            # OCR
            # =========================
            texto_ocr = "\n".join(
                reader.readtext(
                    pre["gray"],
                    detail=0,
                    paragraph=True
                )
            )

            # =========================
            # NLP
            # =========================
            texto_limpio, n_tokens = limpiar_texto(texto_ocr)

            resultados.append({
                "archivo": file.name,

                # TEXTO
                "texto_ocr": texto_ocr,
                "texto_limpio": texto_limpio,

                # CALIDAD IMAGEN (RESTAURADO)
                "brillo": pre["metricas"]["brillo"],
                "contraste": pre["metricas"]["contraste"],
                "blur": pre["metricas"]["blur"],
                "estado_brillo": pre["metricas"]["estado_brillo"],
                "estado_blur": pre["metricas"]["estado_blur"],

                # TOKENS
                "tokens": n_tokens
            })

            progress.progress((i + 1) / len(uploaded_files))
            gc.collect()

        df_final = pd.DataFrame(resultados)

        st.success("PROCESO COMPLETADO")

        # =====================================================
        # 📌 TARJETA VISUAL (ESTILO QUE TENÍAS)
        # =====================================================
        st.markdown("## Información Extraída")

        for _, row in df_final.iterrows():

            st.markdown(f"""
            <div style="
                background:white;
                padding:20px;
                border-radius:15px;
                margin-bottom:20px;
                box-shadow:0 4px 10px rgba(0,0,0,0.1);
            ">

            <h4>{row['archivo']}</h4>

            <b>Texto OCR:</b><br>
            {row['texto_ocr'][:300]}...<br><br>

            <b>Texto Limpio:</b><br>
            {row['texto_limpio'][:200]}...<br><br>

            <b>Brillo:</b> {row['brillo']} ({row['estado_brillo']})<br>
            <b>Contraste:</b> {row['contraste']}<br>
            <b>Blur:</b> {row['blur']} ({row['estado_blur']})<br>

            <b>Tokens:</b> {row['tokens']}

            </div>
            """, unsafe_allow_html=True)

        st.dataframe(df_final)

        # =====================================================
        # WORDCLOUD
        # =====================================================
        texto_total = " ".join(df_final["texto_limpio"])

        if texto_total.strip():

            wc = WordCloud(
                width=900,
                height=400,
                background_color="white"
            ).generate(texto_total)

            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        # =====================================================
        # FRECUENCIAS
        # =====================================================
        palabras = texto_total.split()
        freq = Counter(palabras)

        top20 = freq.most_common(20)

        if top20:

            p, v = zip(*top20)

            fig2, ax2 = plt.subplots(figsize=(10,5))

            ax2.barh(list(reversed(p)), list(reversed(v)))

            st.pyplot(fig2)
