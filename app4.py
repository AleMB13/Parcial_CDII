```python
# =========================================================
# IMPORTS
# =========================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import re
from io import BytesIO
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import cv2
import easyocr

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from wordcloud import WordCloud
from textblob import TextBlob

import spacy
from rapidfuzz import fuzz

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
# CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f4f7fc;
}

.titulo {
    font-size: 42px;
    font-weight: 800;
    color: #0f172a;
    text-align: center;
    margin-top: 10px;
}

.subtitulo {
    font-size: 18px;
    color: #475569;
    text-align: center;
    margin-bottom: 30px;
}

section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

section[data-testid="stSidebar"] * {
    color: white;
}

.metric-card {
    padding: 22px;
    border-radius: 22px;
    color: white;
    text-align: center;
    font-weight: bold;
    box-shadow: 0px 8px 18px rgba(0,0,0,0.15);
    margin-bottom: 10px;
}

.metric-title {
    font-size: 17px;
}

.metric-value {
    font-size: 34px;
}

.card1 {
    background: linear-gradient(135deg,#2563eb,#60a5fa);
}

.card2 {
    background: linear-gradient(135deg,#9333ea,#c084fc);
}

.card3 {
    background: linear-gradient(135deg,#059669,#34d399);
}

.card4 {
    background: linear-gradient(135deg,#ea580c,#fb923c);
}

.card5 {
    background: linear-gradient(135deg,#dc2626,#f87171);
}

.stButton>button {
    background: linear-gradient(90deg,#2563eb,#60a5fa);
    color: white;
    border-radius: 14px;
    border: none;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
}

.stDownloadButton>button {
    background: linear-gradient(90deg,#059669,#34d399);
    color: white;
    border-radius: 14px;
    border: none;
    height: 50px;
    font-size: 16px;
    font-weight: bold;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="titulo">
Receta Médica Perú
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitulo">
Extracción inteligente de texto clínico usando OCR + NLP
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Configuración")

mostrar_preprocesamiento = st.sidebar.checkbox(
    "Mostrar preprocesamiento",
    value=True
)

# =========================================================
# NLTK
# =========================================================

@st.cache_resource
def descargar_nltk():

    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except:
        nltk.download('stopwords')

descargar_nltk()

# =========================================================
# SPACY
# =========================================================

@st.cache_resource
def cargar_spacy():

    nlp = spacy.blank("es")

    return nlp

nlp = cargar_spacy()

# =========================================================
# OCR
# =========================================================

@st.cache_resource
def cargar_ocr():

    return easyocr.Reader(
        ['es'],
        gpu=False,
        verbose=False
    )

reader = cargar_ocr()

# =========================================================
# STOPWORDS
# =========================================================

STOPWORDS_ES = set(stopwords.words('spanish'))

STOPWORDS_DOMINIO = {

    'sr', 'sra', 'sres',
    'dr', 'dra',
    'certificado',
    'medico',
    'médico',
    'fecha',
    'dias',
    'día',
    'lima',
    'paciente',
    'doctor',
    'doctora'

}

STOPWORDS_COMPLETO = (
    STOPWORDS_ES |
    STOPWORDS_DOMINIO
)

# =========================================================
# PREPROCESAMIENTO
# =========================================================

def corregir_rotacion(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(
        gray,
        0,
        255,
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

    M = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated

def analizar_calidad_imagen(gray):

    brillo = np.mean(gray)

    contraste = np.std(gray)

    blur = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return {

        "brillo": round(brillo,2),
        "contraste": round(contraste,2),
        "blur": round(blur,2)

    }

def preprocesar_imagen(pil_image):

    img = np.array(pil_image)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )

    img = corregir_rotacion(img)

    altura, ancho = img.shape[:2]

    scale = 1.3

    width = int(ancho * scale)
    height = int(altura * scale)

    img = cv2.resize(
        img,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    metricas = analizar_calidad_imagen(gray)

    return {

        'gray': gray,
        'metricas': metricas

    }

# =========================================================
# NLP
# =========================================================

def pipeline_limpieza(texto):

    texto = texto.lower()

    texto = re.sub(
        r'[^a-záéíóúüñ0-9\s]',
        ' ',
        texto
    )

    texto = re.sub(
        r'\s+',
        ' ',
        texto
    )

    tokens = word_tokenize(
        texto,
        language='spanish'
    )

    tokens_limpios = [

        t for t in tokens

        if t not in STOPWORDS_COMPLETO
        and len(t) > 2
        and not t.isdigit()

    ]

    texto_final = " ".join(tokens_limpios)

    return {

        'texto_preprocesado': texto_final,

        'tokens': tokens_limpios,

        'n_tokens_original': len(tokens),

        'n_tokens_limpios': len(tokens_limpios)

    }

# =========================================================
# CLASIFICACIÓN
# =========================================================

def clasificar_documento(texto):

    texto = texto.upper()

    categorias = {

        "Traumatología": [
            "TRAUMA",
            "FRACTURA",
            "ORTOPEDIA"
        ],

        "Neurología": [
            "NEURO",
            "MIGRAÑA"
        ],

        "Cardiología": [
            "CARDIO",
            "CORAZON"
        ]

    }

    for categoria, palabras in categorias.items():

        for palabra in palabras:

            if palabra in texto:
                return categoria

    return "Medicina General"

# =========================================================
# EXTRACCIONES
# =========================================================

def extraer_dni(texto):

    patron = r'(\b\d{8}\b)'

    resultado = re.search(
        patron,
        texto
    )

    return resultado.group(1) if resultado else None

def extraer_edad(texto):

    edad = re.search(
        r'(\d{1,3})\s*años',
        texto,
        re.IGNORECASE
    )

    return edad.group(1) if edad else None

def extraer_cmp(texto):

    cmp = re.search(
        r'cmp\s*[:\-]?\s*(\d+)',
        texto,
        re.IGNORECASE
    )

    return cmp.group(1) if cmp else None

def extraer_fechas(texto):

    fechas = re.findall(
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        texto
    )

    return " | ".join(fechas) if fechas else None

# =========================================================
# SENTIMIENTO
# =========================================================

def analizar_sentimiento(texto):

    try:

        polaridad = TextBlob(
            texto
        ).sentiment.polarity

        if polaridad > 0:
            return "Positivo"

        elif polaridad < 0:
            return "Negativo"

        else:
            return "Neutral"

    except:

        return "Neutral"

# =========================================================
# RESUMEN
# =========================================================

def generar_resumen(texto):

    palabras = texto.split()

    return " ".join(palabras[:25])

# =========================================================
# UPLOADER
# =========================================================

uploaded_files = st.file_uploader(

    "Subir recetas médicas",

    type=["jpg", "jpeg", "png"],

    accept_multiple_files=True

)

# =========================================================
# PREVIEW
# =========================================================

if uploaded_files:

    st.markdown("## Vista previa")

    cols = st.columns(3)

    for idx, file in enumerate(uploaded_files[:6]):

        img = Image.open(file)

        with cols[idx % 3]:

            st.image(
                img,
                use_container_width=True
            )

# =========================================================
# PROCESAMIENTO
# =========================================================

if uploaded_files:

    if st.button("EJECUTAR OCR + NLP"):

        resultados_finales = []

        progress = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):

            image = Image.open(uploaded_file)

            pre = preprocesar_imagen(image)

            if mostrar_preprocesamiento:

                st.markdown(f"## {uploaded_file.name}")

                c1, c2 = st.columns(2)

                with c1:
                    st.image(
                        image,
                        use_container_width=True
                    )

                with c2:
                    st.image(
                        pre['gray'],
                        caption="Escala de grises",
                        use_container_width=True
                    )

            resultado_ocr = reader.readtext(

                pre['gray'],

                detail=0,

                paragraph=True

            )

            texto_ocr = "\n".join(resultado_ocr)

            resultado_nlp = pipeline_limpieza(texto_ocr)

            categoria = clasificar_documento(texto_ocr)

            sentimiento = analizar_sentimiento(texto_ocr)

            resumen = generar_resumen(
                resultado_nlp['texto_preprocesado']
            )

            metricas = pre['metricas']

            resultados_finales.append({

                'archivo': uploaded_file.name,

                'dni': extraer_dni(texto_ocr),

                'edad': extraer_edad(texto_ocr),

                'cmp': extraer_cmp(texto_ocr),

                'fechas': extraer_fechas(texto_ocr),

                'categoria': categoria,

                'sentimiento': sentimiento,

                'resumen': resumen,

                'texto_ocr': texto_ocr,

                'texto_preprocesado':
                    resultado_nlp['texto_preprocesado'],

                'tokens_originales':
                    resultado_nlp['n_tokens_original'],

                'tokens_limpios':
                    resultado_nlp['n_tokens_limpios'],

                'brillo':
                    metricas['brillo'],

                'contraste':
                    metricas['contraste'],

                'blur':
                    metricas['blur']

            })

            progress.progress(
                (i + 1) / len(uploaded_files)
            )

            gc.collect()

        st.session_state.df_final = pd.DataFrame(
            resultados_finales
        )

        st.success("OCR + NLP completado")

# =========================================================
# RESULTADOS
# =========================================================

if 'df_final' in st.session_state:

    df_final = st.session_state.df_final

    texto_total = " ".join(
        df_final['texto_preprocesado']
    )

    palabras = texto_total.split()

    frecuencias = Counter(palabras)

    st.markdown("## Métricas")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="metric-card card1">
            <div class="metric-title">Documentos</div>
            <div class="metric-value">{len(df_final)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card card2">
            <div class="metric-title">Tokens</div>
            <div class="metric-value">
            {int(df_final['tokens_originales'].mean())}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card card3">
            <div class="metric-title">Limpios</div>
            <div class="metric-value">
            {int(df_final['tokens_limpios'].mean())}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card card4">
            <div class="metric-title">Categorías</div>
            <div class="metric-value">
            {df_final['categoria'].nunique()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card card5">
            <div class="metric-title">Keywords</div>
            <div class="metric-value">
            {len(frecuencias)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([

        "Resultados",
        "NLP",
        "WordCloud",
        "Tabla",
        "Exportar"

    ])

    # =====================================================
    # RESULTADOS
    # =====================================================

    with tab1:

        archivo = st.selectbox(
            "Seleccionar archivo",
            df_final['archivo']
        )

        fila = df_final[
            df_final['archivo'] == archivo
        ].iloc[0]

        st.markdown("### Datos extraídos")

        st.write(f"**DNI:** {fila['dni']}")
        st.write(f"**Edad:** {fila['edad']}")
        st.write(f"**CMP:** {fila['cmp']}")
        st.write(f"**Fechas:** {fila['fechas']}")
        st.write(f"**Categoría:** {fila['categoria']}")
        st.write(f"**Sentimiento:** {fila['sentimiento']}")

        st.markdown("### Texto OCR")

        st.text_area(
            "",
            fila['texto_ocr'],
            height=250
        )

    # =====================================================
    # NLP
    # =====================================================

    with tab2:

        top20 = frecuencias.most_common(20)

        if len(top20) > 0:

            palabras_top, conteos = zip(*top20)

            fig, ax = plt.subplots(figsize=(12,6))

            ax.barh(

                list(reversed(palabras_top)),
                list(reversed(conteos))

            )

            ax.set_title("Top 20 palabras")

            st.pyplot(fig)

    # =====================================================
    # WORDCLOUD
    # =====================================================

    with tab3:

        texto_wc = " ".join(palabras)

        if texto_wc.strip() != "":

            wordcloud = WordCloud(

                width=1000,
                height=500,

                background_color='white',

                max_words=50,

                stopwords=STOPWORDS_COMPLETO,

                collocations=False

            ).generate(texto_wc)

            fig_wc, ax_wc = plt.subplots(figsize=(12,6))

            ax_wc.imshow(
                wordcloud,
                interpolation='bilinear'
            )

            ax_wc.axis("off")

            st.pyplot(fig_wc)

    # =====================================================
    # TABLA
    # =====================================================

    with tab4:

        st.dataframe(
            df_final,
            use_container_width=True
        )

    # =====================================================
    # EXPORTAR
    # =====================================================

    with tab5:

        csv = df_final.to_csv(
            index=False
        ).encode('utf-8-sig')

        st.download_button(

            "Descargar CSV",

            csv,

            "resultado_ocr.csv",

            "text/csv"

        )

        output = BytesIO()

        with pd.ExcelWriter(

            output,

            engine='openpyxl'

        ) as writer:

            df_final.to_excel(
                writer,
                index=False
            )

        st.download_button(

            "Descargar Excel",

            output.getvalue(),

            "resultado_ocr.xlsx",

            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        )
```
