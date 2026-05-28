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
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from io import BytesIO
from wordcloud import WordCloud
from textblob import TextBlob
#import spacy
import gc
import seaborn as sns

from rapidfuzz import fuzz
#from spacy.pipeline import EntityRuler

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
# CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f4f7fc;
}

/* TITULOS */

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

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* METRICAS */

.metric-card {
    padding: 22px;
    border-radius: 22px;
    color: white;
    text-align: center;
    font-weight: bold;
    box-shadow: 0px 8px 18px rgba(0,0,0,0.15);
    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.metric-title {
    font-size: 17px;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 34px;
}

/* TARJETAS */

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

/* DIVISOR */

.divisor {
    margin-top: 35px;
    margin-bottom: 20px;
    padding: 18px;
    border-radius: 16px;
    background: linear-gradient(90deg,#0f172a,#1e293b);
    color: white;
    text-align: center;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 1px;
}

/* TABS */

.stTabs [data-baseweb="tab-list"] {

    display: grid;

    grid-template-columns: repeat(6, 1fr);

    width: 100%;

    gap: 18px;

    margin-top: 20px;
    margin-bottom: 20px;
}

.stTabs [data-baseweb="tab"] {

    height: 75px;

    width: 100%;

    border-radius: 20px;

    font-size: 17px;

    font-weight: 900;

    color: white;

    border: none;

    transition: 0.3s;
}

/* RESULTADOS */

button[id*="tab-0"] {

    background: linear-gradient(135deg,#2563eb,#3b82f6);
}

/* NLP */

button[id*="tab-1"] {

    background: linear-gradient(135deg,#9333ea,#a855f7);
}

/* WORDCLOUD */

button[id*="tab-2"] {

    background: linear-gradient(135deg,#059669,#10b981);
}

/* CLASIFICACION */

button[id*="tab-3"] {

    background: linear-gradient(135deg,#ea580c,#f97316);
}

/* SENTIMIENTOS */

button[id*="tab-4"] {

    background: linear-gradient(135deg,#dc2626,#ef4444);
}

/* EXPORTAR */

button[id*="tab-5"] {

    background: linear-gradient(135deg,#0f172a,#334155);
}

/* TAB ACTIVA */

.stTabs [aria-selected="true"] {

    transform: scale(1.05);

    box-shadow: 0px 10px 18px rgba(0,0,0,0.25);
}

/* BOTONES */

.stButton>button {
    background: linear-gradient(90deg,#2563eb,#60a5fa);
    color: white;
    border-radius: 14px;
    border: none;
    height: 55px;
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

/* PREVIEW */

.preview-container {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}

/* DATAFRAME */

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 2px solid #e2e8f0;
}

/* ALERTAS */

.stAlert {
    border-radius: 16px;
}

/* TABLA GRANDE */

div[data-testid="stDataFrame"] > div {
    overflow-x: auto !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '''
    <div class="titulo">
    Receta Médica Perú: Extracción Inteligente de Recetas y Documentos Clínicos
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    '''
    <div class="subtitulo">
    Extracción y análisis inteligente de texto en imágenes y documentos utilizando OCR y Procesamiento de Lenguaje Natural
    </div>
    ''',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Panel de Configuración")

st.sidebar.info("""
OCR automático  
Procesamiento NLP  
Clasificación temática  
WordCloud personalizada  
Resumen automático  
Análisis de sentimiento  
Extracción de entidades  
Exportación CSV y Excel  
""")

mostrar_preprocesamiento = st.sidebar.checkbox(
    "Mostrar Preprocesamiento",
    value=True
)

# =========================================================
# DESCARGAS NLTK
# =========================================================

@st.cache_resource
def descargar_nltk():

    nltk.download('punkt')
    nltk.download('stopwords')

descargar_nltk()

# =========================================================
# SPACY
# =========================================================

@st.cache_resource
def cargar_spacy():

    nlp = spacy.blank("es")

    ruler = nlp.add_pipe(
        "entity_ruler"
    )

    patterns = [

        {
            "label": "CLINICA",
            "pattern": "Clínica San Pablo"
        },

        {
            "label": "CLINICA",
            "pattern": "Clínica Ricardo Palma"
        },

        {
            "label": "CLINICA",
            "pattern": "Hospital Rebagliati"
        },

        {
            "label": "SERVICIO",
            "pattern": "Traumatología"
        },

        {
            "label": "SERVICIO",
            "pattern": "Neurología"
        },

        {
            "label": "SERVICIO",
            "pattern": "Cardiología"
        }

    ]

    ruler.add_patterns(patterns)

    return nlp

#nlp = cargar_spacy()

# =========================================================
# OCR
# =========================================================

@st.cache_resource
def cargar_ocr():

    return easyocr.Reader(
        ['es'],
        gpu=False
    )

# =========================================================
# STOPWORDS
# =========================================================

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

    blancos = np.sum(gray > 240)

    porcentaje_blanco = blancos / gray.size * 100

    if brillo < 80:
        estado_brillo = "Oscura"

    elif brillo > 180:
        estado_brillo = "Muy clara"

    else:
        estado_brillo = "Normal"

    if blur < 100:
        estado_blur = "Borroso"

    else:
        estado_blur = "Nítido"

    return {

        "brillo": round(brillo,2),
        "contraste": round(contraste,2),
        "blur": round(blur,2),
        "porcentaje_blanco": round(porcentaje_blanco,2),
        "estado_brillo": estado_brillo,
        "estado_blur": estado_blur

    }

def preprocesar_imagen(pil_image):

    img = np.array(pil_image)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )

    img = corregir_rotacion(img)

    altura, ancho = img.shape[:2]

    scale = 1.5

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

        'original': img,
        'gray': gray,
        'metricas': metricas

    }

# =========================================================
# NLP
# =========================================================

def convertir_minusculas(texto):

    if pd.isna(texto):
        return ''

    return texto.lower()

def eliminar_caracteres_especiales(texto):

    texto = re.sub(
        r'[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9\s.,:/\-]',
        ' ',
        texto
    )

    return texto

def normalizar_espacios(texto):

    texto = re.sub(r'[\n\r\t]+', ' ', texto)

    texto = re.sub(r' {2,}', ' ', texto)

    return texto.strip()

def tokenizar(texto):

    return word_tokenize(
        texto,
        language='spanish'
    )

def eliminar_stopwords(tokens):

    resultado = [

        t for t in tokens

        if t.lower() not in STOPWORDS_COMPLETO
        and len(t) > 2
        and not re.fullmatch(r'\d+', t)

    ]

    return resultado

def pipeline_limpieza(texto):

    t1 = convertir_minusculas(texto)

    t2 = eliminar_caracteres_especiales(t1)

    t3 = normalizar_espacios(t2)

    tokens = tokenizar(t3)

    tokens_limpios = eliminar_stopwords(tokens)

    texto_final = ' '.join(tokens_limpios)

    return {

        'texto_preprocesado': texto_final,

        'tokens': tokens_limpios,

        'n_tokens_original': len(tokens),

        'n_tokens_limpios': len(tokens_limpios)

    }

# =========================================================
# CLASIFICACION
# =========================================================

def clasificar_documento(texto):

    texto = texto.upper()

    categorias = {

        "Traumatología": [
            "TRAUMA",
            "FRACTURA",
            "ORTOPEDIA",
            "LUMBALGIA"
        ],

        "Neurología": [
            "NEURO",
            "CEREBRO",
            "MIGRAÑA"
        ],

        "Cardiología": [
            "CARDIO",
            "CORAZON",
            "PRESION"
        ],

        "Pediatría": [
            "PEDIATR",
            "NIÑO",
            "INFANTE"
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

    patrones = [

        r'(?:DNI|D\.N\.I|DOCUMENTO DE IDENTIDAD|DOC\. IDENTIDAD)'
        r'\s*[:\-]?\s*(\d{8})',

        r'(\b\d{8}\b)'

    ]

    for patron in patrones:

        resultado = re.search(
            patron,
            texto,
            re.IGNORECASE
        )

        if resultado:
            return resultado.group(1)

    return None

def extraer_edad(texto):

    edad = re.search(
        r'(?:edad|años)\s*[:\-]?\s*(\d{1,3})',
        texto,
        re.IGNORECASE
    )

    return edad.group(1) if edad else None

def extraer_cmp(texto):

    cmp = re.search(
        r'(?:cmp)\s*[:\-]?\s*(\d+)',
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

def extraer_nombre(texto):

    patrones = [

        # PACIENTE / NOMBRE
        r'(?:PACIENTE|NOMBRE|APELLIDOS Y NOMBRE|APELLIDOS Y NOMBRES)'
        r'\s*[:\-]?\s*'
        r'([A-ZÁÉÍÓÚÑ,\s]{5,})',

        # SR / SRA
        r'(?:SR\(A\)|SRA|SR|SEÑOR|SEÑORA)'
        r'\s*[:\-]?\s*'
        r'([A-ZÁÉÍÓÚÑ,\s]{5,})',

        # CONSTA POR LA PRESENTE QUE:
        r'(?:CONSTA POR LA PRESENTE QUE)'
        r'\s*[:\-]?\s*'
        r'([A-ZÁÉÍÓÚÑ,\s]{5,})'

    ]

    for patron in patrones:

        resultado = re.search(
            patron,
            texto,
            re.IGNORECASE
        )

        if resultado:

            nombre = resultado.group(1)

            # cortar cuando aparezcan palabras típicas
            cortes = [
                "DNI",
                "C.E",
                "CE",
                "IDENTIFICADO",
                "IDENTIFICADA",
                "EDAD",
                "SEXO",
                "CMP"
            ]

            for corte in cortes:

                if corte in nombre.upper():

                    nombre = nombre.upper().split(corte)[0]

            nombre = re.sub(
                r'\s+',
                ' ',
                nombre
            ).strip()

            return nombre[:80]

    return None

def extraer_sexo(texto):

    sexo = re.search(

        r'(?:sexo|género)'
        r'\s*[:\-]?\s*'
        r'(masculino|femenino|m|f)',

        texto,

        re.IGNORECASE

    )

    if sexo:

        valor = sexo.group(1).upper()

        if valor in ['M','MASCULINO']:
            return "Masculino"

        if valor in ['F','FEMENINO']:
            return "Femenino"

    return None

def extraer_servicio(texto):

    servicios_medicos = [

        "TRAUMATOLOGIA",
        "NEUROLOGIA",
        "CARDIOLOGIA",
        "PEDIATRIA",
        "MEDICINA GENERAL"

    ]

    palabras = texto.split()

    mejor = None
    score_max = 0

    for palabra in palabras:

        for servicio_real in servicios_medicos:

            score = fuzz.ratio(
                palabra.upper(),
                servicio_real.upper()
            )

            if score > score_max:

                score_max = score
                mejor = servicio_real

    if score_max > 80:

        return mejor.title()

    return "Medicina General"

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

    "Subir Certificados Médicos",

    type=["jpg", "jpeg", "png"],

    accept_multiple_files=True

)

# =========================================================
# PREVIEW
# =========================================================

if uploaded_files:

    st.markdown("## Vista Previa")

    cols = st.columns(3)

    for idx, file in enumerate(uploaded_files[:9]):

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

        with st.spinner("Cargando modelo OCR..."):
            reader = cargar_ocr()

        resultados_finales = []

        progress = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):

            image = Image.open(uploaded_file)

            pre = preprocesar_imagen(image)

            if mostrar_preprocesamiento:

                st.markdown(
                    f"## {uploaded_file.name}"
                )

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

                paragraph=True,

                batch_size=1

            )

            texto_ocr = "\n".join(resultado_ocr)

            resultado_nlp = pipeline_limpieza(texto_ocr)

            categoria = clasificar_documento(texto_ocr)

            sentimiento = analizar_sentimiento(texto_ocr)

            resumen = generar_resumen(
                resultado_nlp['texto_preprocesado']
            )

            metricas_img = pre['metricas']

            resultados_finales.append({

                'archivo': uploaded_file.name,

                'nombre': extraer_nombre(texto_ocr),

                'sexo': extraer_sexo(texto_ocr),

                'dni': extraer_dni(texto_ocr),

                'edad': extraer_edad(texto_ocr),

                'cmp': extraer_cmp(texto_ocr),

                'fechas': extraer_fechas(texto_ocr),

                'servicio': extraer_servicio(texto_ocr),

                'categoria_tematica': categoria,

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
                    metricas_img['brillo'],

                'contraste':
                    metricas_img['contraste'],

                'blur':
                    metricas_img['blur'],

                'estado_brillo':
                    metricas_img['estado_brillo'],

                'estado_blur':
                    metricas_img['estado_blur']

            })

            progress.progress(
                (i + 1) / len(uploaded_files)
            )

            gc.collect()

        df_final = pd.DataFrame(resultados_finales)

        st.success("OCR + NLP FINALIZADO")

        texto_total = " ".join(
            df_final['texto_preprocesado']
        )

        palabras = texto_total.split()

        frecuencias = Counter(palabras)

        # =====================================================
        # METRICAS
        # =====================================================

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
                <div class="metric-value">{int(df_final['tokens_originales'].mean())}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card card3">
                <div class="metric-title">Limpios</div>
                <div class="metric-value">{int(df_final['tokens_limpios'].mean())}</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card card4">
                <div class="metric-title">Categorías</div>
                <div class="metric-value">{df_final['categoria_tematica'].nunique()}</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:
            st.markdown(f"""
            <div class="metric-card card5">
                <div class="metric-title">Keywords</div>
                <div class="metric-value">{len(frecuencias)}</div>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# MOSTRAR RESULTADOS SOLO SI EXISTE DF_FINAL
# =====================================================

if 'df_final' in locals():

    # =====================================================
    # DIVISOR
    # =====================================================

    st.markdown("""
    <div class="divisor">
    PANEL DE RESULTADOS Y ANÁLISIS NLP
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([

        "Resultados",
        "NLP",
        "WordCloud",
        "Clasificación",
        "Sentimientos",
        "Métricas Avanzadas",
        "Exportar"

    ])

    # =====================================================
    # MÉTRICAS EXTRA
    # =====================================================

    df_final['reduccion_tokens'] = np.where(

        df_final['tokens_originales'] > 0,

        (
            (
                df_final['tokens_originales']
                - df_final['tokens_limpios']
            )
            / df_final['tokens_originales']
        ) * 100,

        0

    )

    df_final['tokens_limpios_lista'] = df_final[
        'texto_preprocesado'
    ].apply(lambda x: x.split())

    # =====================================================
    # RESULTADOS
    # =====================================================

    with tab1:

        st.markdown("## Resultado Completo")

        archivos_disponibles = df_final['archivo'].tolist()

        archivo_seleccionado = st.selectbox(
            "Seleccionar documento",
            archivos_disponibles
        )

        fila = df_final[
            df_final['archivo'] == archivo_seleccionado
        ].iloc[0]

        # =================================================
        # TARJETAS RESUMEN
        # =================================================

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.info(f"""
            Sexo: {fila['sexo']}
            
            Edad: {fila['edad']}
            """)

        with c2:
            st.info(f"""
            DNI: {fila['dni']}
            
            CMP: {fila['cmp']}
            """)

        with c3:
            st.info(f"""
            Servicio: {fila['servicio']}
            
            Categoría: {fila['categoria_tematica']}
            """)

        with c4:
            st.info(f"""
            Sentimiento: {fila['sentimiento']}
            
            Blur: {fila['estado_blur']}
            """)

        # =================================================
        # DATOS PRINCIPALES
        # =================================================

        st.markdown("### Información Extraída")

        info_html = f"""
        <div style="
            background:white;
            padding:25px;
            border-radius:18px;
            box-shadow:0 4px 12px rgba(0,0,0,0.08);
            color:#0f172a;
            line-height:1.8;
            font-size:16px;
        ">

        <h3 style="color:#2563eb;">
        {fila['archivo']}
        </h3>

        <b>Nombre:</b> {fila['nombre']}<br>
        <b>Sexo:</b> {fila['sexo']}<br>
        <b>DNI:</b> {fila['dni']}<br>
        <b>Edad:</b> {fila['edad']}<br>
        <b>CMP:</b> {fila['cmp']}<br>
        <b>Fechas:</b> {fila['fechas']}<br>
        <b>Servicio:</b> {fila['servicio']}<br>
        <b>Categoría:</b> {fila['categoria_tematica']}<br>
        <b>Sentimiento:</b> {fila['sentimiento']}<br>

        </div>
        """

        st.markdown(
            info_html,
            unsafe_allow_html=True
        )

        # =================================================
        # MÉTRICAS DE IMAGEN
        # =================================================

        st.markdown("### Calidad de Imagen")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Brillo",
                fila['brillo']
            )

        with c2:
            st.metric(
                "Contraste",
                fila['contraste']
            )

        with c3:
            st.metric(
                "Blur",
                fila['blur']
            )

        with c4:
            st.metric(
                "Estado",
                fila['estado_blur']
            )

        # =================================================
        # TEXTO OCR
        # =================================================

        st.markdown("### Texto OCR")

        st.text_area(
            "",
            fila['texto_ocr'],
            height=300
        )

        # =================================================
        # TEXTO PREPROCESADO
        # =================================================

        st.markdown("### Texto Preprocesado")

        st.text_area(
            "",
            fila['texto_preprocesado'],
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

            bars = ax.barh(

                list(reversed(palabras_top)),
                list(reversed(conteos)),

                color=plt.cm.plasma(
                    np.linspace(0.2, 0.9, len(top20))
                ),

                edgecolor='white'

            )

            ax.set_title(
                "Top 20 Palabras",
                fontsize=15,
                fontweight='bold'
            )

            ax.set_xlabel("Frecuencia")

            ax.grid(axis='x', alpha=0.3)

            for bar, val in zip(
                bars,
                list(reversed(conteos))
            ):

                ax.text(
                    val + 0.3,
                    bar.get_y() + bar.get_height()/2,
                    str(val),
                    va='center'
                )

            st.pyplot(fig)

    # =====================================================
    # WORDCLOUD
    # =====================================================

    with tab3:

        texto_final = " ".join(palabras)

        if texto_final.strip() != "":

            wordcloud = WordCloud(

                width=1800,
                height=900,

                background_color='white',

                max_words=50,

                stopwords=STOPWORDS_COMPLETO,

                collocations=False,

                contour_width=2

            ).generate(texto_final)

            fig_wc, ax_wc = plt.subplots(figsize=(16,8))

            ax_wc.imshow(
                wordcloud,
                interpolation='bilinear'
            )

            ax_wc.axis("off")

            st.pyplot(fig_wc)

    # =====================================================
    # CLASIFICACION
    # =====================================================

    with tab4:

        conteo = df_final[
            'categoria_tematica'
        ].value_counts()

        fig2, ax2 = plt.subplots(figsize=(10,6))

        colores_cat = plt.cm.Set2(
            np.linspace(0,1,len(conteo))
        )

        barras = ax2.bar(

            conteo.index,
            conteo.values,

            color=colores_cat,

            edgecolor='black'

        )

        ax2.set_title(
            "Clasificación Temática",
            fontsize=15,
            fontweight='bold'
        )

        ax2.set_ylabel("Cantidad")

        ax2.grid(axis='y', alpha=0.3)

        for bar in barras:

            height = bar.get_height()

            ax2.text(

                bar.get_x() + bar.get_width()/2,
                height + 0.1,
                str(int(height)),

                ha='center',
                fontsize=11,
                fontweight='bold'

            )

        st.pyplot(fig2)

    # =====================================================
    # SENTIMIENTOS
    # =====================================================

    with tab5:

        conteo_sentimientos = df_final[
            'sentimiento'
        ].value_counts()

        fig3, ax3 = plt.subplots(figsize=(7,5))

        colores_sent = [

            '#22c55e',
            '#ef4444',
            '#facc15'

        ]

        ax3.pie(

            conteo_sentimientos.values,

            labels=conteo_sentimientos.index,

            autopct='%1.1f%%',

            startangle=90,

            colors=colores_sent,

            wedgeprops={
                'edgecolor': 'white',
                'linewidth': 2
            }

        )

        ax3.set_title(
            "Análisis de Sentimiento",
            fontsize=15,
            fontweight='bold'
        )

        st.pyplot(fig3)




    # =====================================================
    # MÉTRICAS AVANZADAS
    # =====================================================

    with tab6:

        st.markdown(
            "## Métricas Avanzadas de Limpieza NLP"
        )

        # =================================================
        # FIGURA GENERAL
        # =================================================

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(20, 6)
        )

        fig.suptitle(

            'Métricas de Limpieza — Certificados Médicos Peruanos',

            fontsize=16,

            fontweight='bold'

        )

        colores = ['#2196F3', '#4CAF50']

        # =================================================
        # GRAFICO 1
        # =================================================

        ax1 = axes[0]

        x = range(len(df_final))

        ancho = 0.4

        ax1.bar(

            [i - ancho/2 for i in x],

            df_final['tokens_originales'],

            width=ancho,

            color=colores[0],

            alpha=0.85,

            label='Tokens OCR'

        )

        ax1.bar(

            [i + ancho/2 for i in x],

            df_final['tokens_limpios'],

            width=ancho,

            color=colores[1],

            alpha=0.85,

            label='Tokens Limpios'

        )

        ax1.set_title(

            'Tokens por Certificado\n(antes vs después)',

            fontsize=12,

            fontweight='bold'

        )

        ax1.set_xlabel(
            'Índice del certificado'
        )

        ax1.set_ylabel(
            'Número de tokens'
        )

        ax1.legend()

        ax1.grid(axis='y', alpha=0.3)

        # =================================================
        # GRAFICO 2
        # =================================================

        ax2 = axes[1]

        reduccion_valida = df_final[
            'reduccion_tokens'
        ].dropna()

        ax2.hist(

            reduccion_valida,

            bins=10,

            color='#9C27B0',

            edgecolor='white',

            alpha=0.9

        )

        ax2.axvline(

            reduccion_valida.mean(),

            color='red',

            linestyle='--',

            linewidth=2,

            label=f'Media: {reduccion_valida.mean():.1f}%'

        )

        ax2.set_title(

            'Distribución de la\nReducción de Tokens (%)',

            fontsize=12,

            fontweight='bold'

        )

        ax2.set_xlabel('Reducción (%)')

        ax2.set_ylabel('Frecuencia')

        ax2.legend()

        ax2.grid(axis='y', alpha=0.3)

        # =================================================
        # GRAFICO 3
        # =================================================

        ax3 = axes[2]

        promedio_antes = df_final[
            'tokens_originales'
        ].mean()

        promedio_despues = df_final[
            'tokens_limpios'
        ].mean()

        promedio_eliminados = (
            promedio_antes - promedio_despues
        )

        sizes = [

            promedio_despues,

            promedio_eliminados

        ]

        labels = [

            f'Tokens útiles\n({promedio_despues:.0f} prom.)',

            f'Tokens eliminados\n({promedio_eliminados:.0f} prom.)'

        ]

        colores_torta = [

            '#4CAF50',
            '#F44336'

        ]

        wedges, texts, autotexts = ax3.pie(

            sizes,

            labels=labels,

            colors=colores_torta,

            autopct='%1.1f%%',

            startangle=90,

            wedgeprops={

                'edgecolor': 'white',

                'linewidth': 2

            }

        )

        for at in autotexts:

            at.set_fontsize(12)

            at.set_fontweight('bold')

        ax3.set_title(

            'Proporción Promedio de\nTokens',

            fontsize=12,

            fontweight='bold'

        )

        plt.tight_layout()

        st.pyplot(fig)

        # =================================================
        # TOP 30 PALABRAS
        # =================================================

        st.markdown(
            "## Top 30 Palabras Más Frecuentes"
        )

        from itertools import chain

        todos_tokens = list(

            chain.from_iterable(
                df_final['tokens_limpios_lista']
            )

        )

        frecuencias_top = Counter(todos_tokens)

        top30 = frecuencias_top.most_common(30)

        if len(top30) > 0:

            palabras, conteos = zip(*top30)

            fig2, ax = plt.subplots(
                figsize=(14, 8)
            )

            barras = ax.barh(

                list(reversed(palabras)),

                list(reversed(conteos)),

                color=plt.cm.viridis_r(

                    [i/30 for i in range(30)]

                ),

                edgecolor='white'

            )

            ax.set_title(

                'Top 30 Palabras Más Frecuentes\n'
                '(Texto Preprocesado)',

                fontsize=14,

                fontweight='bold'

            )

            ax.set_xlabel('Frecuencia')

            ax.grid(axis='x', alpha=0.3)

            for bar, val in zip(

                barras,

                list(reversed(conteos))

            ):

                ax.text(

                    val + 0.3,

                    bar.get_y() + bar.get_height()/2,

                    str(val),

                    va='center',

                    fontsize=9

                )

            st.pyplot(fig2)

        # =================================================
        # KPIS EXTRA
        # =================================================

        st.markdown("## Indicadores NLP")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(

                "Reducción promedio",

                f"{df_final['reduccion_tokens'].mean():.1f}%"

            )

        with c2:

            st.metric(

                "Máx. tokens OCR",

                int(df_final['tokens_originales'].max())

            )

        with c3:

            st.metric(

                "Máx. tokens limpios",

                int(df_final['tokens_limpios'].max())

            )

    # =====================================================
    # EXPORTAR
    # =====================================================

    with tab7:

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
