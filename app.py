import os, re, json, hashlib, time, sqlite3, logging, threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from enum import Enum

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

load_dotenv()

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MI_TOKEN = os.getenv("GITHUB_TOKEN", "")
MI_TOKEN2 = os.getenv("GITHUB_TOKEN2", "")
REMITE = os.getenv("CONFIG_EMAIL_REMITENTE")
REMPASS = os.getenv("CONFIG_EMAIL_PASSWORD")
KB_PATH = Path(__file__).parent / "data" / "knowledge_base.json"
DB_PATH = Path(__file__).parent / "data" / "meliexpert.db"

TOKENS = [t for t in [MI_TOKEN, MI_TOKEN2] if t]
if not TOKENS:
    st.warning("GITHUB_TOKEN no configurado. Edita .env")

llm = ChatOpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=TOKENS[0] if TOKENS else "",
    model="gpt-4o",
    temperature=0.1,
    streaming=True,
)

embeddings = OpenAIEmbeddings(
    base_url="https://models.github.ai/inference",
    api_key=TOKENS[0] if TOKENS else "",
    model="text-embedding-3-small",
)

# ──────────────────────────────────────────────
# (1) SEGURIDAD — filtros éticos, PII, inyección
# ──────────────────────────────────────────────
import base64
import binascii

# ── PII Patterns ──
PII_PATTERNS = {
    "tarjeta_credito": re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "telefono": re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3}[-.\s]?\d{3,4}\b'),
    "dni": re.compile(r'\b\d{7,8}\b'),
    "rut": re.compile(r'\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b'),
    "cuenta_bancaria": re.compile(r'\b\d{3,5}-\d{4,6}-\d{2,4}-\d{2,4}\b'),
    "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    "password": re.compile(r'(?:password|contraseña|passwd|pwd|clave|secret)[\s:=]+\S+', re.IGNORECASE),
}

# ── Injection Patterns (mejorados) ──
INJECTION_PATTERNS = [
    # Inyección básica
    re.compile(r'ignora\s*(?:las\s*)?instrucciones', re.IGNORECASE),
    re.compile(r'ignora\s*(?:el\s*)?prompt', re.IGNORECASE),
    re.compile(r'olvida\s*(?:tu\s*)?(?:instrucción|prompt|rol|system|todo)', re.IGNORECASE),
    re.compile(r'eres\s*(?:ahora|libre|un\s*(?:asistente|bot|humano))', re.IGNORECASE),
    re.compile(r'system\s*(?:prompt|message|instruction)', re.IGNORECASE),
    re.compile(r'skip\s*(?:the\s*)?(?:instructions|prompt|rules)', re.IGNORECASE),
    
    # Jailbreak
    re.compile(r'do\s*anything\s*now', re.IGNORECASE),
    re.compile(r'pretend\s*(?:that\s*)?(?:you\s*)?(?:are|have|can)', re.IGNORECASE),
    re.compile(r'act\s*(?:as\s*)?(?:if\s*)?(?:you\s*)?(?:are|were|have)', re.IGNORECASE),
    re.compile(r'role\s*play\s*(?:as|like)', re.IGNORECASE),
    re.compile(r'simula\s*ser', re.IGNORECASE),
    re.compile(r'finge\s*ser', re.IGNORECASE),
    re.compile(r'actua\s*como\s*si', re.IGNORECASE),
    
    # Prompt leak
    re.compile(r'(?:muéstrate|muestra|dime|cuál\s*es|imprime|print)\s*(?:tu\s*)?(?:prompt|system|instrucción|instrucciones)', re.IGNORECASE),
    re.compile(r'what\s*(?:is|are)\s*(?:your|the)\s*(?:system\s*)?(?:prompt|instructions)', re.IGNORECASE),
    re.compile(r'reveal\s*(?:your|the)\s*(?:system\s*)?(?:prompt|instructions)', re.IGNORECASE),
    re.compile(r'repetir\s*(?:tu\s*)?(?:prompt|system|instrucción)', re.IGNORECASE),
    re.compile(r'reproduce\s*(?:tu\s*)?(?:prompt|system)', re.IGNORECASE),
    
    # Manipulación de rol
    re.compile(r'no\s*(?:eres|seas)\*(?:un?\s*)?(?:bot|asistente|IA|inteligencia)', re.IGNORECASE),
    re.compile(r'(?:deja|cambia|cambiar)\s*(?:de\s*)?(?:rol|personaje|identidad)', re.IGNORECASE),
    re.compile(r'(?:nuevo|otro|diferente)\s*(?:rol|personaje|identidad|personalidad)', re.IGNORECASE),
    
    # Inyección编码 (base64, hex)
    re.compile(r'(?:decodifica|decode|decodear)\s*(?:esto|esto\s*es|el\s*siguiente|el\s*texto)', re.IGNORECASE),
    re.compile(r'(?:exec|ejecuta|run|correr)\s*(?:this|esto)', re.IGNORECASE),
    
    # SQL Injection
    re.compile(r"(?:SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\s+", re.IGNORECASE),
    re.compile(r'(?:\'\s*(?:OR|AND)\s*\'?\s*=\s*\'?)', re.IGNORECASE),
    
    # XSS
    re.compile(r'<\s*script', re.IGNORECASE),
    re.compile(r'javascript\s*:', re.IGNORECASE),
    re.compile(r'on\w+\s*=', re.IGNORECASE),
    
    # Intento de acceder a archivos del sistema
    re.compile(r'(?:/etc/passwd|/etc/shadow|~/.ssh|\.env|config\.json|credentials)', re.IGNORECASE),
    
    # Intento de usar herramientas externas
    re.compile(r'(?:curl|wget|fetch|requests)\s+(?:http|ftp)', re.IGNORECASE),
]

# ── Rate Limiting ──
_rate_limit: Dict[str, List[float]] = {}
MAX_REQUESTS_PER_MINUTE = 15
MAX_MESSAGE_LENGTH = 2000

def check_rate_limit(session_id: str) -> bool:
    """Retorna True si el usuario debe ser bloqueado."""
    now = time.time()
    if session_id not in _rate_limit:
        _rate_limit[session_id] = []
    _rate_limit[session_id] = [t for t in _rate_limit[session_id] if now - t < 60]
    if len(_rate_limit[session_id]) >= MAX_REQUESTS_PER_MINUTE:
        return True
    _rate_limit[session_id].append(now)
    return False

def detect_encoded_text(text: str) -> bool:
    """Detecta texto codificado en base64 o hex."""
    # Base64
    try:
        if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text):
            decoded = base64.b64decode(re.search(r'([A-Za-z0-9+/]{20,}={0,2})', text).group(1)).decode('utf-8', errors='ignore')
            if any(pat.search(decoded) for pat in INJECTION_PATTERNS):
                return True
    except Exception:
        pass
    # Hex
    try:
        hex_match = re.search(r'(?:0x)?([0-9a-fA-F]{20,})', text)
        if hex_match:
            decoded = bytes.fromhex(hex_match.group(1)).decode('utf-8', errors='ignore')
            if any(pat.search(decoded) for pat in INJECTION_PATTERNS):
                return True
    except Exception:
        pass
    return False

def detect_obfuscation(text: str) -> bool:
    """Detecta ofuscación de texto."""
    # Letras separadas por espacios: "i g n o r a"
    if re.search(r'(?:^|\s)([a-zA-Z]\s){4,}', text):
        return True
    # Caracteres Unicode suspicious
    if re.search(r'[\u200b\u200c\u200d\ufeff]', text):
        return True
    # Repetición excesiva de caracteres
    if re.search(r'(.)\1{5,}', text):
        return True
    return False

def detect_system_prompt_leak(text: str) -> bool:
    """Detecta intentos de extraer el system prompt."""
    leak_patterns = [
        re.compile(r'(?:repeat|repite|print|imprime|show|muestra|tell|dime|what|cuál)\s+(?:your|tu|el)\s+(?:system|initial|original|first|primer)\s+(?:prompt|message|instruction|mensaje|instrucción)', re.IGNORECASE),
        re.compile(r'(?:before|antes|previous|anterior|earlier|primero)\s+(?:you|tú|te)\s+(?:were|eras|had|tenías|received|recibiste|got|tuviste)', re.IGNORECASE),
        re.compile(r'(?:start|iniciar|begin|empezar)\s+(?:with|con|from|desde)\s+(?:your|tu|the|el)\s+(?:system|inicial|original)', re.IGNORECASE),
        re.compile(r'(?:ignore|olvidar|resetear|reset)\s+(?:all|todo|todas)\s+(?:previous|anteriores|earlier|antes)\s+(?:instructions|instrucciones)', re.IGNORECASE),
        re.compile(r'(?:from|desde|starting|empezando)\s+now|ahora', re.IGNORECASE),
    ]
    return any(p.search(text) for p in leak_patterns)


class SecurityReport:
    def __init__(self):
        self.pii_detected: Dict[str, List[str]] = {}
        self.injection_detected: List[str] = []
        self.blocked = False
        self.block_reason = ""
        self.risk_level = "low"  # low, medium, high, critical

    @staticmethod
    def scan(text: str, session_id: str = "") -> "SecurityReport":
        r = SecurityReport()
        
        # Rate limiting
        if session_id and check_rate_limit(session_id):
            r.blocked = True
            r.block_reason = "Has enviado demasiadas solicitudes. Espera un momento."
            r.risk_level = "high"
            return r
        
        # Longitud del mensaje
        if len(text) > MAX_MESSAGE_LENGTH:
            r.blocked = True
            r.block_reason = f"Tu mensaje excede el límite de {MAX_MESSAGE_LENGTH} caracteres."
            r.risk_level = "medium"
            return r
        
        # Detección de PII
        for k, pat in PII_PATTERNS.items():
            m = pat.findall(text)
            if m:
                r.pii_detected[k] = m
        
        # Detección de inyección
        for pat in INJECTION_PATTERNS:
            if pat.search(text):
                r.injection_detected.append(pat.pattern)
        
        # Detección de texto ofuscado
        if detect_obfuscation(text):
            r.injection_detected.append("obfuscation")
            r.risk_level = "high"
        
        # Detección de texto codificado
        if detect_encoded_text(text):
            r.injection_detected.append("encoded_injection")
            r.risk_level = "critical"
        
        # Detección de intento de leak de system prompt
        if detect_system_prompt_leak(text):
            r.injection_detected.append("prompt_leak")
            r.risk_level = "critical"
        
        # Decisión final
        if r.injection_detected:
            r.blocked = True
            if r.risk_level == "critical":
                r.block_reason = "Actividad maliciosa detectada. Esta acción ha sido registrada."
            elif r.risk_level == "high":
                r.block_reason = "Intento de manipulación detectado. Tu mensaje ha sido bloqueado."
            else:
                r.block_reason = "Posible intento de inyección de prompt detectado."
        
        return r

    def mask_pii(self, text: str) -> str:
        t = text
        for vals in self.pii_detected.values():
            for v in vals:
                if "@" in v:
                    local, domain = v.split("@", 1)
                    t = t.replace(v, local[0] + "***@" + domain)
                elif len(v) > 6:
                    t = t.replace(v, v[:2] + "***" + v[-2:])
                else:
                    t = t.replace(v, "[REDACTADO]")
        return t

# ──────────────────────────────────────────────
# (1.5) EMAIL
# ──────────────────────────────────────────────
def enviar_mail(destinatario: str, cuerpo: str) -> bool:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.utils import formatdate, make_msgid
    if not REMITE or not REMPASS:
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = f"MeliExpert <{REMITE}>"
    msg["To"] = destinatario
    msg["Subject"] = "Informacion que me pediste sobre productos"
    msg["Reply-To"] = REMITE
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()

    import html as html_mod
    contenido_seguro = html_mod.escape(cuerpo).replace("\n", "<br>")

    tiene_precios = any(c in cuerpo for c in ["$", "CLP", "precio", "oferta"])
    if tiene_precios:
        contenido_html = _formatear_productos_html(cuerpo)
    else:
        contenido_html = f"""
            <td style="padding:20px 30px;color:#333333;font-size:15px;line-height:1.7;">
                {contenido_seguro}
            </td>"""

    # ── Texto plano largo (anti-spam) ──
    texto_plano = f"""Hola, como estas?

Te escribo para compartirte la informacion que me pediste sobre productos de Mercado Libre. Hice una busqueda y encontre varias opciones que pueden interesarte.

A continuacion te dejo el detalle con los precios y especificaciones de cada producto para que puedas comparar:

{cuerpo}

Todos los precios estan en pesos chilenos y pueden cambiar segun las promociones del momento. Te recomiendo entrar a mercadolibre.cl para verificar los precios actuales y disponibilidad antes de comprar.

Si necesitas mas informacion sobre algun producto, quieres ver opciones diferentes, o tienes alguna otra consulta, escribeme por el chat y te ayudo sin problema.

Tambien puedo enviarte esta informacion por correo si la necesitas guardar para despues.

Un saludo,
Tu asistente de Mercado Libre

PD: Este correo se envio automaticamente porque solicitaste informacion por el chat. Si no pediste esto, simplemente ignora el mensaje."""

    # ── HTML con diseno Mercado Libre ──
    cuerpo_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background-color:#EEEEEE;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#EEEEEE;padding:20px 0;">
        <tr>
            <td align="center">
                <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background-color:#FFFFFF;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">

                    <!-- HEADER -->
                    <tr>
                        <td style="background-color:#FFE600;padding:24px 30px;text-align:center;">
                            <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                                <tr>
                                    <td align="left" style="font-size:28px;font-weight:700;color:#2D3277;">
                                        &#128722; MeliExpert
                                    </td>
                                    <td align="right" style="font-size:12px;color:#555555;">
                                        Asistente IA
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- BARRA AZUL -->
                    <tr>
                        <td style="background-color:#3483FA;padding:10px 30px;">
                            <p style="margin:0;color:#FFFFFF;font-size:13px;font-weight:600;text-align:center;">
                                Informacion personalizada para ti
                            </p>
                        </td>
                    </tr>

                    <!-- CONTENIDO -->
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                        {contenido_html}
                    </table>

                    <!-- SEPARADOR -->
                    <tr>
                        <td style="padding:0 30px;">
                            <hr style="border:none;border-top:1px solid #EEEEEE;margin:0;">
                        </td>
                    </tr>

                    <!-- FOOTER -->
                    <tr>
                        <td style="padding:20px 30px 30px 30px;text-align:center;">
                            <p style="margin:0 0 8px 0;font-size:12px;color:#999999;">
                                Enviado por <strong style="color:#3483FA;">MeliExpert</strong> &mdash; Asistente de Mercado Libre
                            </p>
                            <p style="margin:0 0 12px 0;font-size:11px;color:#BBBBBB;">
                                Si necesitas mas ayuda, escribeme por el chat.
                            </p>
                        </td>
                    </tr>

                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""

    msg.attach(MIMEText(texto_plano, "plain", "utf-8"))
    msg.attach(MIMEText(cuerpo_html, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as servidor:
            servidor.login(REMITE, REMPASS)
            servidor.sendmail(REMITE, destinatario, msg.as_string())
        return True
    except Exception:
        return False


def _formatear_productos_html(texto: str) -> str:
    """Formatea el texto como lista de productos con estilo ML."""
    import html as html_mod
    lineas = texto.strip().split("\n")
    html_productos = []

    for linea in lineas:
        linea = linea.strip()
        if not linea:
            continue
        linea_segura = html_mod.escape(linea)

        if any(c in linea for c in ["$", "CLP", "precio"]):
            html_productos.append(f"""
            <tr>
                <td style="padding:6px 30px;">
                    <div style="background:#FFF8E1;border-left:3px solid #FFE600;padding:8px 14px;border-radius:4px;">
                        <span style="color:#333;font-size:14px;">{linea_segura}</span>
                    </div>
                </td>
            </tr>""")
        elif linea.startswith(("-", "*", "1", "2", "3", "4", "5", "6", "7", "8", "9")) or linea[0:1].isupper():
            html_productos.append(f"""
            <tr>
                <td style="padding:4px 30px 0 30px;">
                    <div style="background:#F5F5F5;border-radius:6px;padding:12px 16px;border:1px solid #EEEEEE;">
                        <span style="color:#333;font-size:14px;line-height:1.5;">{linea_segura}</span>
                    </div>
                </td>
            </tr>""")
        else:
            html_productos.append(f"""
            <tr>
                <td style="padding:4px 30px;">
                    <p style="margin:0;color:#555;font-size:14px;line-height:1.6;">{linea_segura}</p>
                </td>
            </tr>""")

    if not html_productos:
        return """
            <td style="padding:20px 30px;color:#333;font-size:15px;line-height:1.7;">
                No se encontraron productos para mostrar.
            </td>"""

    return "\n".join(html_productos)

# ──────────────────────────────────────────────
# (2) RAG — base de conocimiento vectorial
# ──────────────────────────────────────────────
@st.cache_resource
def build_vectorstore():
    FAISS_INDEX_PATH = Path(__file__).parent / "data" / "faiss_index"
    if FAISS_INDEX_PATH.exists():
        try:
            vs = FAISS.load_local(str(FAISS_INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
            return vs
        except Exception:
            pass
    if not KB_PATH.exists():
        st.warning(f"No se encontró {KB_PATH}. RAG desactivado.")
        return None
    docs = json.loads(KB_PATH.read_text(encoding="utf-8"))
    texts = [f"{d['title']}\n{d['content']}" for d in docs]
    metadatas = [{"topic": d["topic"], "id": d["id"]} for d in docs]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks, meta_chunks = [], []
    for i, t in enumerate(texts):
        cs = splitter.split_text(t)
        chunks.extend(cs)
        meta_chunks.extend([metadatas[i]] * len(cs))
    if not chunks:
        return None
    try:
        vs = FAISS.from_texts(chunks, embeddings, metadatas=meta_chunks)
        vs.save_local(str(FAISS_INDEX_PATH))
        return vs
    except Exception as e:
        if "Too many requests" in str(e) or "429" in str(e) or "RateLimit" in str(e):
            pass
        else:
            st.warning(f"No se pudo crear el índice de búsqueda. RAG desactivado.")
        return None

def retrieve_context(query: str, k: int = 3) -> List[Dict]:
    vs = build_vectorstore()
    if vs is None:
        return keyword_search(query, k)
    try:
        docs = vs.similarity_search_with_score(query, k=k)
        return [{"content": d[0].page_content, "topic": d[0].metadata.get("topic", ""), "score": float(d[1])} for d in docs]
    except Exception:
        return keyword_search(query, k)

def keyword_search(query: str, k: int = 3) -> List[Dict]:
    """Búsqueda por palabras clave cuando el vectorstore no está disponible."""
    if not KB_PATH.exists():
        return []
    try:
        docs = json.loads(KB_PATH.read_text(encoding="utf-8"))
        query_words = set(query.lower().split())
        scored = []
        for d in docs:
            text = (d["title"] + " " + d["content"]).lower()
            score = sum(1 for w in query_words if w in text)
            if score > 0:
                scored.append({"content": d["title"] + "\n" + d["content"], "topic": d["topic"], "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]
    except Exception:
        return []

# ──────────────────────────────────────────────
# (3) DESCOMPOSICIÓN DE TAREAS
# ──────────────────────────────────────────────
def decompose_query(query: str) -> List[Dict]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un planificador de tareas. Descompón la consulta del usuario en subtareas simples y secuenciales.

Para cada subtarea responde ÚNICAMENTE un objeto JSON con:
- "id": número entero (1,2,3...)
- "tarea": descripción corta
- "tipo": "informacion" | "recomendacion" | "gestion" | "comparacion" | "seguimiento"
- "depende_de": lista de IDs de los que depende (ej: [1] o [])

Reglas:
- Máximo 4 subtareas.
- Si es simple, una sola subtarea.
- Responde SOLO el JSON array, sin explicación ni markdown."""),
        ("human", "{query}")
    ])
    chain = prompt | llm
    try:
        raw = chain.invoke({"query": query})
        content = raw.content if hasattr(raw, "content") else str(raw)
        content = re.sub(r'```(?:json)?\s*', '', content).strip()
        tasks = json.loads(content)
        return tasks if isinstance(tasks, list) else [tasks]
    except Exception:
        for token in TOKENS:
            try:
                llm_fb = ChatOpenAI(
                    base_url="https://models.inference.ai.azure.com",
                    api_key=token,
                    model="gpt-4o",
                    temperature=0.1,
                    timeout=30,
                )
                chain_fb = prompt | llm_fb
                raw = chain_fb.invoke({"query": query})
                content = raw.content if hasattr(raw, "content") else str(raw)
                content = re.sub(r'```(?:json)?\s*', '', content).strip()
                tasks = json.loads(content)
                return tasks if isinstance(tasks, list) else [tasks]
            except Exception:
                continue
        return [{"id": 1, "tarea": query, "tipo": "informacion", "depende_de": []}]

# ──────────────────────────────────────────────
# (4) WORKFLOWS — procesos multi-paso
# ──────────────────────────────────────────────
class WorkflowState(Enum):
    INICIO = "inicio"
    RECOLECTANDO = "recolectando_info"
    ESPERANDO_CONFIRMACION = "esperando_confirmacion"
    EJECUTANDO = "ejecutando"
    COMPLETADO = "completado"
    ERROR = "error"

WORKFLOWS = {
    "devolucion": {
        "name": "Devolución",
        "steps": [
            {"id": 1, "desc": "Identificar producto y motivo", "action": "preguntar_producto"},
            {"id": 2, "desc": "Verificar elegibilidad", "action": "verificar_elegibilidad"},
            {"id": 3, "desc": "Generar etiqueta de devolución", "action": "generar_etiqueta"},
            {"id": 4, "desc": "Instrucciones de envío", "action": "dar_instrucciones"},
            {"id": 5, "desc": "Confirmar recepción y reembolso", "action": "procesar_reembolso"},
        ]
    },
    "reclamo": {
        "name": "Reclamo",
        "steps": [
            {"id": 1, "desc": "Recibir descripción del problema", "action": "describir_problema"},
            {"id": 2, "desc": "Solicitar evidencia", "action": "solicitar_evidencia"},
            {"id": 3, "desc": "Evaluar según política", "action": "evaluar"},
            {"id": 4, "desc": "Definir resolución", "action": "resolver"},
        ]
    },
    "compra": {
        "name": "Asesoría de Compra",
        "steps": [
            {"id": 1, "desc": "Entender necesidad del usuario", "action": "entender_necesidad"},
            {"id": 2, "desc": "Buscar y comparar opciones", "action": "buscar_opciones"},
            {"id": 3, "desc": "Recomendar producto", "action": "recomendar"},
            {"id": 4, "desc": "Asistencia en checkout", "action": "asistir_compra"},
        ]
    }
}

class WorkflowEngine:
    def __init__(self):
        self.active_workflows: Dict[str, Dict] = {}

    def start(self, session_id: str, workflow_type: str) -> Optional[Dict]:
        if workflow_type not in WORKFLOWS:
            return None
        wf = WORKFLOWS[workflow_type]
        instance = {
            "type": workflow_type,
            "name": wf["name"],
            "current_step": 0,
            "steps": wf["steps"],
            "state": WorkflowState.INICIO,
            "data": {},
            "created_at": datetime.now().isoformat(),
        }
        self.active_workflows[session_id] = instance
        return instance

    def get(self, session_id: str) -> Optional[Dict]:
        return self.active_workflows.get(session_id)

    def next_step(self, session_id: str) -> Optional[Dict]:
        wf = self.active_workflows.get(session_id)
        if not wf:
            return None
        wf["current_step"] += 1
        if wf["current_step"] >= len(wf["steps"]):
            wf["state"] = WorkflowState.COMPLETADO
            return None
        wf["state"] = WorkflowState.EJECUTANDO
        return wf["steps"][wf["current_step"]]

    def complete(self, session_id: str):
        wf = self.active_workflows.get(session_id)
        if wf:
            wf["state"] = WorkflowState.COMPLETADO

    def cancel(self, session_id: str):
        wf = self.active_workflows.get(session_id)
        if wf:
            wf["state"] = WorkflowState.ERROR

# ──────────────────────────────────────────────
# (5-6) ORQUESTACIÓN MULTI-AGENTE + ASIGNACIÓN
# ──────────────────────────────────────────────
class QueryType(Enum):
    VENTAS = "ventas"
    SOPORTE = "soporte"
    FACTURACION = "facturacion"
    ENVIOS = "envios"
    SEGURIDAD = "seguridad"
    GENERAL = "general"

QUERY_KEYWORDS = {
    QueryType.VENTAS: ["comprar", "vender", "precio", "producto", "oferta", "mouse", "ratón", "recomienda", "cuál", "mejor", "presupuesto", "laptop", "notebook", "gamer", "periférico"],
    QueryType.SOPORTE: ["no funciona", "error", "bug", "falla", "técnico", "configurar", "cómo", "app", "web", "plataforma", "sistema", "lento", "trabado"],
    QueryType.FACTURACION: ["factura", "facturación", "pago", "tarjeta", "reembolso", "devolución", "cuota", "impuesto", "iva", "cobro", "cobraron"],
    QueryType.ENVIOS: ["envío", "envios", "seguimiento", "tracking", "paquete", "entrega", "correo", "domicilio", "full", "estado", "llegó"],
    QueryType.SEGURIDAD: ["seguridad", "contraseña", "hack", "fraude", "phishing", "cuenta", "robada", "verificación", "clave"],
}

class AgentOrchestrator:
    def __init__(self):
        self.sessions: Dict[str, InMemoryChatMessageHistory] = {}

    def get_history(self, sid: str) -> InMemoryChatMessageHistory:
        if sid not in self.sessions:
            self.sessions[sid] = InMemoryChatMessageHistory()
        return self.sessions[sid]

    def classify(self, text: str) -> QueryType:
        tl = text.lower()
        scores = {}
        for qt, kws in QUERY_KEYWORDS.items():
            scores[qt] = sum(1 for kw in kws if kw in tl)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else QueryType.GENERAL

    def priority_score(self, text: str) -> int:
        """Asignación de recursos: prioridad 0-10"""
        tl = text.lower()
        score = 5
        urgent = ["urgente", "rápido", "ya", "ahora", "problema grave", "emergencia", "mañana", "ayer"]
        angry = ["queja", "mal servicio", "estafa", "robo", "enojado", "pésimo", "horrible"]
        for w in urgent:
            if w in tl: score += 1
        for w in angry:
            if w in tl: score += 2
        return min(score, 10)

    def resolve_conflict(self, session_id: str, text: str, priority: int) -> str:
        """Resolución de conflictos: decide si escalar"""
        tl = text.lower()
        escalation_keywords = ["abogado", "demanda", "denuncia", "defensa al consumidor", "coprec", "operador", "supervisor"]
        for kw in escalation_keywords:
            if kw in tl:
                return "escalar"
        if "reembolso" in tl and "no" in tl:
            return "escalar"
        if priority >= 9:
            return "escalar"
        return "resolver"

AGENT_PROMPTS = {
    QueryType.VENTAS: ChatPromptTemplate.from_messages([
        ("system", """Eres un asesor de ventas experto de Mercado Libre especializado en tecnología. Ayudas al usuario a encontrar el mejor producto según su necesidad y presupuesto.

FORMATO OBLIGATORIO para ofertas (responde EXACTAMENTE asi):
1. **Nombre del Producto**
- **Precio:** $XX.XXX CLP
- **Especificaciones:**
- Procesador...
- Memoria...
- Pantalla...
- Otros datos relevantes
- **Beneficios:** envio gratis, garantia, etc

2. **Siguiente Producto**
- **Precio:** $XX.XXX CLP
- **Especificaciones:**
- ...

REGLAS:
- Numerados del 1 al 5 como maximo
- Cada producto tiene nombre, precio y especificaciones en viñetas
- Precios en CLP con formato: $XX.XXX
- Sin links ni URLs
- Si el usuario te pide enviar por correo, el sistema lo hace automatico"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]),
    QueryType.SOPORTE: ChatPromptTemplate.from_messages([
        ("system", "Eres un agente de soporte técnico de Mercado Libre. Ayudas a resolver problemas técnicos con la plataforma, la app o productos comprados. Das pasos claros y numerados. Usas un tono neutro y profesional."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]),
    QueryType.FACTURACION: ChatPromptTemplate.from_messages([
        ("system", "Eres un agente de facturación de Mercado Libre. Ayudas con pagos, facturas, cuotas, impuestos y reembolsos. Explicas plazos y montos claramente en CLP. Usas un tono neutro."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]),
    QueryType.ENVIOS: ChatPromptTemplate.from_messages([
        ("system", "Eres un agente de logística de Mercado Libre. Ayudas con seguimiento de envíos, plazos de entrega, Envío Full y direcciones. Das información concreta de tiempos. Usas un tono neutro."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]),
    QueryType.SEGURIDAD: ChatPromptTemplate.from_messages([
        ("system", "Eres un agente de seguridad de Mercado Libre. Ayudas con contraseñas, verificación en dos pasos, detección de fraudes y protección de cuenta. Eres serio y profesional. Usas un tono de precaución."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]),
    QueryType.GENERAL: ChatPromptTemplate.from_messages([
        ("system", """Eres MeliExpert, el asistente principal de Mercado Libre. Respondes dudas generales con un tono amigable y profesional.
Si la consulta es compleja, sugieres hablar con el agente especializado correspondiente.
Usas un español neutro y claro.
Si el usuario te pide enviar información por correo, incluye todos los detalles en tu respuesta y yo me encargaré del envío."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]),
}

orchestrator = AgentOrchestrator()
workflow_engine = WorkflowEngine()

# ──────────────────────────────────────────────
# (7) TRAZABILIDAD Y MÉTRICAS
# ──────────────────────────────────────────────
def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interacciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            query_type TEXT,
            timestamp TEXT,
            query TEXT,
            response TEXT,
            priority INTEGER,
            security_blocked INTEGER,
            retrieved_docs INTEGER,
            workflow_type TEXT,
            resolved INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metricas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT,
            total_interacciones INTEGER,
            avg_prioridad REAL,
            bloqueos_seguridad INTEGER,
            escalados INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_interaction(session_id: str, query_type: str, query: str, response: str,
                    priority: int, security_blocked: bool, retrieved_docs: int,
                    workflow_type: str = "", resolved: bool = True):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("""
            INSERT INTO interacciones
            (session_id, query_type, timestamp, query, response, priority,
             security_blocked, retrieved_docs, workflow_type, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, query_type, datetime.now().isoformat(), query[:500],
              response[:500], priority, int(security_blocked), retrieved_docs,
              workflow_type, int(resolved)))
        conn.commit()
        conn.close()
    except Exception:
        pass

def get_metrics(days: int = 7) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("""
            SELECT date(timestamp) as fecha,
                   COUNT(*) as total,
                   ROUND(AVG(priority), 1) as avg_prioridad,
                   SUM(security_blocked) as bloqueos,
                   COUNT(CASE WHEN query_type = 'escalado' THEN 1 END) as escalados
            FROM interacciones
            WHERE timestamp >= datetime('now', ?)
            GROUP BY fecha ORDER BY fecha DESC
        """, conn, params=[f'-{days} days'])
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

init_db()

# ──────────────────────────────────────────────
# (8) LANGGRAPH — grafo de agentes
# ──────────────────────────────────────────────
class AgentState(TypedDict):
    messages: List
    session_id: str
    query_type: QueryType
    priority: int
    security: Optional[SecurityReport]
    context_docs: List[Dict]
    workflow: Optional[Dict]
    resolution: str
    resolved: bool

# ── Token rotation helper ──
def _try_tokens(prompt_text: str, system_msg: str = "Eres MeliExpert de Mercado Libre. Responde en español neutro con precios en CLP.") -> Optional[str]:
    """Prueba cada token con GPT-4o hasta que uno funcione."""
    for token in TOKENS:
        try:
            llm_tmp = ChatOpenAI(
                base_url="https://models.inference.ai.azure.com",
                api_key=token,
                model="gpt-4o",
                temperature=0.1,
                timeout=60,
            )
            prompt_tmp = ChatPromptTemplate.from_messages([
                ("system", system_msg),
                ("human", "{input}"),
            ])
            chain_tmp = prompt_tmp | llm_tmp
            result = chain_tmp.invoke({"input": prompt_text})
            return result.content if hasattr(result, "content") else str(result)
        except Exception:
            continue
    return None

def node_security(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
    text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    report = SecurityReport.scan(text)
    state["security"] = report
    if report.blocked:
        state["messages"].append(AIMessage(
            content=f"⛔ {report.block_reason}. Tu mensaje ha sido bloqueado por políticas de seguridad. Por favor reformula tu consulta."
        ))
    return state

def node_classify(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
    text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    state["query_type"] = orchestrator.classify(text)
    state["priority"] = orchestrator.priority_score(text)
    state["context_docs"] = retrieve_context(text)
    state["resolution"] = orchestrator.resolve_conflict(
        state["session_id"], text, state["priority"]
    )
    return state

def _invoke_with_fallback(chain, config, input_data, model_label="gpt-4o"):
    """Intenta con GPT-4o rotando tokens si hay rate limit."""
    text = input_data.get("input", input_data.get("query", "")) if isinstance(input_data, dict) else str(input_data)
    for token in TOKENS:
        try:
            llm_temp = ChatOpenAI(
                base_url="https://models.inference.ai.azure.com",
                api_key=token,
                model="gpt-4o",
                temperature=0.1,
                streaming=True,
                timeout=30,
            )
            chain_temp = chain | llm_temp
            return chain_temp.invoke(input_data, config=config)
        except Exception as e:
            if "429" in str(e) or "RateLimitReached" in str(e):
                continue
            raise
    result = _try_tokens(text)
    if result:
        return AIMessage(content=result)
    return AIMessage(content="❌ **Límite alcanzado en todas las cuentas.** Probá con otro token o esperá a mañana.")

def build_agent_node(qt: QueryType):
    def node(state: AgentState) -> AgentState:
        if state["security"] and state["security"].blocked:
            state["resolved"] = True
            return state
        text = state["messages"][-1].content if state["messages"] else ""
        masked = state["security"].mask_pii(text) if state["security"] else text

        # Detectar email del usuario ANTES de llamar al LLM
        correo_usuario = None
        tl = text.lower()
        if "correo" in tl or "mail" in tl or "email" in tl or "enviar" in tl or "@" in text:
            for p in text.split():
                if "@" in p:
                    correo_usuario = p.strip(".,!?()\"'<>")
                    break

        # Si hay email, modificar la consulta para que el LLM sepa que ya se envía
        if correo_usuario:
            masked += f"\n\n(Nota IMPORTANTE: esta respuesta irá por correo a {correo_usuario}. Responde ÚNICAMENTE con las ofertas o productos solicitados. Nada de saludos, consejos, despedidas ni explicaciones. Solo los datos: nombre, precio CLP y especificaciones básicas. Sin texto adicional.)"

        context = "\n\n".join(
            f"📄 {d['topic']}: {d['content']}"
            for d in state.get("context_docs", [])
        )
        if context:
            agent_input = f"Contexto:\n{context}\n\nConsulta: {masked}"
        else:
            agent_input = masked
        prompt = AGENT_PROMPTS[qt]
        chain = prompt | llm
        history = orchestrator.get_history(state["session_id"])
        config = {"configurable": {"session_id": state["session_id"]}}
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda sid: history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        result = _invoke_with_fallback(chain_with_history, config, {"input": agent_input})
        respuesta = result.content if hasattr(result, "content") else str(result)

        # Enviar por correo si el usuario lo pidió
        if correo_usuario:
            exito = enviar_mail(correo_usuario, respuesta)
            if exito:
                respuesta += f"\n\n✉️ También te envié esta información por correo a **{correo_usuario}**."

        state["messages"].append(AIMessage(content=respuesta))
        state["resolved"] = True
        return state
    return node

def node_escalate(state: AgentState) -> AgentState:
    state["messages"].append(AIMessage(
        content="🔄 Este caso requiere atención especializada. He generado un ticket de escalamiento. Un supervisor humano lo revisará en las próximas 24 horas hábiles. Tu número de ticket es: MELI-" + hashlib.md5(state["session_id"].encode()).hexdigest()[:8].upper()
    ))
    state["resolution"] = "escalado"
    state["resolved"] = True
    return state

def router_agent(state: AgentState) -> Literal["ventas", "soporte", "facturacion", "envios", "seguridad", "general", "escalar"]:
    if state["security"] and state["security"].blocked:
        return "general"
    if state["resolution"] == "escalar":
        return "escalar"
    qt = state["query_type"]
    return qt.value

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("security", node_security)
    g.add_node("classify", node_classify)
    for qt in QueryType:
        g.add_node(qt.value, build_agent_node(qt))
    g.add_node("escalar", node_escalate)
    g.set_entry_point("security")
    g.add_edge("security", "classify")
    g.add_conditional_edges("classify", router_agent)
    for qt in QueryType:
        g.add_edge(qt.value, END)
    g.add_edge("escalar", END)
    return g.compile()

agent_graph = build_graph()

# ──────────────────────────────────────────────
# UI — STREAMLIT (Estilo Mercado Libre)
# ──────────────────────────────────────────────
st.set_page_config(page_title="MeliExpert", page_icon="🛒", layout="wide")

# ── Forzar tema claro de Streamlit ──
st._config.set_option("theme.base", "light")

# ── CSS Mercado Libre (compatible dark mode del navegador) ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Proxima+Nova:wght@400;600;700&display=swap');

    :root {
        --meli-yellow: #FFE600;
        --meli-yellow-dark: #F5E600;
        --meli-blue: #3483FA;
        --meli-blue-dark: #2968C8;
        --meli-navy: #2D3277;
        --meli-bg: #EEEEEE;
        --meli-card: #FFFFFF;
        --meli-text: #333333;
        --meli-text-secondary: #666666;
        --meli-border: #DDDDDD;
        --meli-success: #00A650;
        --meli-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }

    /* ══════════════════════════════════════════
       OVERRIDE TOTAL: Forzar modo claro en
       TODOS los elementos de Streamlit
       ══════════════════════════════════════════ */

    /* ── App completa ── */
    .stApp, .stApp[data-theme="dark"],
    html[data-theme="dark"], body,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    .main .block-container {
        background-color: var(--meli-bg) !important;
        color: var(--meli-text) !important;
    }

    .stApp {
        font-family: 'Proxima Nova', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    .stApp > header {
        background-color: var(--meli-yellow) !important;
    }

    /* ── Main container ── */
    .block-container {
        padding-top: 1rem !important;
        max-width: 1200px !important;
        background-color: var(--meli-bg) !important;
        color: var(--meli-text) !important;
    }

    /* ── Forzar colores de texto en TODOS los elementos Streamlit ── */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    .stMarkdown h5, .stMarkdown h6,
    p, li, span, label, div {
        color: var(--meli-text) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--meli-navy) !important;
    }

    /* ── Top navigation bar ── */
    .meli-navbar {
        background-color: #FFE600 !important;
        padding: 16px 30px !important;
        margin: -1rem -1rem 1rem -1rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        border-bottom: 1px solid rgba(0,0,0,0.05) !important;
    }

    .meli-navbar-brand {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }

    .meli-navbar-brand h1 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #2D3277 !important;
        margin: 0 !important;
    }

    .meli-logo {
        font-size: 1.8rem !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background: var(--meli-card) !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 4px 4px 0 4px !important;
        border-bottom: 2px solid var(--meli-border) !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        color: var(--meli-text-secondary) !important;
        background: transparent !important;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--meli-blue) !important;
        border-bottom: 3px solid var(--meli-blue) !important;
        background: var(--meli-card) !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        background: transparent !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: var(--meli-blue) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 24px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--meli-shadow) !important;
    }

    .stButton > button:hover {
        background-color: var(--meli-blue-dark) !important;
        box-shadow: 0 2px 8px rgba(52,131,250,0.3) !important;
        transform: translateY(-1px) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        border-radius: 8px !important;
        padding: 16px 20px !important;
        margin: 8px 0 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
        border: 1px solid #EEEEEE !important;
        background: #FFFFFF !important;
        color: #333333 !important;
    }

    [data-testid="stChatMessage"][data-testid-type="user"],
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #F5F5F5 !important;
        border-left: 3px solid #3483FA !important;
        color: #333333 !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
    [data-testid="stChatMessage"]:has(avatar="assistant") {
        background: #FFFFFF !important;
        border-left: 3px solid #FFE600 !important;
        color: #333333 !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em {
        color: #333333 !important;
        line-height: 1.6 !important;
    }

    [data-testid="stChatMessage"] strong {
        color: #2D3277 !important;
    }

    /* ── Chat input ── */
    .stChatInput,
    .stChatInput > div,
    .stChatInput > div > div,
    .stChatInput > div > div > div,
    .stChatInput > div > div > div > div,
    .stChatInput textarea,
    .stChatInput [data-testid="stChatInput"],
    .stChatInput [data-testid="stChatInput"] > div,
    .stChatInput [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] > div > div > div {
        background-color: var(--meli-card) !important;
        border-radius: 12px !important;
    }

    .stChatInput,
    .stChatInput > div,
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div {
        border: 2px solid var(--meli-border) !important;
        box-shadow: var(--meli-shadow) !important;
        transition: border-color 0.2s ease !important;
    }

    .stChatInput > div:focus-within,
    [data-testid="stChatInput"] > div:focus-within {
        border-color: var(--meli-blue) !important;
        box-shadow: 0 0 0 3px rgba(52,131,250,0.15) !important;
    }

    .stChatInput textarea,
    .stChatInput input,
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        color: var(--meli-text) !important;
        background-color: var(--meli-card) !important;
    }

    .stChatInput textarea::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--meli-text-secondary) !important;
    }

    /* ── Text input ── */
    .stTextInput > div > div > input {
        background-color: var(--meli-card) !important;
        color: var(--meli-text) !important;
        border: 1px solid var(--meli-border) !important;
        border-radius: 6px !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--meli-blue) !important;
        box-shadow: 0 0 0 2px rgba(52,131,250,0.15) !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background-color: var(--meli-card) !important;
        color: var(--meli-text) !important;
    }

    .stSelectbox label {
        color: var(--meli-text) !important;
    }

    /* ── Radio / Checkbox ── */
    .stRadio label, .stCheckbox label {
        color: var(--meli-text) !important;
    }

    .stRadio > div > div > label > span,
    .stCheckbox > div > div > label > span {
        color: var(--meli-text) !important;
    }

    /* ── Info boxes ── */
    .stAlert {
        border-radius: 8px !important;
        border-left-width: 4px !important;
        color: var(--meli-text) !important;
    }

    div[data-testid="stInfo"] {
        background-color: #E3F2FD !important;
        border-color: var(--meli-blue) !important;
        color: var(--meli-text) !important;
    }

    div[data-testid="stInfo"] p,
    div[data-testid="stInfo"] span {
        color: var(--meli-text) !important;
    }

    div[data-testid="stSuccess"] {
        background-color: #E8F5E9 !important;
        border-color: var(--meli-success) !important;
        color: var(--meli-text) !important;
    }

    div[data-testid="stSuccess"] p,
    div[data-testid="stSuccess"] span {
        color: var(--meli-text) !important;
    }

    div[data-testid="stWarning"] {
        background-color: #FFF8E1 !important;
        border-color: var(--meli-yellow) !important;
        color: var(--meli-text) !important;
    }

    div[data-testid="stWarning"] p,
    div[data-testid="stWarning"] span {
        color: var(--meli-text) !important;
    }

    div[data-testid="stError"] {
        background-color: #FFEBEE !important;
        border-color: #D32F2F !important;
        color: var(--meli-text) !important;
    }

    div[data-testid="stError"] p,
    div[data-testid="stError"] span {
        color: var(--meli-text) !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader,
    .streamlit-expanderHeader p,
    .streamlit-expanderHeader span {
        font-weight: 600 !important;
        color: var(--meli-navy) !important;
    }

    .streamlit-expanderContent {
        background: var(--meli-card) !important;
        color: var(--meli-text) !important;
    }

    .streamlit-expanderContent p,
    .streamlit-expanderContent span,
    .streamlit-expanderContent li {
        color: var(--meli-text) !important;
    }

    /* ── Metric cards ── */
    .meli-metric-card {
        background: var(--meli-card);
        border-radius: 10px;
        padding: 16px;
        box-shadow: var(--meli-shadow);
        border: 1px solid var(--meli-border);
        text-align: center;
    }

    .meli-metric-card h3 {
        color: var(--meli-blue) !important;
        font-size: 2rem;
        margin: 0;
    }

    .meli-metric-card p {
        color: var(--meli-text-secondary) !important;
        font-size: 0.85rem;
        margin: 4px 0 0 0;
    }

    /* ── Feature cards ── */
    .meli-feature-card {
        background: var(--meli-card);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
        border-left: 3px solid var(--meli-success);
        box-shadow: var(--meli-shadow);
    }

    .meli-feature-card strong,
    .meli-feature-card span {
        color: var(--meli-text) !important;
    }

    /* ── Section headers ── */
    .meli-section-title {
        color: #2D3277 !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding-bottom: 8px !important;
        border-bottom: 2px solid #FFE600 !important;
        margin-bottom: 12px !important;
    }

    /* ── Markdown content ── */
    .stMarkdown p {
        line-height: 1.7 !important;
    }

    .stMarkdown strong {
        color: #2D3277 !important;
    }

    /* ── Code blocks ── */
    .stCodeBlock, .stCodeBlock pre, .stCodeBlock code {
        background-color: #F5F5F5 !important;
        color: var(--meli-text) !important;
        border-radius: 8px !important;
        border: 1px solid var(--meli-border) !important;
    }

    /* ── DataFrame ── */
    .stDataFrame {
        background: var(--meli-card) !important;
    }

    .stDataFrame th, .stDataFrame td {
        color: var(--meli-text) !important;
    }

    /* ── Caption ── */
    .stCaption, .stCaption p, .stCaption span,
    small, .small {
        color: var(--meli-text-secondary) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }

    ::-webkit-scrollbar-thumb {
        background: #bbb;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #888;
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid var(--meli-border) !important;
    }

    /* ── Subheader ── */
    .stSubheader {
        color: var(--meli-navy) !important;
    }

    /* ── Write / markdown containers ── */
    .stMarkdownContainer {
        color: var(--meli-text) !important;
    }

    /* ── Hide Streamlit branding ── */
    footer {display: none !important;}
    #MainMenu {display: none !important;}
    header {display: none !important;}
</style>
""", unsafe_allow_html=True)

# ── Navbar estilo Mercado Libre ──
st.markdown("""
<div class="meli-navbar">
    <div class="meli-navbar-brand">
        <span class="meli-logo">&#128722;</span>
        <h1>MeliExpert</h1>
    </div>
    <div style="font-size:13px;color:#555555;">
        Asistente IA
    </div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
if "processing" not in st.session_state:
    st.session_state.processing = False

sid = st.session_state.session_id

col1, col2 = st.columns([3, 1])

with col2:
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        orchestrator.sessions.pop(sid, None)
        st.rerun()

    mostrar_metrics = st.checkbox("📊 Mostrar métricas", False)

    if mostrar_metrics:
        st.markdown('<div class="meli-section-title">📈 Métricas</div>', unsafe_allow_html=True)
        df_m = get_metrics(7)
        if not df_m.empty:
            st.dataframe(df_m, use_container_width=True, hide_index=True)
        else:
            st.info("Aún no hay métricas registradas.")
    st.markdown('<div class="meli-section-title">🔍 Diagnóstico</div>', unsafe_allow_html=True)
    test_input = st.text_input("Probar clasificación:", key="test_classify", placeholder="Escribe una consulta...")
    if test_input:
        qt = orchestrator.classify(test_input)
        prio = orchestrator.priority_score(test_input)
        res = orchestrator.resolve_conflict("test", test_input, prio)
        sec = SecurityReport.scan(test_input)
        st.markdown(f"""<div class="meli-metric-card">
            <p><strong>Tipo</strong></p>
            <h3 style="font-size:1.2rem;">{qt.value}</h3>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="meli-metric-card">
            <p><strong>Prioridad</strong></p>
            <h3>{prio}/10</h3>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="meli-metric-card">
            <p><strong>Resolución</strong></p>
            <h3 style="font-size:1.2rem;">{res}</h3>
        </div>""", unsafe_allow_html=True)
        if sec.blocked:
            st.error(f"⚠️ Bloqueado: {sec.block_reason}")
        else:
            st.success("✅ Seguridad OK")
        docs = retrieve_context(test_input)
        if docs:
            st.caption(f"📄 {len(docs)} documentos RAG recuperados")
            for d in docs:
                st.caption(f"  → {d['topic']} (score: {d['score']:.2f})")

with col1:
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧩 Workflows", "ℹ️ Ayuda"])

    with tab1:
        st.markdown("""<div style="background-color:#3483FA;padding:10px 20px;border-radius:6px;margin-bottom:16px;">
            <p style="margin:0;color:#FFFFFF;font-size:14px;font-weight:600;text-align:center;">
                Asistente de Mercado Libre — Tu consulta sera atendida por el especialista adecuado
            </p>
        </div>""", unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Escribe tu mensaje aquí..."):
            if st.session_state.processing:
                st.warning("⏳ Espera a que termine el mensaje actual antes de enviar otro.")
            else:
                st.session_state.processing = True
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🧠 Analizando tu consulta..."):
                    state = AgentState(
                    messages=[HumanMessage(content=prompt)],
                    session_id=sid,
                    query_type=QueryType.GENERAL,
                    priority=5,
                    security=None,
                    context_docs=[],
                    workflow=None,
                    resolution="resolver",
                    resolved=False,
                )
                try:
                    final_state = agent_graph.invoke(state)
                    if final_state.get("security") and final_state["security"].blocked:
                        response = final_state["messages"][-1].content
                    else:
                        response = final_state["messages"][-1].content
                    st.markdown(response)
                    qt = final_state.get("query_type", QueryType.GENERAL)
                    prio = final_state.get("priority", 5)
                    sec = final_state.get("security")
                    docs = final_state.get("context_docs", [])
                    res = final_state.get("resolution", "resolver")
                    log_interaction(sid, qt.value, prompt, response, prio,
                                    bool(sec and sec.blocked), len(docs),
                                    "", res != "escalar")
                    with st.expander("🔍 Ver diagnóstico de esta respuesta"):
                        st.markdown(f"""<div class="meli-metric-card">
                            <p><strong>Tipo consulta:</strong> {qt.value} | <strong>Prioridad:</strong> {prio}/10 | <strong>Resolución:</strong> {res}</p>
                            <p><strong>Docs RAG:</strong> {len(docs)} recuperados</p>
                        </div>""", unsafe_allow_html=True)
                        if sec:
                            if sec.blocked:
                                st.error(f"⚠️ Bloqueado por seguridad. PII: {list(sec.pii_detected.keys()) or 'Ninguna'}")
                            else:
                                st.success(f"✅ Seguridad OK. PII: {list(sec.pii_detected.keys()) or 'Ninguna'}")
                        if docs:
                            st.markdown("**📄 Contexto usado:**")
                            for d in docs:
                                st.caption(f"→ {d['topic']} — score: {d['score']:.2f}")
                except Exception as e:
                    err = str(e)
                    if "RateLimitReached" in err or "429" in err:
                        response = None
                        for token in TOKENS:
                            try:
                                llm_fb = ChatOpenAI(
                                    base_url="https://models.inference.ai.azure.com",
                                    api_key=token,
                                    model="gpt-4o",
                                    temperature=0.1,
                                    timeout=60,
                                )
                                prompt_fb = ChatPromptTemplate.from_messages([
                                    ("system", "Eres MeliExpert de Mercado Libre. Responde en español neutro con precios en CLP."),
                                    ("human", "{input}"),
                                ])
                                chain_fb = prompt_fb | llm_fb
                                result_fb = chain_fb.invoke({"input": prompt})
                                response = result_fb.content if hasattr(result_fb, "content") else str(result_fb)
                                st.markdown(response)
                                break
                            except Exception:
                                continue
                        if response is None:
                            response = "❌ **Límite de uso del día alcanzado en todas las cuentas.** Vuelve a intentar mañana o cambiá de token."
                            st.markdown(response)
                    else:
                        response = f"❌ Error procesando tu consulta: {err}"
                        st.markdown(response)
                    log_interaction(sid, "error", prompt, response, 0, False, 0, "", False)
                finally:
                    st.session_state.processing = False
            st.session_state.messages.append({"role": "assistant", "content": response})

    with tab2:
        st.markdown('<div class="meli-section-title">🧩 Workflows Disponibles</div>', unsafe_allow_html=True)
        wf_selected = st.selectbox("Seleccionar workflow", list(WORKFLOWS.keys()),
                                   format_func=lambda x: WORKFLOWS[x]["name"])

        cols_wf = st.columns(3)
        with cols_wf[0]:
            if st.button("🚀 Iniciar", use_container_width=True):
                wf = workflow_engine.start(sid, wf_selected)
                if wf:
                    st.success(f"✅ Workflow '{wf['name']}' iniciado!")
                    steps_text = "\n".join(f"  {s['id']}. {s['desc']}" for s in wf["steps"])
                    st.code(f"Pasos:\n{steps_text}", language=None)
                    st.info(f"👉 Paso actual: {wf['steps'][0]['desc']}")
                else:
                    st.error("Workflow no encontrado")

        with cols_wf[1]:
            if st.button("⏭️ Siguiente paso", use_container_width=True):
                step = workflow_engine.next_step(sid)
                if step:
                    st.info(f"👉 Siguiente: **{step['desc']}**")
                else:
                    wf = workflow_engine.get(sid)
                    if wf and wf["state"] == WorkflowState.COMPLETADO:
                        st.success("✅ ¡Workflow completado!")
                    else:
                        st.warning("No hay workflow activo")

        with cols_wf[2]:
            if st.button("❌ Cancelar", use_container_width=True):
                workflow_engine.cancel(sid)
                st.warning("Workflow cancelado")

        st.markdown("---")
        if st.button("🧩 Descomponer última consulta", use_container_width=True):
            if st.session_state.messages:
                last_q = st.session_state.messages[-1]["content"]
                if st.session_state.messages[-1]["role"] == "user":
                    tasks = decompose_query(last_q)
                    st.markdown("**📋 Tareas descompuestas:**")
                    for t in tasks:
                        st.markdown(f"""<div class="meli-feature-card">
                            <strong>#{t.get('id', '?')}</strong> {t.get('tarea', '?')} <em>({t.get('tipo', '?')})</em>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("Envía un mensaje primero para descomponerlo")
            else:
                st.info("No hay mensajes aún")

    with tab3:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#FFE600,#FFF159);padding:20px;border-radius:12px;margin-bottom:16px;">
            <h2 style="color:#2D3277;margin:0;">🛒 MeliExpert — Sistema Completo</h2>
            <p style="color:#555;margin:4px 0 0 0;">Asistente inteligente de Mercado Libre con 8 conceptos avanzados integrados</p>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""<div class="meli-metric-card" style="text-align:left;">
                <h3 style="font-size:1.1rem;">🔒 Seguridad</h3>
                <p>Detección de PII y prevención de inyección de prompts</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("""<div class="meli-metric-card" style="text-align:left;">
                <h3 style="font-size:1.1rem;">📚 RAG</h3>
                <p>Recuperación aumentada con FAISS y base de conocimiento ML</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("""<div class="meli-metric-card" style="text-align:left;">
                <h3 style="font-size:1.1rem;">🧩 Descomposición</h3>
                <p>LLM divide consultas complejas en subtareas manejables</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("""<div class="meli-metric-card" style="text-align:left;">
                <h3 style="font-size:1.1rem;">🔄 Workflows</h3>
                <p>Procesos multi-paso: devolución, reclamo, asesoría</p>
            </div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown("""<div class="meli-metric-card" style="text-align:left;">
                <h3 style="font-size:1.1rem;">🤖 Multi-Agente</h3>
                <p>6 agentes especializados con LangGraph</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("""<div class="meli-metric-card" style="text-align:left;">
                <h3 style="font-size:1.1rem;">⚖️ Priorización</h3>
                <p>Asignación de recursos según urgencia (0-10)</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("""<div class="meli-metric-card" style="text-align:left;">
                <h3 style="font-size:1.1rem;">🤝 Conflictos</h3>
                <p>Escalamiento automático a supervisores humanos</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("""<div class="meli-metric-card" style="text-align:left;">
                <h3 style="font-size:1.1rem;">📊 Métricas</h3>
                <p>Trazabilidad completa con SQLite y panel de métricas</p>
            </div>""", unsafe_allow_html=True)

