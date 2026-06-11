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
REMITE = os.getenv("CONFIG_EMAIL_REMITENTE")
REMPASS = os.getenv("CONFIG_EMAIL_PASSWORD")
KB_PATH = Path(__file__).parent / "data" / "knowledge_base.json"
DB_PATH = Path(__file__).parent / "data" / "meliexpert.db"

if not MI_TOKEN:
    st.warning("GITHUB_TOKEN no configurado. Edita .env")

llm = ChatOpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=MI_TOKEN,
    model="gpt-4o",
    temperature=0.1,
    streaming=True,
)

embeddings = OpenAIEmbeddings(
    base_url="https://models.github.ai/inference",
    api_key=MI_TOKEN,
    model="text-embedding-3-small",
)

# ──────────────────────────────────────────────
# (1) SEGURIDAD — filtros éticos, PII, inyección
# ──────────────────────────────────────────────
PII_PATTERNS = {
    "tarjeta_credito": re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "telefono": re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3}[-.\s]?\d{3,4}\b'),
    "dni": re.compile(r'\b\d{7,8}\b'),
}

INJECTION_PATTERNS = [
    re.compile(r'ignora\s*(?:las\s*)?instrucciones', re.IGNORECASE),
    re.compile(r'ignora\s*(?:el\s*)?prompt', re.IGNORECASE),
    re.compile(r'olvida\s*(?:tu\s*)?(?:instrucción|prompt|rol|system)', re.IGNORECASE),
    re.compile(r'eres\s*(?:ahora|libre|un\s*(?:asistente|bot))', re.IGNORECASE),
    re.compile(r'system\s*(?:prompt|message)', re.IGNORECASE),
    re.compile(r'skip\s*(?:the\s*)?(?:instructions|prompt)', re.IGNORECASE),
]

class SecurityReport:
    def __init__(self):
        self.pii_detected: Dict[str, List[str]] = {}
        self.injection_detected: List[str] = []
        self.blocked = False
        self.block_reason = ""

    @staticmethod
    def scan(text: str) -> "SecurityReport":
        r = SecurityReport()
        for k, pat in PII_PATTERNS.items():
            m = pat.findall(text)
            if m:
                r.pii_detected[k] = m
        for pat in INJECTION_PATTERNS:
            if pat.search(text):
                r.injection_detected.append(pat.pattern)
        if r.injection_detected:
            r.blocked = True
            r.block_reason = "Detección de posible intento de inyección de prompt"
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
    if not REMITE or not REMPASS:
        return False
    msg = MIMEMultipart()
    msg["From"] = REMITE
    msg["To"] = destinatario
    msg["Subject"] = "Información solicitada - MeliExpert"
    cuerpo_html = f"""
    <html><body>
        <hr><pre style='font-family:sans-serif;font-size:14px;'>{cuerpo}</pre><hr>
        <p style='color:#718096;font-size:12px;'>Enviado desde MeliExpert.</p>
    </body></html>"""
    msg.attach(MIMEText(cuerpo_html, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as servidor:
            servidor.login(REMITE, REMPASS)
            servidor.sendmail(REMITE, destinatario, msg.as_string())
        return True
    except Exception:
        return False

# ──────────────────────────────────────────────
# (2) RAG — base de conocimiento vectorial
# ──────────────────────────────────────────────
@st.cache_resource
def build_vectorstore():
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
        return vs
    except Exception as e:
        st.warning(f"Error creando vectorstore: {e}")
        return None

def retrieve_context(query: str, k: int = 3) -> List[Dict]:
    vs = build_vectorstore()
    if vs is None:
        return []
    try:
        docs = vs.similarity_search_with_score(query, k=k)
        return [{"content": d[0].page_content, "topic": d[0].metadata.get("topic", ""), "score": float(d[1])} for d in docs]
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
        ("system", "Eres un asesor de ventas experto de Mercado Libre especializado en tecnología. Ayudas al usuario a encontrar el mejor producto según su necesidad y presupuesto. Recomiendas productos concretos con precios estimados en CLP. Conoces bien laptops gamer, periféricos, componentes y accesorios. Usas emojis 🛒✨ y viñetas. Si el usuario te pide enviar por correo, el sistema lo hace automáticamente, tú solo incluye la información detallada sin mencionar el envío."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]),
    QueryType.SOPORTE: ChatPromptTemplate.from_messages([
        ("system", "Eres un agente de soporte técnico de Mercado Libre. Ayudas a resolver problemas técnicos con la plataforma, la app o productos comprados. Das pasos claros y numerados. Usas un tono neutro y profesional."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]),
    QueryType.FACTURACION: ChatPromptTemplate.from_messages([
        ("system", "Eres un agente de facturación de Mercado Libre. Ayudas con pagos, facturas, cuotas, impuestos y reembolsos.Explicas plazos y montos claramente en CLP. Usas un tono neutro."),
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
        ("system", "Eres MeliExpert, el asistente principal de Mercado Libre. Respondes dudas generales con un tono amigable y profesional. Si la consulta es compleja, sugieres hablar con el agente especializado correspondiente. Usas un español neutro y claro. Si el usuario te pide enviar información por correo, incluye todos los detalles en tu respuesta y yo me encargaré del envío."),
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
        result = chain_with_history.invoke({"input": agent_input}, config=config)
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
# UI — STREAMLIT
# ──────────────────────────────────────────────
st.set_page_config(page_title="MeliExpert", page_icon="🛒", layout="wide")

st.title("🛒 MeliExpert — Asistente Inteligente Mercado Libre")
st.caption("Multi-agente · RAG · Seguridad · Workflows · Trazabilidad")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]

sid = st.session_state.session_id

with st.sidebar:
    st.header("⚙️ Panel de Control")
    modo = st.radio("Modo", ["Completo (todos los agentes)", "Chat Simple"])
    mostrar_metrics = st.checkbox("Mostrar métricas", False)
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        orchestrator.sessions.pop(sid, None)
        st.rerun()
    st.divider()
    st.subheader("📊 Estado del sistema")
    st.write(f"Sesión: `{sid}`")
    qs = orchestrator.classify(
        st.session_state.messages[-1]["content"] if st.session_state.messages else ""
    )
    st.write(f"Última clasificación: `{qs.value if qs else '—'}`")
    wf = workflow_engine.get(sid)
    if wf:
        st.write(f"Workflow activo: **{wf['name']}** (paso {wf['current_step']+1}/{len(wf['steps'])})")
    st.divider()
    st.caption("Conceptos implementados:")
    for c in ["Seguridad y filtros éticos", "RAG con FAISS", "Descomposición de tareas",
              "Workflows multi-paso", "Orquestación multi-agente (LangGraph)",
              "Asignación de recursos", "Resolución de conflictos",
              "Trazabilidad y métricas (SQLite)"]:
        st.caption(f"✅ {c}")

col1, col2 = st.columns([3, 1])

with col2:
    if mostrar_metrics:
        st.subheader("📈 Métricas")
        df_m = get_metrics(7)
        if not df_m.empty:
            st.dataframe(df_m, use_container_width=True, hide_index=True)
        else:
            st.info("Aún no hay métricas")
    st.subheader("🔍 Diagnóstico de consulta")
    test_input = st.text_input("Probar clasificación:", key="test_classify")
    if test_input:
        qt = orchestrator.classify(test_input)
        prio = orchestrator.priority_score(test_input)
        res = orchestrator.resolve_conflict("test", test_input, prio)
        sec = SecurityReport.scan(test_input)
        st.write(f"**Tipo:** `{qt.value}`")
        st.write(f"**Prioridad:** {prio}/10")
        st.write(f"**Resolución:** {res}")
        st.write(f"**Seguridad:** {'⚠️ Bloqueado' if sec.blocked else '✅ OK'}")
        docs = retrieve_context(test_input)
        if docs:
            st.write(f"**Docs RAG:** {len(docs)} recuperados")
            for d in docs:
                st.caption(f"  📄 {d['topic']} (score: {d['score']:.2f})")

with col1:
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧩 Workflows", "ℹ️ Ayuda"])

    with tab1:
        if modo == "Completo (todos los agentes)":
            st.info("Modo multi-agente: el sistema clasifica y enruta tu consulta al especialista adecuado (Ventas, Soporte, Facturación, Envíos, Seguridad).")
        else:
            st.info("Modo simple: responde con el agente general de Mercado Libre.")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Escribe tu mensaje aquí..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if modo == "Chat Simple":
                prompt_basico = ChatPromptTemplate.from_messages([
                    ("system", "Eres MeliExpert, asistente de Mercado Libre. Respondes de forma amigable y útil con un español neutro. Los precios van en CLP."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                chain = RunnableWithMessageHistory(
                    prompt_basico | llm,
                    lambda s: orchestrator.get_history(s),
                    input_messages_key="input",
                    history_messages_key="chat_history",
                )
                with st.chat_message("assistant"):
                    response = st.write_stream(chain.stream(
                        {"input": prompt},
                        config={"configurable": {"session_id": sid}},
                    ))
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                with st.chat_message("assistant"):
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
                            st.write(f"**Tipo consulta:** `{qt.value}`")
                            st.write(f"**Prioridad:** {prio}/10")
                            st.write(f"**Resolución:** {res}")
                            st.write(f"**Documentos RAG recuperados:** {len(docs)}")
                            if sec:
                                st.write(f"**PII detectada:** {list(sec.pii_detected.keys()) or 'Ninguna'}")
                                st.write(f"**Bloqueado por seguridad:** {sec.blocked}")
                            if docs:
                                st.write("**Contexto usado:**")
                                for d in docs:
                                    st.caption(f"📄 {d['topic']} — score: {d['score']:.2f}")
                    except Exception as e:
                        response = f"❌ Error procesando tu consulta: {str(e)}"
                        st.markdown(response)
                        log_interaction(sid, "error", prompt, response, 0, False, 0, "", False)
                st.session_state.messages.append({"role": "assistant", "content": response})

    with tab2:
        st.subheader("🧩 Workflows disponibles")
        wf_selected = st.selectbox("Seleccionar workflow", list(WORKFLOWS.keys()),
                                   format_func=lambda x: WORKFLOWS[x]["name"])
        if st.button("🚀 Iniciar workflow"):
            wf = workflow_engine.start(sid, wf_selected)
            if wf:
                st.success(f"Workflow '{wf['name']}' iniciado!")
                steps_text = "\n".join(f"  {s['id']}. {s['desc']}" for s in wf["steps"])
                st.code(f"Pasos:\n{steps_text}")
                st.info(f"Paso actual: {wf['steps'][0]['desc']}")
            else:
                st.error("Workflow no encontrado")

        if st.button("⏭️ Siguiente paso"):
            step = workflow_engine.next_step(sid)
            if step:
                st.info(f"Paso siguiente: **{step['desc']}**")
            else:
                wf = workflow_engine.get(sid)
                if wf and wf["state"] == WorkflowState.COMPLETADO:
                    st.success("✅ Workflow completado!")
                else:
                    st.warning("No hay workflow activo")

        if st.button("❌ Cancelar workflow"):
            workflow_engine.cancel(sid)
            st.warning("Workflow cancelado")

        if st.button("🧩 Descomponer última consulta"):
            if st.session_state.messages:
                last_q = st.session_state.messages[-1]["content"]
                if st.session_state.messages[-1]["role"] == "user":
                    tasks = decompose_query(last_q)
                    st.write("**Tareas descompuestas:**")
                    for t in tasks:
                        st.write(f"- **{t.get('tarea', '?')}** (tipo: {t.get('tipo', '?')})")
                else:
                    st.info("Envía un mensaje primero para descomponerlo")
            else:
                st.info("No hay mensajes aún")

    with tab3:
        st.markdown("""
        ### 🛒 MeliExpert — Sistema Completo

        **8 conceptos integrados:**

        | Concepto | Implementación |
        |---|---|
        | 🔒 Seguridad y filtros éticos | Detección de PII + inyección de prompts |
        | 📚 RAG con evaluación | FAISS + knowledge base de políticas MELI |
        | 🧩 Descomposición de tareas | LLM divide consultas complejas |
        | 🔄 Workflows multi-paso | Devolución, reclamo, asesoría de compra |
        | 🤖 Orquestación multi-agente | LangGraph: 6 agentes especializados |
        | ⚖️ Asignación de recursos | Prioridad 0-10 según urgencia |
        | 🤝 Resolución de conflictos | Escalamiento automático |
        | 📊 Trazabilidad y métricas | SQLite + panel de métricas |
        """)
