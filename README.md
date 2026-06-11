# MeliExpert — Asistente Inteligente de Mercado Libre

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![GPT-4o](https://img.shields.io/badge/Model-GPT--4o-412991?logo=openai&logoColor=white)](https://github.com/marketplace/models)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-0066FF)](https://github.com/facebookresearch/faiss)
[![SQLite](https://img.shields.io/badge/DB-SQLite-003B57?logo=sqlite&logoColor=white)](https://sqlite.org)

> Chatbot multi-agente con 8 conceptos de IA integrados, desplegado en AWS EC2 con dominio DuckDNS.

## Demo

**http://meliexpert.duckdns.org:8501**

## Arquitectura

```
Usuario → Seguridad (PII + inyección) → Clasificación (tipo/prioridad/conflicto)
         → Router condicional → 6 Agentes especializados (LangGraph)
                              → o Escalamiento automático
                              → siempre: RAG (FAISS) + Trazabilidad (SQLite)
```

## Conceptos Implementados

| # | Concepto | Implementación |
|---|----------|---------------|
| 1 | **Seguridad y filtros éticos** | Detección de PII (tarjetas, emails, teléfonos, DNI) con enmascaramiento automático + detección de inyección de prompts |
| 2 | **RAG con evaluación** | FAISS sobre 21 documentos de políticas de MELI con RecursiveCharacterTextSplitter |
| 3 | **Descomposición de tareas** | LLM divide consultas complejas en subtareas secuenciales con dependencias |
| 4 | **Workflows multi-paso** | 3 workflows: Devolución (5 pasos), Reclamo (4 pasos), Asesoría de Compra (4 pasos) con máquina de estados |
| 5 | **Orquestación multi-agente** | LangGraph con 6 agentes (Ventas, Soporte, Facturación, Envíos, Seguridad, General) + enrutamiento condicional |
| 6 | **Asignación de recursos** | Prioridad 0-10 basada en palabras clave de urgencia e insatisfacción |
| 7 | **Resolución de conflictos** | Escalamiento automático por keywords legales o prioridad ≥ 9 |
| 8 | **Trazabilidad y métricas** | SQLite con panel de métricas en sidebar (últimos 7 días) |

## Stack Tecnológico

- **Frontend:** Streamlit
- **LLM:** GPT-4o vía GitHub Models (gratuito)
- **Embeddings:** text-embedding-3-small
- **Vector Store:** FAISS (CPU)
- **Orquestación:** LangGraph
- **Base de datos:** SQLite
- **Infraestructura:** AWS EC2 (t2.micro), DuckDNS, Nginx reverse proxy

## Instalación Local

```bash
git clone https://github.com/tuusuario/meliexpert.git
cd meliexpert
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Crear `.env`:
```env
GITHUB_TOKEN=ghp_tu_token_classic
CONFIG_EMAIL_REMITENTE=tucorreo@gmail.com
CONFIG_EMAIL_PASSWORD=contraseña_app
```

Ejecutar:
```bash
streamlit run app.py --server.port 8501
```

## Despliegue en AWS EC2

1. Lanzar instancia Amazon Linux 2023 (t2.micro)
2. Abrir puertos 22, 80, 8501 en Security Group
3. Asociar Elastic IP
4. Subir proyecto y ejecutar:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

5. (Opcional) Configurar DuckDNS + Nginx para dominio personalizado

## Estructura del Proyecto

```
├── app.py                  # Aplicación Streamlit completa
├── data/
│   ├── knowledge_base.json # Base de conocimiento RAG (21 docs)
│   └── meliexpert.db       # SQLite de trazabilidad
├── requirements.txt
└── .env
```

## Licencia

MIT
