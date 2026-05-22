# ingenieria-en-inteligencia-artificial-parcial1
<<<<<<< HEAD

=======
>>>>>>> 8057d295ea702433102aa2fb7ad7ca01d5e91038
# Proyecto: Chatbot de Soporte Integral - Mercado Libre 🚀

##  Descripción General
<img width="500" height="250" alt="image" src="https://github.com/user-attachments/assets/037db55f-4129-4206-873d-da616ae45532" />

Este proyecto consiste en el desarrollo de un asistente virtual inteligente diseñado para integrarse en el ecosistema de **Mercado Libre**. El chatbot actúa como el primer punto de contacto para usuarios (compradores y vendedores), optimizando la resolución de dudas, el seguimiento de pedidos y la gestión de servicios financieros.

---

## Identificación de la Organización

### **Nombre de la Organización**
**Mercado Libre S.R.L.**

### **Rubro y Sector**
* **E-commerce:** Plataforma líder en compraventa de productos en América Latina.
* **Fintech:** Servicios de procesamiento de pagos, créditos y billetera digital a través de **Mercado Pago**.
* **Logística:** Gestión de envíos y almacenamiento mediante **Mercado Envíos**.

### **Tamaño y Contexto**
* **Categoría:** Gran Empresa / Unicornio Tecnológico.
* **Alcance:** +50,000 colaboradores y presencia en 18 países.
* **Contexto:** La empresa opera en un entorno de **alta criticidad**, donde la velocidad de respuesta impacta directamente en la reputación del vendedor y la confianza del comprador.

---

##  L1.1: Ingeniería de Prompts (Estrategia del Modelo)

Para este caso, se han diseñado prompts específicos que ajustan su estructura y contenido según el requerimiento informacional.

### 1. Definición del Sistema (Role Prompting)
Define la identidad y límites del bot para mantener la coherencia de marca.

> **Prompt:**
> "Actúa como un Asistente Virtual experto de Mercado Libre. Tu tono debe ser profesional, cercano y resolutivo. Si un usuario pregunta por datos sensibles, deniega la petición por seguridad. Tu prioridad es la satisfacción del cliente sin comprometer las políticas de la empresa."

### 2. Clasificación de Requerimientos (Zero-Shot)
Utilizado para dirigir la consulta al departamento correcto (E-commerce o Fintech).

> **Prompt:**
> "Clasifica el siguiente mensaje en una de estas categorías: [LOGISTICA], [PAGOS], [CUENTA], [RECLAMOS]. 
> Mensaje: 'Mi tarjeta fue rechazada pero el banco dice que está bien'.
> Salida: [PAGOS]"

### 3. Extracción de Datos (Information Extraction)
Ajuste de estructura para alimentar bases de datos o APIs de seguimiento.

> **Prompt:**
> "Extrae el número de operación y el motivo de contacto del siguiente texto: 
> 'Hola, tengo un problema con el envío 400029384, el paquete llegó golpeado'.
> 
> Formato de salida (JSON):
> {
>   'order_id': '400029384',
>   'issue': 'producto_dañado'
> }"

---

##  Especificaciones Técnicas del Bot
* **Modelo Base:** GPT-4o / Gemini 1.5 Pro.
* **Arquitectura:** RAG (Retrieval-Augmented Generation) para consultar bases de conocimiento internas.
* **Flujos Principales:**
    1.  Rastreo de envíos en tiempo real.
    2.  Gestión de devoluciones automatizadas.
    3.  Soporte técnico para vendedores.

---

##  Objetivos del Repositorio
* Documentar la lógica de conversación.
* Centralizar los prompts optimizados para el caso Mercado Libre.
* Establecer protocolos de escalada a agentes humanos.

---
> **Estado del Proyecto:** En desarrollo (Fase de diseño de prompts).
