"""
app.py – Gradio web UI for the RAG chatbot.
Connects to the FastAPI backend or runs the chatbot directly.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

# ── Backend mode: direct or API ───────────────────────────────────────────────

USE_API = os.getenv("USE_API", "false").lower() == "true"
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TOKEN = os.getenv("API_TOKEN", "")

_chatbot_instance = None


def _get_chatbot():
    global _chatbot_instance
    if _chatbot_instance is None:
        from src.chatbot import RAGChatbot
        _chatbot_instance = RAGChatbot()
    return _chatbot_instance


def _ask_direct(question: str, session_id: str, persona: str) -> tuple[str, list]:
    """Call chatbot directly (no API)."""
    bot = _get_chatbot()
    result = bot.ask(question, session_id=session_id, persona=persona)
    sources_text = ""
    if result.get("sources"):
        sources_text = "\n\n**Sources:**\n" + "\n".join(
            f"- `{s['source']}` (score: {s['score']})" for s in result["sources"]
        )
    cached_badge = " *(cached)*" if result.get("cached") else ""
    return result["answer"] + sources_text + cached_badge, result.get("sources", [])


def _ask_api(question: str, session_id: str, persona: str) -> tuple[str, list]:
    """Call the FastAPI backend."""
    import requests
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {"question": question, "session_id": session_id, "persona": persona}
    try:
        resp = requests.post(f"{API_URL}/ask", json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["answer"], data.get("sources", [])
    except Exception as exc:
        return f"API error: {exc}", []


# ── Gradio chat function ──────────────────────────────────────────────────────

def chat(
    message: str,
    history: list,
    session_id: str,
    persona: str,
) -> tuple[str, list]:
    if not message.strip():
        return "", history

    if USE_API:
        answer, _ = _ask_api(message, session_id, persona)
    else:
        answer, _ = _ask_direct(message, session_id, persona)

    history.append((message, answer))
    return "", history


def ingest_docs(directory: str) -> str:
    if USE_API:
        return "Ingestion via API not supported in this UI. Use the /ingest endpoint."
    try:
        bot = _get_chatbot()
        count = bot.ingest(directory)
        return f"✅ Ingested {count} chunks from `{directory}`"
    except Exception as exc:
        return f"❌ Error: {exc}"


def submit_feedback(
    session_id: str,
    last_question: str,
    last_answer: str,
    feedback_type: str,
) -> str:
    if not last_question or not last_answer:
        return "No conversation to rate yet."
    try:
        bot = _get_chatbot()
        if feedback_type == "👍 Helpful":
            bot.feedback_store.thumbs_up(session_id, last_question, last_answer)
        else:
            bot.feedback_store.thumbs_down(session_id, last_question, last_answer)
        return f"Feedback recorded: {feedback_type}"
    except Exception as exc:
        return f"Error recording feedback: {exc}"


# ── Gradio UI layout ──────────────────────────────────────────────────────────

with gr.Blocks(
    title="RAG Chatbot",
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 900px; margin: auto; }",
) as demo:
    session_state = gr.State(value=str(uuid.uuid4()))
    last_q_state = gr.State(value="")
    last_a_state = gr.State(value="")

    gr.Markdown(
        """
        # 🤖 RAG Chatbot
        Powered by **Mistral** + **ChromaDB** + **LangChain**
        """
    )

    with gr.Tab("💬 Chat"):
        chatbot_ui = gr.Chatbot(
            label="Conversation",
            height=450,
            show_copy_button=True,
        )
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask a question about your documents...",
                label="Your question",
                scale=4,
                lines=2,
            )
            persona_select = gr.Dropdown(
                choices=["default", "hr", "tech", "admin"],
                value="default",
                label="Persona",
                scale=1,
            )
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary", scale=3)
            clear_btn = gr.Button("Clear", scale=1)

        with gr.Row():
            feedback_radio = gr.Radio(
                choices=["👍 Helpful", "👎 Not helpful"],
                label="Rate last answer",
                interactive=True,
            )
            feedback_btn = gr.Button("Submit Feedback", scale=1)
            feedback_status = gr.Textbox(label="Feedback status", interactive=False, scale=2)

        def on_send(message, history, session_id, persona):
            new_msg, new_history = chat(message, history, session_id, persona)
            last_q = message
            last_a = new_history[-1][1] if new_history else ""
            return new_msg, new_history, last_q, last_a

        send_btn.click(
            on_send,
            inputs=[msg_input, chatbot_ui, session_state, persona_select],
            outputs=[msg_input, chatbot_ui, last_q_state, last_a_state],
        )
        msg_input.submit(
            on_send,
            inputs=[msg_input, chatbot_ui, session_state, persona_select],
            outputs=[msg_input, chatbot_ui, last_q_state, last_a_state],
        )
        clear_btn.click(lambda: ([], ""), outputs=[chatbot_ui, msg_input])

        feedback_btn.click(
            submit_feedback,
            inputs=[session_state, last_q_state, last_a_state, feedback_radio],
            outputs=[feedback_status],
        )

    with gr.Tab("📁 Ingest Documents"):
        gr.Markdown("Upload documents to the knowledge base by specifying a directory path.")
        ingest_dir = gr.Textbox(
            value="./data/documents",
            label="Documents directory",
        )
        ingest_btn = gr.Button("Ingest", variant="primary")
        ingest_status = gr.Textbox(label="Status", interactive=False)
        ingest_btn.click(ingest_docs, inputs=[ingest_dir], outputs=[ingest_status])

    with gr.Tab("ℹ️ Session Info"):
        gr.Markdown("Current session details.")
        session_display = gr.Textbox(
            label="Session ID",
            interactive=False,
        )
        new_session_btn = gr.Button("New Session")

        def show_session(sid):
            return sid

        demo.load(show_session, inputs=[session_state], outputs=[session_display])
        new_session_btn.click(
            lambda: (str(uuid.uuid4()), [], ""),
            outputs=[session_state, chatbot_ui, msg_input],
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
