import html
import os
import time
from pathlib import Path
from typing import Iterable, List, Tuple, Union
from uuid import uuid4

import gradio as gr

from agentwithUi import CONCURRENT_REQUESTS, MAX_QUERY_LENGTH, generate_response_sync

MAX_FILE_SNIPPET_CHARS = int(os.getenv("GRADIO_FILE_SNIPPET_LIMIT", "4000"))
SUPPORTED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".json", ".yaml", ".yml",
    ".csv", ".tsv", ".log", ".html", ".xml", ".js", ".jsx", ".ts",
    ".tsx", ".css", ".scss", ".sql", ".sh", ".bash", ".c", ".cpp",
    ".h", ".hpp", ".java", ".go", ".rs", ".php", ".rb", ".swift"
}

CHAT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    box-sizing: border-box !important;
}

:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --border-primary: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --accent-primary: #2f81f7;
    --accent-hover: #3d8bfd;
    --danger: #f85149;
}

body, html {
    margin: 0 !important;
    padding: 0 !important;
    height: 1000vh !important;
    overflow: hidden !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    height: 100vh !important;
    background: var(--bg-primary) !important;
}

.gradio-container .main {
    padding: 0 !important;
    gap: 0 !important;
    height: 100vh !important;
}

.block, .form, .wrap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Main Grid Layout */
#main-row {
  grid-template-columns: 260px 1fr !important;  /* sidebar 260px */
}

/* === SIDEBAR === */
#sidebar-col {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-primary) !important;
    height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 0 !important;
    margin: 0 !important;
}

#sidebar-header {
    padding: 12px !important;
    border-bottom: 1px solid var(--border-primary) !important;
    flex-shrink: 0 !important;
}

#new-chat-btn {
    width: 100% !important;
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-primary) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

#new-chat-btn:hover {
    background: var(--bg-primary) !important;
    border-color: var(--border-hover) !important;
}

#history-panel {
    flex: 1 !important;
    overflow-y: auto !important;
    padding: 8px !important;
}

#history-panel::-webkit-scrollbar {
    width: 6px;
}

#history-panel::-webkit-scrollbar-thumb {
    background: var(--border-primary);
    border-radius: 4px;
}

.history-empty {
    padding: 20px 12px;
    text-align: center;
    color: var(--text-secondary);
    font-size: 13px;
}

.history-item {
    padding: 10px 12px;
    margin-bottom: 4px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    color: var(--text-secondary);
    font-size: 13px;
    border: 1px solid transparent;
    display: flex;
    align-items: center;
    gap: 8px;
}

.history-item:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
}

.history-item.active {
    background: var(--bg-tertiary);
    border-color: var(--border-primary);
    color: var(--text-primary);
}

/* === CHAT AREA === */
#chat-col {
    background: var(--bg-primary) !important;
    height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 0 !important;
    margin: 0 !important;
    position: relative !important;
}

#chat-header {
    padding: 12px 20px !important;
    border-bottom: 1px solid var(--border-primary) !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    flex-shrink: 0 !important;
    background: var(--bg-primary) !important;
}

.chat-title {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

#clear-btn {
    background: transparent !important;
    border: 1px solid var(--border-primary) !important;
    color: var(--text-secondary) !important;
    border-radius: 6px !important;
    padding: 5px 12px !important;
    font-size: 12px !important;
    height: 28px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

#clear-btn:hover {
    border-color: var(--danger) !important;
    color: var(--danger) !important;
}

/* === MESSAGES AREA === */
#messages-area {
  padding: 24px 20px !important;              /* top/btm 24, sides 20 */
}


/* Hero Section */
.hero-section  { max-width: 740px !important; }

.hero-title {
    font-size: 32px !important;
    font-weight: 700 !important;
    margin-bottom: 12px !important;
    text-align: center !important;
    background: linear-gradient(135deg, #2f81f7 0%, #3fb950 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

.hero-subtitle {
    font-size: 15px !important;
    color: var(--text-secondary) !important;
    text-align: center !important;
    margin-bottom: 28px !important;
}

.hero-cards {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    width: 100%;
    max-width: 480px;
}

.hero-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-primary);
    border-radius: 12px;
    padding: 16px;
    cursor: pointer;
    transition: all 0.2s;
    text-align: center;
}

.hero-card:hover {
    border-color: var(--accent-primary);
    transform: translateY(-2px);
}

.hero-card-icon {
    font-size: 24px;
    margin-bottom: 8px;
}

.hero-card-text {
    font-size: 13px;
    color: var(--text-secondary);
}

/* Chatbot Messages */
#chatbot       { max-width: 740px !important; }

#chatbot > * {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

#chatbot .message { margin-bottom: 16px !important; }


#chatbot .message.user > div,
#chatbot .message.bot  > div {
  max-width: 75% !important;                
}


/* === INPUT AREA === */
#input-container { padding: 0 20px 20px !important; }
#input-wrapper { max-width: 740px !important; }

/* File Attachments */
.file-attachments {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 10px;
}

.file-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-primary);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 13px;
    color: var(--text-primary);
}

.file-chip-icon {
    font-size: 14px;
}

.file-chip-remove {
    background: transparent;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 0 4px;
    font-size: 18px;
    line-height: 1;
    transition: color 0.2s;
    margin-left: 4px;
}

.file-chip-remove:hover {
    color: var(--danger);
}

/* Input Row */
#input-row {
    display: flex !important;
    align-items: flex-end !important;
    gap: 8px !important;
    background: var(--bg-secondary) !important;
    border: 2px solid var(--border-primary) !important;
    border-radius: 12px !important;
    padding: 8px !important;
    transition: border-color 0.2s !important;
}

#input-row:focus-within {
    border-color: var(--accent-primary) !important;
}

#upload-btn {
    background: transparent !important;
    border: none !important;
    color: var(--text-secondary) !important;
    padding: 8px !important;
    min-width: 36px !important;
    width: 36px !important;
    height: 36px !important;
    border-radius: 8px !important;
    font-size: 18px !important;
    flex-shrink: 0 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s !important;
}

#upload-btn:hover {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
}

#msg {
    flex: 1 !important;
}

#msg textarea {
    background: transparent !important;
    border: none !important;
    color: var(--text-primary) !important;
    font-size: 14px !important;
    padding: 8px 12px !important;
    resize: none !important;
    max-height: 200px !important;
    min-height: 36px !important;
    line-height: 1.5 !important;
}

#msg textarea:focus {
    outline: none !important;
    box-shadow: none !important;
}

#msg textarea::placeholder {
    color: var(--text-secondary) !important;
}

#send-btn {
    background: var(--accent-primary) !important;
    border: none !important;
    color: white !important;
    padding: 8px 20px !important;
    min-width: 70px !important;
    height: 36px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    flex-shrink: 0 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

#send-btn:hover {
    background: var(--accent-hover) !important;
}

#send-btn:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
}

/* Thinking indicator */
.thinking-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--text-secondary);
    font-size: 13px;
    padding: 8px 0;
}

.thinking-dots {
    display: flex;
    gap: 4px;
}

.thinking-dot {
    width: 6px;
    height: 6px;
    background: var(--text-secondary);
    border-radius: 50%;
    animation: thinking-pulse 1.4s ease-in-out infinite;
}

.thinking-dot:nth-child(2) {
    animation-delay: 0.2s;
}

.thinking-dot:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes thinking-pulse {
    0%, 60%, 100% { 
        opacity: 0.3;
        transform: scale(0.8);
    }
    30% { 
        opacity: 1;
        transform: scale(1.1);
    }
}

/* Responsive */
@media (max-width: 1024px) {
  #main-row { grid-template-columns: 1fr !important; }
  #sidebar-col { display: none !important; }
}

/* Hide Gradio footer */
footer {
    display: none !important;
}

"""

ATTACHMENT_JS = """
<script>
window.removeAttachment = function(name) {
    const target = document.getElementById('remove-target');
    const trigger = document.getElementById('remove-trigger');
    if (target && trigger) {
        target.value = name;
        trigger.click();
    }
};

window.selectSession = function(sessionId) {
    const target = document.getElementById('session-target');
    const trigger = document.getElementById('session-trigger');
    if (target && trigger) {
        target.value = sessionId;
        trigger.click();
    }
};
</script>
"""

HERO_HTML = """
<div class="hero-section">
    <h1 class="hero-title">What can I help you with?</h1>
    <p class="hero-subtitle">Ask me anything or upload files to get started</p>
    <div class="hero-cards">
        <div class="hero-card">
            <div class="hero-card-icon">📝</div>
            <div class="hero-card-text">Write & Edit</div>
        </div>
        <div class="hero-card">
            <div class="hero-card-icon">💡</div>
            <div class="hero-card-text">Brainstorm Ideas</div>
        </div>
        <div class="hero-card">
            <div class="hero-card-icon">🔍</div>
            <div class="hero-card-text">Research & Learn</div>
        </div>
        <div class="hero-card">
            <div class="hero-card-icon">🛠️</div>
            <div class="hero-card-text">Code & Debug</div>
        </div>
    </div>
</div>
"""


def _file_label(upload) -> str:
    """Extract filename from upload object"""
    candidate = getattr(upload, 'name', None) or getattr(upload, 'orig_name', None) or getattr(upload, 'path', None)
    if candidate:
        return Path(candidate).name
    return 'attachment'


def _decode_bytes(data: bytes) -> Tuple[str | None, str | None]:
    """Decode bytes to text"""
    if not data:
        return "", None
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding), None
        except UnicodeDecodeError:
            continue
    return None, "binary or unsupported encoding"


def _read_uploaded(upload) -> Tuple[str, str | None, str | None]:
    """Read uploaded file"""
    name = Path(getattr(upload, "name", "uploaded_file")).name
    raw_bytes = b""
    try:
        if hasattr(upload, "seek"):
            upload.seek(0)
        if hasattr(upload, "read"):
            raw_bytes = upload.read()
        elif isinstance(upload, str):
            raw_bytes = Path(upload).read_bytes()
    except Exception as exc:
        return name, None, f"read error: {exc}"
    
    text, err = _decode_bytes(raw_bytes)
    if err:
        return name, None, err
    
    snippet = text[:MAX_FILE_SNIPPET_CHARS]
    return name, snippet, None


def _prepare_context(files: Iterable) -> Tuple[List[str], List[str], List[str]]:
    """Process uploaded files"""
    display_names: List[str] = []
    warnings: List[str] = []
    excerpts: List[str] = []
    
    for item in files or []:
        name, snippet, issue = _read_uploaded(item)
        suffix = Path(name).suffix.lower()
        
        if issue:
            warnings.append(f"{name} ({issue})")
            continue
        
        if suffix and suffix not in SUPPORTED_TEXT_EXTENSIONS:
            warnings.append(f"{name} (unsupported)")
            continue
        
        display_names.append(name)
        excerpts.append(f"File: {name}\n{snippet}".strip())
    
    return display_names, warnings, excerpts


def render_file_attachments(names: List[str]) -> str:
    """Render file chips"""
    if not names:
        return ""
    
    chips = []
    for name in names:
        safe_name = html.escape(name)
        js_name = name.replace("'", "\\'").replace('"', '\\"')
        chip = f"""
        <div class="file-chip">
            <span class="file-chip-icon">📎</span>
            <span>{safe_name}</span>
            <button class="file-chip-remove" onclick="removeAttachment('{js_name}')" aria-label="Remove">×</button>
        </div>
        """
        chips.append(chip)
    
    return f'<div class="file-attachments">{"".join(chips)}</div>'


def render_history(sessions: List[dict], current_id: str | None) -> str:
    """Render sidebar history"""
    if not sessions:
        return '<div class="history-empty">No conversations yet</div>'
    
    items = []
    for session in sessions:
        title = session.get('title') or 'New conversation'
        sid = session.get('id')
        safe_title = html.escape(title[:50])
        active = ' active' if sid == current_id else ''
        js_sid = (sid or '').replace("'", "\\'").replace('"', '\\"')
        
        item = f"""
        <div class="history-item{active}" onclick="selectSession('{js_sid}')">
            <span>💬</span>
            <span>{safe_title}</span>
        </div>
        """
        items.append(item)
    
    return ''.join(items)


MessageHistory = List[dict[str, str]]


def _normalise_history(history: Union[MessageHistory, List[Tuple[str, str]], None]) -> MessageHistory:
    """Normalize history format"""
    if not history:
        return []
    
    normalised: MessageHistory = []
    for item in history:
        if isinstance(item, dict):
            role = str(item.get('role') or 'assistant')
            content = str(item.get('content') or '')
            normalised.append({'role': role, 'content': content})
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            normalised.append({'role': 'user', 'content': str(item[0] or '')})
            normalised.append({'role': 'assistant', 'content': str(item[1] or '')})
    
    return normalised


def handle_upload(new_files: List, current_files: List) -> Tuple[List, str]:
    """Handle file uploads"""
    combined: List = list(current_files or [])
    batch = new_files or []
    
    if not isinstance(batch, list):
        batch = [batch]
    
    for file_obj in batch:
        if file_obj not in combined:
            combined.append(file_obj)
    
    names, _, _ = _prepare_context(combined)
    return combined, render_file_attachments(names)


def remove_attachment(name: str, attachments: List) -> Tuple[List, str]:
    """Remove file attachment"""
    filtered = [item for item in (attachments or []) if _file_label(item) != name]
    names, _, _ = _prepare_context(filtered)
    return filtered, render_file_attachments(names)


def upsert_session(sessions: List[dict], session_id: str, messages: MessageHistory, first_prompt: str) -> Tuple[List[dict], dict]:
    """Update or create session"""
    found = None
    for session in sessions:
        if session.get('id') == session_id:
            found = session
            break
    
    if found is None:
        found = {'id': session_id, 'title': '', 'messages': []}
        sessions.append(found)
    
    found['messages'] = messages
    
    if first_prompt and not found.get('title'):
        clipped = first_prompt.strip().split('\n')[0][:48]
        found['title'] = clipped or 'New conversation'
    
    found['updated'] = time.time()
    sessions.sort(key=lambda item: item.get('updated', 0), reverse=True)
    
    return sessions[:30], found


def load_session(session_id: str, sessions: List[dict]) -> Tuple[MessageHistory, object, List, str, str, str, str]:
    """Load chat session"""
    session = next((item for item in sessions if item.get('id') == session_id), None)
    
    if not session:
        return [], gr.update(visible=True), [], "", "", session_id, render_history(sessions, session_id)
    
    messages: MessageHistory = session.get('messages', [])
    hero_visible = not bool(messages)
    
    return (
        messages,
        gr.update(visible=hero_visible),
        [],
        "",
        "",
        session_id,
        render_history(sessions, session_id),
    )


def new_chat(sessions: List[dict]) -> Tuple[MessageHistory, object, List, str, str, List[dict], str, str]:
    """Start new chat"""
    session_id = str(uuid4())
    sessions = [{'id': session_id, 'title': '', 'messages': [], 'updated': time.time()}] + sessions
    
    return (
        [],
        gr.update(visible=True),
        [],
        "",
        "",
        sessions,
        session_id,
        render_history(sessions, session_id),
    )


def respond(
    message: str,
    history: Union[MessageHistory, List[Tuple[str, str]], None],
    attachments: List,
    sessions: List[dict],
    session_id: str | None,
) -> Iterable[Tuple[MessageHistory, str, List, str, object, List[dict], str, str]]:
    """Generate response"""
    history_messages = _normalise_history(history)
    attachments = attachments or []
    sessions = sessions or []
    session_id = session_id or str(uuid4())
    
    base_message = (message or "").strip()
    display_names, warnings, excerpts = _prepare_context(attachments)
    
    if not base_message and not excerpts:
        base_message = "Please review the attached files."
    
    augmented = base_message
    if excerpts:
        excerpt_block = "\n\n".join(excerpts)
        augmented = f"{augmented}\n\n---\nAttached files:\n{excerpt_block}".strip()
    
    user_display = base_message
    if display_names:
        user_display += "\n\n📎 " + ", ".join(display_names)
    if warnings:
        user_display += "\n\n⚠️ " + "; ".join(warnings)
    
    thinking_markup = '''<div class="thinking-indicator">
        <span>Thinking</span>
        <div class="thinking-dots">
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
        </div>
    </div>'''
    
    thinking_history = history_messages + [
        {'role': 'user', 'content': user_display},
        {'role': 'assistant', 'content': thinking_markup}
    ]
    
    yield (
        thinking_history,
        "",
        [],
        "",
        gr.update(visible=False),
        sessions,
        session_id,
        render_history(sessions, session_id),
    )
    
    trimmed_prompt = augmented[:MAX_QUERY_LENGTH] if len(augmented) > MAX_QUERY_LENGTH else augmented
    truncated = len(augmented) > MAX_QUERY_LENGTH
    
    try:
        result = generate_response_sync(trimmed_prompt)
        answer = (result.get("answer") or "").strip()
        reasoning = (result.get("reasoning") or "").strip()
    except TimeoutError:
        answer = "⏱️ Request timed out. Please try again."
        reasoning = ""
    except Exception as exc:
        answer = f"❌ Error: {exc}"
        reasoning = ""
    
    thinking_section = ""
    if reasoning:
        thinking_section = f'''
<details style="margin-top: 16px; padding: 12px; background: var(--bg-secondary); border-radius: 10px; border: 1px solid var(--border-primary);">
    <summary style="cursor: pointer; font-size: 13px; color: var(--text-secondary); font-weight: 500;">💭 View reasoning</summary>
    <pre style="margin-top: 12px; padding: 12px; background: var(--bg-tertiary); border-radius: 8px; font-size: 12px; color: var(--text-secondary); white-space: pre-wrap; overflow-x: auto; line-height: 1.6;">{html.escape(reasoning)}</pre>
</details>
'''
    
    assistant_reply = answer or "(no response)"
    if truncated:
        assistant_reply += "\n\n_Note: Input was truncated._"
    if thinking_section:
        assistant_reply += "\n\n" + thinking_section
    
    final_history = thinking_history[:-1] + [{'role': 'assistant', 'content': assistant_reply}]
    
    sessions, _ = upsert_session(sessions, session_id, final_history, base_message)
    
    yield (
        final_history,
        "",
        [],
        "",
        gr.update(visible=False),
        sessions,
        session_id,
        render_history(sessions, session_id),
    )


def clear_chat(sessions: List[dict], session_id: str | None) -> Tuple[MessageHistory, str, List, str, object, List[dict], str | None, str]:
    """Clear current chat"""
    sessions = sessions or []
    
    if session_id:
        existing = next((item for item in sessions if item.get('id') == session_id), None)
        if existing and existing.get('messages'):
            sessions, _ = upsert_session(sessions, session_id, [], '')
    
    return (
        [],
        "",
        [],
        "",
        gr.update(visible=True),
        sessions,
        session_id,
        render_history(sessions, session_id),
    )


def build_ui() -> gr.Blocks:
    """Build Gradio interface"""
    with gr.Blocks(title="AI Chat Assistant", theme=gr.themes.Soft(), css=CHAT_CSS) as demo:
        gr.HTML(ATTACHMENT_JS, visible=False)
        
        # State
        attachments_state = gr.State([])
        sessions_state = gr.State([])
        session_id_state = gr.State(str(uuid4()))
        
        # Hidden triggers
        remove_target = gr.Textbox(visible=False, elem_id="remove-target")
        remove_trigger = gr.Button("", visible=False, elem_id="remove-trigger")
        session_target = gr.Textbox(visible=False, elem_id="session-target")
        session_trigger = gr.Button("", visible=False, elem_id="session-trigger")
        
        with gr.Row(elem_id="main-row"):
            # Sidebar
            with gr.Column(elem_id="sidebar-col", scale=0):
                with gr.Group(elem_id="sidebar-header"):
                    new_chat_btn = gr.Button("+ New Chat", elem_id="new-chat-btn")
                
                history_panel = gr.HTML(
                    '<div class="history-empty">No conversations yet</div>',
                    elem_id="history-panel"
                )
            
            # Main chat
            with gr.Column(elem_id="chat-col", scale=1):
                # Header
                with gr.Row(elem_id="chat-header"):
                    gr.HTML('<div class="chat-title">AI Chat Assistant</div>')
                    clear_btn = gr.Button("Clear", elem_id="clear-btn")
                
                # Messages
                with gr.Column(elem_id="messages-area"):
                    hero = gr.HTML(HERO_HTML, visible=True)
                    chatbot = gr.Chatbot(
                        height=None,
                        label="",
                        type="messages",
                        elem_id="chatbot",
                        show_label=False,
                    )
                
                # Input
                with gr.Column(elem_id="input-container"):
                    with gr.Group(elem_id="input-wrapper"):
                        attachment_display = gr.HTML("", elem_id="attachment-display")
                        
                        with gr.Row(elem_id="input-row"):
                            upload_btn = gr.UploadButton(
                                "📎",
                                file_types=["text"],
                                file_count="multiple",
                                elem_id="upload-btn",
                                size="sm",
                            )
                            
                            msg = gr.Textbox(
                                placeholder="Type your message here...",
                                lines=1,
                                max_lines=8,
                                elem_id="msg",
                                show_label=False,
                                container=False,
                            )
                            
                            send_btn = gr.Button("Send", elem_id="send-btn")
        
        # Events
        upload_btn.upload(
            handle_upload,
            inputs=[upload_btn, attachments_state],
            outputs=[attachments_state, attachment_display],
        )
        
        remove_trigger.click(
            remove_attachment,
            inputs=[remove_target, attachments_state],
            outputs=[attachments_state, attachment_display],
        )
        
        session_trigger.click(
            load_session,
            inputs=[session_target, sessions_state],
            outputs=[chatbot, hero, attachments_state, attachment_display, msg, session_id_state, history_panel],
        )
        
        clear_btn.click(
            clear_chat,
            inputs=[sessions_state, session_id_state],
            outputs=[chatbot, msg, attachments_state, attachment_display, hero, sessions_state, session_id_state, history_panel],
        )
        
        new_chat_btn.click(
            new_chat,
            inputs=[sessions_state],
            outputs=[chatbot, hero, attachments_state, attachment_display, msg, sessions_state, session_id_state, history_panel],
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot, attachments_state, sessions_state, session_id_state],
            outputs=[chatbot, msg, attachments_state, attachment_display, hero, sessions_state, session_id_state, history_panel],
        )
        
        send_btn.click(
            respond,
            inputs=[msg, chatbot, attachments_state, sessions_state, session_id_state],
            outputs=[chatbot, msg, attachments_state, attachment_display, hero, sessions_state, session_id_state, history_panel],
        )
        
        return demo


def main() -> None:
    """Launch app"""
    demo = build_ui()
    concurrency = max(1, int(os.getenv("GRADIO_CONCURRENCY", str(CONCURRENT_REQUESTS))))
    server_name = os.getenv("GRADIO_HOST", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "0") == "1"
    
    demo.queue(default_concurrency_limit=concurrency)
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_api=False,
    )


if __name__ == "__main__":
    main()