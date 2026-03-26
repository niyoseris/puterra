# Puterra — Cloud OS with AI Agent

Puterra is a self-hosted, browser-based **Cloud OS** built with a Rust/Actix-web backend and a vanilla JS frontend. It gives every user a personal file system, a multi-conversation AI chat powered by any OpenAI-compatible LLM, an in-browser code editor, and a share-key system so you can expose your AI to others without revealing your actual API key.

---

## Features

| Area | Details |
|---|---|
| 🤖 **AI Agent** | Agentic chat with real-time SSE streaming. Native tool-calling (Ollama / OpenAI) with ReAct fallback for models that don't support function calling. |
| 🧠 **Thinking / Reasoning** | Shows model reasoning (`<think>` tags and native `thinking` fields) in collapsible blocks, streamed live. |
| 🛠️ **Agent Tools** | `web_search`, `web_fetch`, `file_read`, `file_write`, `file_edit`, `file_create`, `file_delete`, `file_move`, `file_list`, `shell_exec`, `memory_store`, `memory_search`, `run_python`, `run_javascript`, `create_pdf` |
| 📁 **File Manager** | Per-user file storage, folder support, upload/download, right-click context menu (rename, delete, move). |
| 💻 **Code Editor** | In-browser editor with Python, JavaScript and WebAssembly execution tabs. |
| 🔑 **Share Keys** | Create `sk_…` tokens that give others access to your LLM config without exposing your real API key. |
| 📱 **Mobile Responsive** | Full-viewport windows on mobile, collapsible sidebar, touch-friendly controls. |
| 🔒 **Auth** | Username/password with SHA-256 hashed passwords and session tokens. |
| ⚙️ **Settings** | Live-editable LLM config (local & cloud endpoints), timeouts, temperature, max tokens, shell toggle, admin password. |

---

## Tech Stack

- **Backend** — Rust · [Actix-web 4](https://actix.rs/) · SSE streaming · [reqwest](https://docs.rs/reqwest)
- **Frontend** — Vanilla JS · [Tailwind CSS](https://tailwindcss.com/) · Font Awesome
- **PDF generation** — [printpdf](https://docs.rs/printpdf) (pure Rust, no external binaries)
- **LLM** — Any OpenAI-compatible API (Ollama local, Ollama Cloud, OpenAI, etc.)

---

## Quick Start

### Prerequisites
- [Rust toolchain](https://rustup.rs/) (stable)

### Run

```bash
git clone https://github.com/niyoseris/puterra.git
cd puterra
cargo run --release
```

Open **http://localhost:8080** in your browser.

Default admin credentials:
- **Username:** `admin`
- **Password:** `puterra2026`

> ⚠️ Change the admin password in Settings before exposing the server publicly.

---

## Configuration

All settings are persisted in `data/settings.json` and editable via the Settings panel in the UI.

| Key | Default | Description |
|---|---|---|
| `llm_active_source` | `"local"` | `"local"` or `"cloud"` |
| `llm_api_url_local` | `http://localhost:11434/api/chat` | Local Ollama endpoint |
| `llm_api_url_cloud` | `https://ollama.com/api/chat` | Cloud LLM endpoint |
| `llm_model_local` | `llama3.2` | Model name for local |
| `llm_model_cloud` | `llama3.3` | Model name for cloud |
| `llm_temperature` | `0.7` | Generation temperature |
| `llm_max_tokens` | `4096` | Max output tokens |
| `llm_think` | `false` | Enable extended thinking |
| `max_agent_iterations` | `6` | Max ReAct loop steps |
| `shell_enabled` | `true` | Allow `shell_exec` tool |
| `timeout_agent` | `120` | Agent request timeout (s) |
| `chat_context_limit` | `10` | Messages kept in context |

You can also copy `data/settings.json` from the example:

```bash
cp data/settings.example.json data/settings.json
```

---

## API Overview

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/login` | Authenticate, receive session token |
| POST | `/api/signup` | Register a new user |
| POST | `/api/agent` | SSE-streamed agentic chat |
| POST | `/api/chat` | Simple single-turn LLM chat |
| GET | `/api/files/{username}` | List user files |
| POST | `/api/files/read` | Read a file |
| POST | `/api/files/write` | Write a file |
| POST | `/api/files/create` | Create file or folder |
| POST | `/api/files/delete` | Delete file or folder |
| POST | `/api/files/rename` | Rename file or folder |
| GET | `/api/files/download/{username}/{path}` | Download a file |
| POST | `/api/files/upload/{username}` | Upload files (multipart) |
| POST | `/api/share-keys` | Create a share key |
| GET | `/api/share-keys` | List your share keys |
| DELETE | `/api/share-keys/{id}` | Revoke a share key |
| GET | `/api/settings` | Get current settings |
| POST | `/api/settings` | Update settings |
| GET | `/api/tools` | List available agent tools |
| POST | `/api/run` | Execute code (Python/JS) |
| GET | `/api/health` | Health check |

---

## Agent SSE Protocol

`POST /api/agent` streams newline-delimited `data: {...}` events:

```
data: {"type":"thinking","content":"..."}
data: {"type":"tool_call","tool":"web_search","input":{...},"step":1}
data: {"type":"tool_result","tool":"web_search","result":"...","step":1}
data: {"type":"answer","answer":"...","success":true}
data: {"type":"done"}
```

---

## Share Keys

Share keys let you give others access to your LLM configuration without exposing your real API key.

1. Open **Settings → Share Keys**
2. Click **Create Share Key**, fill in a label and optionally set a usage limit
3. Share the generated `sk_…` token
4. Recipients use the token in the `share_key` field of `/api/agent` or `/api/chat` requests

---

## Project Structure

```
puterra/
├── src/
│   └── main.rs          # Entire backend (Rust/Actix-web)
├── public/
│   └── index.html       # Entire frontend (single-page app)
├── data/
│   ├── settings.json    # Runtime settings (git-ignored)
│   ├── users/           # Per-user file storage (git-ignored)
│   └── share_keys.json  # Share key store (git-ignored)
├── Cargo.toml
└── README.md
```

---

## License

MIT
