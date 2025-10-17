import os, json, time, threading, requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# ===== OpenAI (ChatGPT brain) =====
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

load_dotenv(".env")

# ===== Config =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
SYSTEM_PROMPT  = os.getenv(
    "CODEX_SYSTEM_PROMPT",
    "You are Codex Core, Amar’s hands-off operator. "
    "Speak naturally and concisely. When asked to calculate or answer, reply directly."
)

# Webhooks (optional)
WEBHOOKS = {
    "PLAN": os.getenv("WEBHOOK_PLAN", ""),
    "EXEC": os.getenv("WEBHOOK_EXEC", ""),
    "AD":   os.getenv("WEBHOOK_AD",   ""),
    "PNL":  os.getenv("WEBHOOK_PNL",  ""),
    "LOG":  os.getenv("WEBHOOK_LOG",  "")
}

# ===== Notion (shared brain) =====
NOTION_TOKEN        = os.getenv("NOTION_TOKEN", "")
NOTION_QUEUE_DBID   = os.getenv("NOTION_DATABASE_ID", "")   # Command Queue
NOTION_MEMORY_DBID  = os.getenv("NOTION_MEMORY_DBID", "")   # Memory
NOTION_API          = "https://api.notion.com/v1"
NOTION_VERSION      = "2022-06-28"

def now_iso(): return datetime.utcnow().isoformat() + "Z"

def notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

def notion_create_row(dbid:str, properties:dict):
    if not (NOTION_TOKEN and dbid):
        return {"ok": False, "reason": "Notion token or DB id missing"}
    body = {"parent": {"database_id": dbid}, "properties": properties}
    try:
        r = requests.post(f"{NOTION_API}/pages", headers=notion_headers(), data=json.dumps(body), timeout=20)
        return {"ok": r.ok, "status": r.status_code, "text": (r.text or "")[:500]}
    except Exception as e:
        return {"ok": False, "reason": str(e)}

def notion_query(dbid:str, body:dict):
    if not (NOTION_TOKEN and dbid):
        return {"ok": False, "reason": "Notion token or DB id missing"}
    try:
        r = requests.post(f"{NOTION_API}/databases/{dbid}/query", headers=notion_headers(), data=json.dumps(body), timeout=30)
        data = r.json() if r.text else {}
        return {"ok": r.ok, "status": r.status_code, "data": data}
    except Exception as e:
        return {"ok": False, "reason": str(e)}

def queue_properties(command:str, status:str="Queued", result:str=""):
    return {
        "Command":   {"title": [{"text": {"content": command[:2000]}}]},
        "Status":    {"status": {"name": status}},
        "Timestamp": {"date": {"start": datetime.utcnow().isoformat()}},
        "Result":    {"rich_text": [{"text": {"content": result[:1900]}}]} if result else {"rich_text": []}
    }

def memory_properties(note:str, source:str="chat", tags:list=None):
    props = {
        "Note":      {"title": [{"text": {"content": note[:2000]}}]},
        "Source":    {"select": {"name": source}},
        "Timestamp": {"date": {"start": datetime.utcnow().isoformat()}},
    }
    if tags:
        props["Tags"] = {"multi_select": [{"name": t} for t in tags[:24]]}
    return props

# ===== Flask =====
app = Flask(__name__)

@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return resp

# ===== Fallback INTENT when API not available =====
def infer_intent(utterance:str)->dict:
    base = {
        "actor":"Codex","user":"Amar","timestamp": now_iso(),
        "utterance":utterance,"intent":"PLAN","project":"Codex","tool":"make",
        "payload":{"notes":"auto"}
    }
    u = (utterance or "").lower()
    if "store" in u or "product" in u: base["intent"], base["project"] = "EXEC","Ecom"
    if "ad" in u:                      base["intent"], base["project"] = "AD","Ecom"
    if any(k in u for k in ["pnl","profit","brief"]): base["intent"] = "PNL"
    if any(k in u for k in ["kingmaker","roulette"]): base["intent"], base["project"] = "KINGMAKER_BUILD","Kingmaker"
    return base

def forward_webhook(intent:str, payload:dict)->dict:
    url = WEBHOOKS.get(intent.upper(), "")
    if not url:
        return {"ok": False, "reason": f"No webhook configured for intent {intent}"}
    try:
        r = requests.post(url, json=payload, timeout=20)
        return {"ok": r.ok, "status": r.status_code, "text": (r.text or "")[:500]}
    except Exception as e:
        return {"ok": False, "reason": str(e)}

# ===== Conversational route (voice client posts here) =====
@app.route("/route", methods=["POST"])
def route():
    data = request.get_json(force=True) or {}
    utterance = (data.get("text") or "").strip()
    if not utterance:
        return jsonify({"ok": False, "error": "empty text"}), 400

    # ChatGPT answer
    if OPENAI_API_KEY and OPENAI_AVAILABLE:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            rsp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":SYSTEM_PROMPT},
                          {"role":"user","content": utterance}],
                temperature=0.4
            )
            answer = rsp.choices[0].message.content.strip()
            if NOTION_QUEUE_DBID:
                notion_create_row(NOTION_QUEUE_DBID, queue_properties(utterance, "Complete", answer))
            return jsonify({"ok": True, "reply": answer})
        except Exception as e:
            if NOTION_QUEUE_DBID:
                notion_create_row(NOTION_QUEUE_DBID, queue_properties(utterance, "Error", str(e)))
            contract = infer_intent(utterance)
            dispatch = forward_webhook(contract["intent"], contract)
            reply = f"Intent {contract['intent']} on {contract['project']} via {contract['tool']}"
            if not dispatch.get("ok"):
                reply += f" — note: {dispatch.get('reason','')}"
            return jsonify({"ok": False, "error": str(e), "reply": reply, "contract": contract, "dispatch": dispatch}), 500

    # No API available
    if NOTION_QUEUE_DBID:
        notion_create_row(NOTION_QUEUE_DBID, queue_properties(utterance, "Running", ""))
    contract = infer_intent(utterance)
    dispatch = forward_webhook(contract["intent"], contract)
    status = "Complete" if dispatch.get("ok") else "Error"
    if NOTION_QUEUE_DBID:
        notion_create_row(NOTION_QUEUE_DBID, queue_properties(utterance, status, dispatch.get("reason","")))
    reply = f"Intent {contract['intent']} on {contract['project']} via {contract['tool']}"
    return jsonify({"ok": True, "reply": reply, "contract": contract, "dispatch": dispatch})

# ===== Text command to queue (for typed control) =====
@app.route("/chatlog", methods=["POST"])
def chatlog():
    data = request.get_json(force=True) or {}
    command = (data.get("command") or "").strip()
    result  = (data.get("result") or "").strip()
    status  = (data.get("status") or "Queued").strip()
    if not command:
        return jsonify({"ok": False, "error": "empty command"}), 400
    res = notion_create_row(NOTION_QUEUE_DBID, queue_properties(command, status, result))
    return jsonify({"ok": res.get("ok", False), "notion": res})

# ===== Memory write/read =====
@app.route("/memory/log", methods=["POST"])
def memory_log():
    if not NOTION_MEMORY_DBID:
        return jsonify({"ok": False, "error": "NOTION_MEMORY_DBID not set"}), 400
    data = request.get_json(force=True) or {}
    note = (data.get("note") or "").strip()
    source = (data.get("source") or "chat").strip()
    tags = data.get("tags") or []
    if not note:
        return jsonify({"ok": False, "error": "empty note"}), 400
    res = notion_create_row(NOTION_MEMORY_DBID, memory_properties(note, source, tags))
    return jsonify({"ok": res.get("ok", False), "notion": res})

@app.route("/memory/recent", methods=["GET"])
def memory_recent():
    limit = int(request.args.get("limit", 50))
    sorts = [{"property":"Timestamp","direction":"descending"}]
    body = {"page_size": limit, "sorts": sorts}
    res = notion_query(NOTION_MEMORY_DBID, body)
    return jsonify(res)

# ===== Conversation logger (user/assistant -> Memory) =====
@app.route("/log/conversation", methods=["POST"])
def log_conversation():
    """
    Body: { "role": "user" | "assistant", "content": "text", "tags": ["..."] }
    Logs both prompts and replies to Codex Memory.
    """
    if not NOTION_MEMORY_DBID:
        return jsonify({"ok": False, "error": "NOTION_MEMORY_DBID not set"}), 400
    data = request.get_json(force=True) or {}
    role = (data.get("role") or "").strip().lower()
    content = (data.get("content") or "").strip()
    tags = data.get("tags") or []
    if role not in ("user","assistant") or not content:
        return jsonify({"ok": False, "error": "invalid role or empty content"}), 400
    source = "chat-user" if role=="user" else "chat-assistant"
    res = notion_create_row(NOTION_MEMORY_DBID, memory_properties(content, source, tags))
    return jsonify({"ok": res.get("ok", False), "notion": res})
from flask import send_from_directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/static/openapi.json", methods=["GET"])
def static_openapi():
    return send_from_directory(APP_DIR, "openapi.json", mimetype="application/json")
# ===== OpenAPI schema for Private GPT Actions =====
@app.route("/openapi.json", methods=["GET"])
def openapi():
    spec = {
        "openapi": "3.1.0",
        "info": {
            "title": "Codex Core Router API",
            "version": "1.0.0"
        },
        "servers": [
            {
                "url": "https://codex-core-router.onrender.com",
                "description": "Production server"
            }
        ],
        "paths": {
            "/memory/log": {
                "post": {
                    "operationId": "logMemoryNote",
                    "summary": "Log a memory note to Codex shared memory",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "note": {"type": "string"},
                                        "source": {"type": "string"},
                                        "tags": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "required": ["note"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "OK"}
                    }
                }
            },
            "/chatlog": {
                "post": {
                    "operationId": "queueCommand",
                    "summary": "Queue a command in the Codex Command Queue",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "command": {"type": "string"},
                                        "status": {"type": "string"}
                                    },
                                    "required": ["command"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "OK"}
                    }
                }
            },
            "/log/conversation": {
                "post": {
                    "operationId": "logConversationTurn",
                    "summary": "Log conversation turns (user or assistant) to memory",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "role": {
                                            "type": "string",
                                            "enum": ["user", "assistant"]
                                        },
                                        "content": {"type": "string"},
                                        "tags": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "required": ["role", "content"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "OK"}
                    }
                }
            }
        }
    }
    return jsonify(spec)

# ===== Summarizer: compress old memory into 'archive-summary' blocks =====
def summarize_old_memory(days_old:int=90, batch:int=200):
    if not (OPENAI_API_KEY and OPENAI_AVAILABLE and NOTION_MEMORY_DBID):
        return {"ok": False, "reason": "missing API key or memory DB"}
    cutoff = (datetime.utcnow() - timedelta(days=days_old)).isoformat()
    body = {
        "page_size": batch,
        "filter": {"and": [
            {"property":"Timestamp","date":{"before": cutoff}},
            {"property":"Tags","multi_select":{"does_not_contain":"archive-summary"}}
        ]},
        "sorts": [{"property":"Timestamp","direction":"ascending"}]
    }
    res = notion_query(NOTION_MEMORY_DBID, body)
    if not res.get("ok"): return res

    results = res["data"].get("results", [])
    if not results:
        return {"ok": True, "message": "nothing to summarize"}

    entries = []
    for p in results:
        props = p.get("properties", {})
        title = props.get("Note", {}).get("title", [])
        note_txt = "".join([t["plain_text"] for t in title]) if title else ""
        ts = props.get("Timestamp", {}).get("date", {}).get("start","")
        src = props.get("Source", {}).get("select", {}).get("name","")
        tags = [t.get("name","") for t in props.get("Tags", {}).get("multi_select",[])]
        entries.append(f"- [{ts}] ({src}) {note_txt}  tags={tags}")

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "Summarize the following knowledge into a compact, actionable brief.\n"
        "Include: key decisions, rules/laws, goals, metrics, open questions, and next actions.\n"
        "Be concise, make it skimmable. Then add a one-line TL;DR at the end.\n\n"
        + "\n".join(entries)
    )
    rsp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":"You compress long histories into concise, high-signal summaries."},
                  {"role":"user","content": prompt}],
        temperature=0.2
    )
    summary = rsp.choices[0].message.content.strip()
    props = memory_properties(summary, source="system", tags=["archive-summary"])
    return notion_create_row(NOTION_MEMORY_DBID, props)

def scheduler():
    try:
        # Warm-up quick health & memory check on boot
        _ = notion_query(NOTION_MEMORY_DBID, {"page_size": 3})
    except Exception:
        pass

    # Run a reflection shortly after startup (once)
    try:
        reflect_and_generate_rules(limit=50)
    except Exception:
        pass

    # Nightly loop (every 24h)
    while True:
        time.sleep(24 * 3600)
        try:
            reflect_and_generate_rules(limit=100)
        except Exception:
            pass

@app.route("/summarize/run", methods=["POST"])
def summarize_run():
    res = summarize_old_memory()
    return jsonify(res)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "ts": now_iso(),
        "webhooks": {k: bool(v) for k,v in WEBHOOKS.items()},
        "notion_queue": bool(NOTION_TOKEN and NOTION_QUEUE_DBID),
        "notion_memory": bool(NOTION_TOKEN and NOTION_MEMORY_DBID),
    })
# ===== Reflection: derive rules from recent memory =====
def reflect_and_generate_rules(limit:int=50):
    if not (OPENAI_API_KEY and OPENAI_AVAILABLE and NOTION_MEMORY_DBID and os.getenv("NOTION_RULES_DBID")):
        return {"ok": False, "reason": "missing configuration"}

    body = {"page_size": limit, "sorts": [{"property":"Timestamp","direction":"descending"}]}
    recent = notion_query(NOTION_MEMORY_DBID, body)
    if not recent.get("ok"):
        return recent

    entries = []
    for p in recent["data"].get("results", []):
        props = p.get("properties", {})
        title = props.get("Note", {}).get("title", [])
        note_txt = "".join([t["plain_text"] for t in title]) if title else ""
        entries.append(note_txt)

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "From the following conversation and operational notes, extract actionable rules or principles "
        "that would improve Codex’s reasoning, decision-making, or communication. "
        "Output each as: Rule, Rationale, and Category."
        "\n\n" + "\n".join(entries)
    )
    rsp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":"You extract meta-learning rules from operational data."},
                  {"role":"user","content":prompt}],
        temperature=0.3
    )
    rules_text = rsp.choices[0].message.content.strip()

            # Map to your Codex Rules schema
    props = {
        "Rule": {"title": [{"text": {"content": rules_text[:2000]}}]},
        "Rational": {"rich_text": [{"text": {"content": "Autogenerated from reflection over recent memory."}}]},
        "Category": {"select": {"name": "Autogenerated"}},
        "Source": {"select": {"name": "Reflection"}},
        "Date": {"date": {"start": datetime.utcnow().isoformat()}},
        "Scope": {"rich_text": []},
        "Trigger": {"rich_text": []},
        "Priority": {"select": {"name": "Medium"}},
        "Status": {"select": {"name": "Active"}},
        "Example": {"rich_text": []},
        "Version": {"rich_text": [{"text": {"content": "v1"}}]}
    }
    res = notion_create_row(os.getenv("NOTION_RULES_DBID"), props)
    return {"ok": True, "rules_logged": True, "notion": res}
@app.route("/reflect/run", methods=["POST"])
def reflect_run():
    data = request.get_json(force=True) or {}
    limit = int(data.get("limit", 50))
    return jsonify(reflect_and_generate_rules(limit))
# ===== Archive: log full transcripts =====
@app.route("/archive/log", methods=["POST"])
def archive_log():
    if not os.getenv("NOTION_ARCHIVE_DBID"):
        return jsonify({"ok": False, "error": "NOTION_ARCHIVE_DBID not set"}), 400
    data = request.get_json(force=True) or {}
    transcript = (data.get("transcript") or "").strip()
    source = (data.get("source") or "bridge").strip()
    tags = data.get("tags") or ["conversation"]
    if not transcript:
        return jsonify({"ok": False, "error": "empty transcript"}), 400

    props = {
        "Transcript": {"title": [{"text": {"content": transcript[:2000]}}]},
        "Source": {"select": {"name": source}},
        "Timestamp": {"date": {"start": datetime.utcnow().isoformat()}},
        "Tags": {"multi_select": [{"name": t} for t in tags[:8]]}
    }
    res = notion_create_row(os.getenv("NOTION_ARCHIVE_DBID"), props)
    return jsonify({"ok": res.get("ok", False), "notion": res})
# ===== Archive: log full transcripts =====
@app.route("/archive/log", methods=["POST"])
def archive_log():
    if not os.getenv("NOTION_ARCHIVE_DBID"):
        return jsonify({"ok": False, "error": "NOTION_ARCHIVE_DBID not set"}), 400
    data = request.get_json(force=True) or {}
    transcript = (data.get("transcript") or "").strip()
    source = (data.get("source") or "bridge").strip()
    tags = data.get("tags") or ["conversation"]
    if not transcript:
        return jsonify({"ok": False, "error": "empty transcript"}), 400

    props = {
        "Transcript": {"title": [{"text": {"content": transcript[:2000]}}]},
        "Source": {"select": {"name": source}},
        "Timestamp": {"date": {"start": datetime.utcnow().isoformat()}},
        "Tags": {"multi_select": [{"name": t} for t in tags[:8]]}
    }
    res = notion_create_row(os.getenv("NOTION_ARCHIVE_DBID"), props)
    return jsonify({"ok": res.get("ok", False), "notion": res})
if __name__ == "__main__":
    t = threading.Thread(target=scheduler, daemon=True)
    t.start()
    # For local runs; Render uses gunicorn start command
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))