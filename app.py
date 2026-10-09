from __future__ import annotations

import json
import os
import re
import base64
import gzip
import hashlib
from pathlib import Path

import streamlit as st
from rag_core import (
    DEFAULT_MIN_SCORE,
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_TOP_K,
    INDEX_FORMAT_VERSION,
    MAX_CONTEXT_CHARS,
    MAX_HISTORY_CHARS,
    MAX_HISTORY_MESSAGES,
    build_search_index,
    cached_pdf_chunks as core_cached_pdf_chunks,
    conversation_context,
    explicit_model_codes,
    extract_pdf_bytes,
    extractive_answer,
    generate_with_ollama,
    rebuild_documents,
    retrieve,
)


DATA_DIR = Path(__file__).parent / "data" / "manuals"
INDEX_PATH = Path(__file__).parent / "data" / "chunks.json"
PACKED_INDEX_DIR = Path(__file__).parent / "data" / "packed_chunks"
INGEST_CACHE_DIR = Path(__file__).parent / "data" / "manual_index_cache"
PRODUCTS_PATH = Path(__file__).parent / "data" / "laptops.json"
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_SESSION_UPLOAD_BYTES = 50 * 1024 * 1024
def _bounded_setting(name: str, default: float, lower: float, upper: float, integer: bool = False) -> float | int:
    try:
        value = float(os.getenv(name, str(default)))
        if not (lower <= value <= upper):
            value = default
    except (TypeError, ValueError):
        value = default
    return int(value) if integer else value


RETRIEVAL_TOP_K = _bounded_setting("RAG_TOP_K", DEFAULT_TOP_K, 1, 10, integer=True)
RETRIEVAL_MIN_SCORE = _bounded_setting("RAG_MIN_SCORE", DEFAULT_MIN_SCORE, 0, 100)
RETRIEVAL_BM25_K1 = _bounded_setting("RAG_BM25_K1", DEFAULT_BM25_K1, 0.1, 3)
RETRIEVAL_BM25_B = _bounded_setting("RAG_BM25_B", DEFAULT_BM25_B, 0, 1)


@st.cache_data
def load_products() -> list[dict]:
    if not PRODUCTS_PATH.exists():
        return []
    return json.loads(PRODUCTS_PATH.read_text(encoding="utf-8"))


def show_product_card(product: dict) -> None:
    with st.container(border=True):
        st.markdown(f"### {product['name']}")
        st.caption(product.get("sku", "Representative model"))
        st.markdown(
            f'<div class="product-photo"><img src="{product["image"]}" alt="{product["name"]}" loading="lazy"></div>',
            unsafe_allow_html=True,
        )
        specs = product.get("specs", [])
        if specs:
            st.markdown("  ".join(f"**{label}:** {value}  ·" for label, value in specs))
        if product.get("price"):
            st.markdown(f"**{product['price']}**  ·  {product.get('rating', 'Rating unavailable')}")
        st.caption(product.get("price_note", "Specifications vary by exact configuration. Check the listing for current availability and price."))
        st.link_button("View product listing", product["url"], use_container_width=True)


def product_specs_answer(question: str, model_filter: str, products: list[dict]) -> str | None:
    """Answer broad specifications questions from the structured product catalog."""
    if not re.search(r"\b(specs?|specifications?|configuration|configurations|hardware details)\b", question, re.I):
        return None

    question_lower = question.lower()
    requested_codes = set(re.findall(r"\b(?:x|m|e|s)\d{3,4}[a-z0-9]*\b", question_lower))
    # Users commonly call the E25362 family "Vivobook 16X" without knowing
    # the exact X1605 model code.
    if re.search(r"\b16\s*x\b", question_lower):
        requested_codes.add("x1605")

    candidates = []
    if requested_codes:
        candidates = [
            product for product in products
            if any(code in f"{product.get('family', '')} {product.get('sku', '')}".lower() for code in requested_codes)
        ]
    elif model_filter != "All indexed models":
        candidates = [product for product in products if product.get("family", "").lower() in model_filter.lower()]
        if re.search(r"\bvivobook\s+16\b", question_lower):
            candidates = [product for product in candidates if "vivobook 16" in product.get("name", "").lower()]
    elif re.search(r"\bvivobook\s+16\b", question_lower):
        candidates = [product for product in products if "vivobook 16" in product.get("name", "").lower()]

    if len(candidates) != 1:
        if candidates:
            return "I found more than one VivoBook 16 configuration. Choose the exact model family above (for example X1605 or M1605YA), then ask for its specifications."
        return None

    product = candidates[0]
    lines = [f"**{product['name']} — {product.get('sku', product.get('family', 'representative configuration'))}**"]
    lines.extend(f"- **{label}:** {value}" for label, value in product.get("specs", []))
    lines.append("These are representative or maximum listed specifications; exact hardware depends on the full SKU.")
    if product.get("url"):
        lines.append(f"[View model details]({product['url']})")
    return "\n\n".join(lines)


def extract_pdf(pdf_path: Path) -> list[dict]:
    return extract_pdf_bytes(pdf_path.read_bytes(), pdf_path.name)


def cached_pdf_chunks(pdf_path: Path) -> list[dict]:
    return core_cached_pdf_chunks(pdf_path, INGEST_CACHE_DIR)


def rebuild_index() -> list[dict]:
    # Root PDFs are the bundled local manuals; data/manuals is reserved for
    # explicitly installed static manuals. Browser uploads are session-only.
    pdf_paths = sorted(set(Path(__file__).parent.glob("ASUS*.pdf")) | set(DATA_DIR.glob("*.pdf")))
    chunks, failed_files, preserved = rebuild_documents(pdf_paths, INDEX_PATH, INGEST_CACHE_DIR)
    for failed_file in failed_files:
        st.error(f"Could not index {failed_file}. Check that it is a readable, unencrypted PDF.")
    if preserved and not pdf_paths:
        st.info("No local PDFs are available to rebuild. The existing bundled manual index was kept.")
    elif preserved:
        st.warning("No new passages were extracted; the existing manual index was preserved.")
    return chunks


@st.cache_data
def load_index() -> list[dict]:
    # Prefer the readable index, but fall back to packed chunks if it is absent,
    # empty, or malformed instead of presenting an empty app.
    if INDEX_PATH.exists():
        try:
            docs = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            if isinstance(docs, list) and docs:
                return docs
        except (json.JSONDecodeError, OSError):
            pass
    try:
        parts = sorted(PACKED_INDEX_DIR.glob("*.b64"))
        if parts:
            encoded = "".join(part.read_text(encoding="ascii").strip() for part in parts)
            compressed = base64.b64decode(encoded, validate=True)
            docs = json.loads(gzip.decompress(compressed).decode("utf-8"))
            return docs if isinstance(docs, list) else []
    except (json.JSONDecodeError, OSError, ValueError, gzip.BadGzipFile):
        return []
    return []


def search(query: str, docs: list[dict], limit: int = 5) -> list[tuple[float, dict]]:
    # The compiled BM25 index is private to this Streamlit session so uploaded
    # manuals never enter Streamlit's process-global cache.
    signature_input = json.dumps(
        {
            "index_version": INDEX_FORMAT_VERSION,
            "documents": [
                (d.get("source"), d.get("page"), d.get("model"), d.get("text"), d.get("section"))
                for d in docs
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    signature = hashlib.sha256(signature_input).hexdigest()
    cache = st.session_state.setdefault("_rag_search_indexes", {})
    if signature not in cache:
        cache[signature] = build_search_index(docs)
        while len(cache) > 3:
            cache.pop(next(iter(cache)))
    return retrieve(
        query,
        cache[signature],
        top_k=limit,
        min_score=RETRIEVAL_MIN_SCORE,
        bm25_k1=RETRIEVAL_BM25_K1,
        bm25_b=RETRIEVAL_BM25_B,
    )


def is_battery_capacity_question(question: str) -> bool:
    text = question.lower()
    return "battery" in text and any(term in text for term in ("capacity", "how much", "size", "watt hour", "watt-hour"))


def find_battery_capacity_specs(docs: list[dict]) -> list[tuple[str, dict]]:
    pattern = re.compile(r"\b(\d+(?:\.\d+)?)\s*(Wh|mAh|Ah)\b", re.I)
    found = []
    seen = set()
    for doc in docs:
        text = doc["text"]
        for match in pattern.finditer(text):
            context = text[max(0, match.start() - 100):match.end() + 100]
            if not re.search(r"battery|battery pack", context, re.I):
                continue
            value = f"{match.group(1)} {match.group(2)}"
            identity = (doc.get("model", doc["source"]), value.lower())
            if identity in seen:
                continue
            seen.add(identity)
            found.append((value, doc))
    return found


def is_adapter_power_question(question: str) -> bool:
    text = question.lower()
    asks_power = bool(re.search(r"\b(wattage|watts|power output|output power|how much power)\b", text))
    asks_adapter = bool(re.search(r"\b(chargers?|adapters?|charging)\b", text))
    return asks_power and asks_adapter


def find_adapter_power_specs(docs: list[dict]) -> list[dict]:
    found = []
    seen = set()
    for doc in docs:
        text = doc["text"]
        if not re.search(r"power adapter information", text, re.I):
            continue
        wattages = re.findall(r"\b(\d+(?:\.\d+)?)\s*W\b", text, re.I)
        if not wattages:
            continue
        identity = (doc.get("model", doc["source"]), doc.get("page"))
        if identity in seen:
            continue
        seen.add(identity)
        found.append(doc)
    return found


def adapter_power_answer(docs: list[dict], model: str) -> str:
    wattages = sorted({f"{number} W" for doc in docs for number in re.findall(
        r"\b(\d+(?:\.\d+)?)\s*W\b", doc["text"], re.I
    )}, key=lambda value: float(value.split()[0]))
    voltages = sorted({f"{number} V" for doc in docs for number in re.findall(
        r"Rating output voltage:\s*(\d+(?:\.\d+)?)\s*V", doc["text"], re.I
    )}, key=lambda value: float(value.split()[0]))
    rating_text = ", ".join(wattages)
    if voltages:
        rating_text += f"; listed output voltage: {', '.join(voltages)}"
    return (
        f"The {model} manual lists adapter output ratings of {rating_text}. "
        "The correct rating depends on the exact laptop configuration; check the label on the adapter supplied with your laptop."
    )


def unsupported_spec_answer(question: str, model: str) -> str | None:
    text = question.lower()
    requested = None
    if re.search(r"\b(processor|cpu|chipset)\b", text):
        requested = "the processor model"
    elif re.search(r"\b(ram|memory)\b", text):
        requested = "the installed memory configuration"
    elif re.search(r"\b(storage|ssd|hdd)\b", text):
        requested = "the storage configuration"
    elif re.search(r"\b(gpu|graphics card|graphics processor)\b", text):
        requested = "the graphics processor"
    elif re.search(r"\b(resolution|screen resolution)\b", text):
        requested = "the display resolution"
    elif re.search(r"\b(wattage|watts|charger output|power output)\b", text):
        requested = "the power adapter output rating"
    if not requested:
        return None
    return (
        f"The {model} manual does not list {requested}. VivoBook hardware can vary by exact configuration. "
        "Check the full model code on the laptop and its ASUS specifications page for that detail."
    )


def needs_exact_model(question: str) -> bool:
    return bool(re.search(
        r"\b(bios|post|boot|firmware|processor|cpu|chipset|ram|memory|storage|ssd|hdd|gpu|graphics|"
        r"resolution|battery capacity|wattage|charger output|power adapter|port|ports|usb|hdmi|"
        r"keyboard shortcut|function key|fn key|power button|shut down|shutdown|hibernate|sleep mode)\b",
        question,
        re.I,
    ))


def generate_answer(question: str, results: list[tuple[float, dict]]) -> str | None:
    try:
        host = (os.getenv("OLLAMA_HOST") or st.secrets.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL") or st.secrets.get("OLLAMA_MODEL") or "llama3.2"
    except Exception:
        host, model = os.getenv("OLLAMA_HOST", "http://localhost:11434"), os.getenv("OLLAMA_MODEL", "llama3.2")
    history = conversation_context(
        st.session_state.get("chat_history", [])[:-1],
        max_messages=MAX_HISTORY_MESSAGES,
        max_chars=MAX_HISTORY_CHARS,
    )
    try:
        timeout = float(os.getenv("OLLAMA_TIMEOUT_SEC", "3"))
        answer = generate_with_ollama(
            question, results, host=host, model=model, history=history,
            timeout=timeout, max_context_chars=MAX_CONTEXT_CHARS,
        )
    except (ValueError, TypeError):
        answer = None
    if answer is None:
        if not st.session_state.get("_ollama_failure_notified", False):
            st.toast("Local answer generation is unavailable; using retrieved manual text instead.", icon="ℹ️")
            st.session_state["_ollama_failure_notified"] = True
    return answer


def announce_model_change() -> None:
    selected = st.session_state.get("model_family_filter", "All indexed models")
    st.toast(f"Manual search switched to: {selected}")


def render_sources_button(source_docs: list[dict], key: str) -> None:
    """Keep source details hidden unless the user explicitly opens them."""
    if not source_docs:
        return
    visible_key = f"sources_visible_{key}"
    st.session_state.setdefault(visible_key, False)
    if st.button(
        "Hide sources" if st.session_state[visible_key] else "Sources",
        key=f"sources_toggle_{key}",
        type="secondary",
    ):
        st.session_state[visible_key] = not st.session_state[visible_key]
    if st.session_state[visible_key]:
        for rank, hit in enumerate(source_docs, start=1):
            refs = hit.get("source_refs") or [{
                "source": hit.get("source"), "page": hit.get("page"), "model": hit.get("model"),
                "section": hit.get("section"),
            }]
            for ref in refs:
                st.markdown(
                    f"**[{rank}] {ref.get('model') or ref.get('source')} · "
                    f"{ref.get('source')} — page {ref.get('page', '?')}**"
                )
            if hit.get("section"):
                st.caption(f"Section: {hit['section']}")
            st.write(hit["text"])


def append_chat_message(role: str, content: str, sources: list[dict] | None = None) -> None:
    message = {"role": role, "content": content}
    if sources:
        message["sources"] = sources
    st.session_state.chat_history.append(message)
    # Keep memory and prompts bounded while retaining a useful visible chat.
    st.session_state.chat_history = st.session_state.chat_history[-60:]


st.set_page_config(page_title="VivoBook Manual Assistant", page_icon="💻", layout="wide")
st.markdown("""
<style>
:root { --ink:#111827; --muted:#64748b; --blue:#2563eb; --blue-dark:#1d4ed8; --line:#e2e8f0; --pale:#eff6ff; --black:#111318; }
[data-testid="stAppViewContainer"] { background:#f5f7fa; color:var(--ink); }
[data-testid="stHeader"] { background:rgba(245,247,250,.94); }
[data-testid="stMain"] { background:#f5f7fa; }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label { color:var(--ink)!important; }
[data-testid="stMainBlockContainer"] { padding-top:1.5rem; max-width:1240px; }
h1,h2,h3 { color:var(--ink)!important; letter-spacing:-.025em; }
p, label, [data-testid="stCaptionContainer"] { color:var(--muted); }
.hero { position:relative; overflow:hidden; padding:2.15rem 2.4rem; margin:0 0 1.6rem; border:1px solid #262a33; border-radius:22px; background:radial-gradient(ellipse at 88% 5%,rgba(37,99,235,.26),transparent 38%),linear-gradient(125deg,#111318 0%,#191d25 72%,#17243a 100%); box-shadow:0 16px 40px rgba(15,23,42,.13); }
.hero:after { content:""; position:absolute; left:0; bottom:0; height:4px; width:100%; background:linear-gradient(90deg,#3b82f6 0%,#60a5fa 42%,transparent 88%); }
.hero-kicker { display:flex; align-items:center; gap:.55rem; color:#bfdbfe; font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
.hero-dot { width:8px; height:8px; border-radius:50%; background:#60a5fa; box-shadow:0 0 12px #60a5fa; }
.hero h1 { margin:.8rem 0 .45rem; color:#fff!important; font-size:clamp(2rem,4vw,3rem); line-height:1.08; font-weight:750; letter-spacing:-.045em; }
.hero-copy { max-width:710px; margin:0; color:#cbd5e1!important; font-size:1.03rem; line-height:1.7; }
.hero-tags { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.25rem; }
.hero-tags span { padding:.35rem .68rem; color:#e2e8f0; border:1px solid #3c4656; border-radius:999px; background:rgba(255,255,255,.045); font-size:.72rem; font-weight:600; letter-spacing:.035em; }
.st-key-product_catalog { padding:1.15rem 1.35rem 1.3rem; margin:1rem 0 1.5rem; border:1px solid #263a62; border-radius:20px; background:radial-gradient(ellipse at 90% 0%,rgba(37,99,235,.24),transparent 38%),linear-gradient(135deg,#081a3a 0%,#0c1220 58%,#101114 100%); box-shadow:0 14px 34px rgba(3,10,25,.14); }
.st-key-product_catalog h2,.st-key-product_catalog h3 { color:#f8fafc!important; }
.st-key-product_catalog p,.st-key-product_catalog label { color:#e2e8f0!important; }
.st-key-product_catalog [data-testid="stCaptionContainer"] { color:#a9bddb!important; }
.st-key-product_catalog [data-testid="stToggle"] label { color:#f8fafc!important; }
.st-key-product_catalog [data-testid="stToggle"] [role="switch"] { background:#64748b; }
.st-key-product_catalog [data-testid="stToggle"] [role="switch"][aria-checked="true"] { background:#3b82f6; }
.st-key-product_catalog [data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-color:#dbe5f1; border-radius:15px; }
.st-key-product_catalog [data-testid="stLinkButton"] a { background:#fff!important; color:#111827!important; border-color:#cbd5e1!important; }
.st-key-product_catalog [data-testid="stLinkButton"] * { color:#111827!important; }
[data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:18px; padding:1.05rem 1.25rem; margin:1rem 0; background:#fff; box-shadow:0 7px 22px rgba(15,23,42,.045); }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { color:var(--ink)!important; line-height:1.75; }
[class*="product-photo"] { height:220px; display:flex; align-items:center; justify-content:center; background:#f8fafc; border:1px solid var(--line); border-radius:14px; overflow:hidden; margin:.5rem 0 1rem; }
.product-photo img { width:100%; height:100%; object-fit:contain; padding:10px; }
[data-testid="stChatInput"] { background:#fff; border:1px solid #cbd5e1; border-radius:16px; box-shadow:0 8px 24px rgba(15,23,42,.07); }
[data-testid="stChatInput"]:focus-within { border-color:var(--blue); box-shadow:0 0 0 3px rgba(37,99,235,.12); }
[data-testid="stChatInput"] textarea { color:var(--ink)!important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { background:#fff; border-color:#cbd5e1; border-radius:11px; }
[data-testid="stSelectbox"] [data-baseweb="select"] * { color:#111827!important; }
[data-baseweb="menu"], [role="listbox"] { background:#fff!important; border:1px solid #cbd5e1!important; }
[role="option"] { color:#111827!important; background:#fff!important; }
[role="option"]:hover,[role="option"][aria-selected="true"] { color:#111827!important; background:#eff6ff!important; }
[data-testid="stToggle"] label,[data-testid="stCheckbox"] label,[data-testid="stRadio"] label { color:#1f2937!important; }
[data-testid="stFileUploaderDropzone"] { background:#f8fafc!important; border:1px dashed #94a3b8!important; }
[data-testid="stFileUploaderDropzone"] * { color:#1f2937!important; }
[data-testid="stFileUploaderDropzone"] button { color:#fff!important; background:#111318!important; border-color:#111318!important; }
[data-testid="stButton"] button { color:#111827!important; background:#fff!important; border:1px solid #cbd5e1!important; border-radius:10px!important; }
[data-testid="stButton"] button:hover { color:#123b7a!important; background:#eff6ff!important; border-color:#2563eb!important; }
button[kind="primary"] { color:#fff!important; background:var(--black)!important; border-color:var(--black)!important; border-radius:10px!important; font-weight:650!important; }
button[kind="primary"]:hover { background:var(--blue-dark)!important; border-color:var(--blue-dark)!important; }
button[kind="secondary"] { color:#1e293b!important; border-color:#cbd5e1!important; border-radius:10px!important; }
[data-testid="stFileUploaderDropzone"] button { color:#fff!important; background:#2563eb!important; border-color:#2563eb!important; }
[data-testid="stFileUploaderDropzone"] button:hover { color:#fff!important; background:#1d4ed8!important; border-color:#1d4ed8!important; }
[data-testid="stFileUploaderDropzone"] button svg { color:#fff!important; fill:#fff!important; }
[data-testid="stExpander"] { background:#fff; border:1px solid var(--line); border-radius:14px; }
[data-testid="stMetric"] { padding:.8rem 1rem; background:#fff; border:1px solid var(--line); border-radius:13px; }
[data-testid="stMetricLabel"] { color:var(--muted)!important; font-size:.75rem!important; text-transform:uppercase; letter-spacing:.07em; }
[data-testid="stMetricValue"] { color:var(--black)!important; font-weight:700; }
hr { border-color:var(--line); }
@media (max-width:700px) { .hero { padding:1.55rem 1.25rem; border-radius:17px; } .hero-copy { font-size:.94rem; } [data-testid="stMainBlockContainer"] { padding-top:1rem; } }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<section class="hero">
  <div class="hero-kicker"><span class="hero-dot"></span> ASUS VIVOBOOK · PRODUCT KNOWLEDGE</div>
  <h1>VivoBook Manual Assistant</h1>
  <p class="hero-copy">A model-aware support assistant for setup, hardware details, and troubleshooting—grounded in the included ASUS manuals.</p>
  <div class="hero-tags"><span>MANUAL-BASED ANSWERS</span><span>MODEL-AWARE SEARCH</span><span>OPTIONAL PAGE SOURCES</span></div>
</section>
""", unsafe_allow_html=True)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("session_manual_docs", [])
st.session_state.setdefault("session_manual_hashes", [])
st.session_state.setdefault("session_manual_bytes", 0)

with st.sidebar:
    st.header("Manual library")
    st.write("Search the included manual library or add an English PDF for another exact model.")
    uploads = st.file_uploader(
        "Add manuals (PDF)", type=["pdf"], accept_multiple_files=True,
        max_upload_size=25, help="PDF uploads are indexed only in your current session (25 MB per file, 50 MB total).",
    )
    st.caption("Session uploads are not saved to a shared folder or included in other users’ searches.")
    if uploads and st.button("Index manuals for this session", type="primary"):
        queued = [(Path(upload.name).name, upload.getvalue()) for upload in uploads]
        new_total = sum(len(data) for _, data in queued if hashlib.sha256(data).hexdigest() not in st.session_state.session_manual_hashes)
        if st.session_state.session_manual_bytes + new_total > MAX_SESSION_UPLOAD_BYTES:
            st.error("The session upload limit is 50 MB total. Remove some selected files and try again.")
        else:
            indexed_count = 0
            with st.spinner("Extracting pages and preparing session-only search entries…"):
                for safe_name, content in queued:
                    digest = hashlib.sha256(content).hexdigest()
                    if digest in st.session_state.session_manual_hashes:
                        continue
                    if len(content) > MAX_PDF_BYTES:
                        st.error(f"{safe_name} exceeds the 25 MB limit and was skipped.")
                        continue
                    try:
                        chunks = extract_pdf_bytes(content, safe_name)
                    except Exception:
                        st.error(f"{safe_name} could not be read. Use an unencrypted, text-based PDF.")
                        continue
                    if not chunks:
                        st.warning(f"{safe_name} contains no extractable text; OCR is not available in this app.")
                        continue
                    st.session_state.session_manual_docs.extend(chunks)
                    st.session_state.session_manual_hashes.append(digest)
                    st.session_state.session_manual_bytes += len(content)
                    indexed_count += len(chunks)
            st.session_state.pop("_rag_search_indexes", None)
            st.success(f"Added {indexed_count} passages to this session.")
    if st.session_state.session_manual_docs and st.button("Remove session manuals"):
        st.session_state.session_manual_docs = []
        st.session_state.session_manual_hashes = []
        st.session_state.session_manual_bytes = 0
        st.session_state.pop("_rag_search_indexes", None)
        st.rerun()
    if st.button("Rebuild search index"):
        with st.spinner("Rebuilding…"):
            count = len(rebuild_index())
        load_index.clear()
        st.session_state.pop("_rag_search_indexes", None)
        st.success(f"Indexed {count} text chunks.")
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()
    st.divider()
    st.caption("Answers are grounded in retrieved manual text. Open Sources under a reply to inspect the supporting pages.")

docs = load_index()
local_pdfs = list(Path(__file__).parent.glob("ASUS*.pdf")) + list(DATA_DIR.glob("*.pdf"))
if not docs and local_pdfs:
    with st.spinner("Building the search index from the included manuals…"):
        docs = rebuild_index()
    load_index.clear()
docs = docs + st.session_state.session_manual_docs
if not docs:
    st.info("No manuals indexed yet. Add an English VivoBook manual PDF in the sidebar to get started.")
    st.markdown("Example: **ASUS Vivobook 15 (X1504)** — see the official [ASUS support page](https://www.asus.com/supportonly/x1504za/helpdesk_manual/).")

if docs:
    available_models = sorted({doc.get("model", doc["source"]) for doc in docs})
    model_filter = st.selectbox(
        "Choose the manual family to search",
        ["All indexed models", *available_models],
        key="model_family_filter",
        on_change=announce_model_change,
        help="This limits retrieval to the selected family's manuals. Then ask a question below.",
    )
    active_docs = docs if model_filter == "All indexed models" else [
        doc for doc in docs if doc.get("model", doc["source"]) == model_filter
    ]
    with st.container(key="product_catalog"):
        st.subheader("Product catalog")
        if st.toggle("Browse model photos and buying details", value=False, key="show_product_catalog"):
            products = load_products()
            if products:
                if model_filter == "All indexed models":
                    visible_products = products
                    st.caption("Product photos, configurations, and marketplace details for the included model families.")
                else:
                    visible_products = [p for p in products if p["family"] in model_filter]
                    if not visible_products:
                        visible_products = [p for p in products if p.get("family") == "default"]
                    st.caption("Representative configuration for this manual family. Select the exact product code before comparing prices or specifications.")
                cols = st.columns(min(3, max(1, len(visible_products))))
                for index, product in enumerate(visible_products):
                    with cols[index % len(cols)]:
                        show_product_card(product)
else:
    active_docs = []

for message_index, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources_button(message["sources"], f"history_{message_index}")

question = st.chat_input("Ask about setup, charging, keyboard shortcuts, or troubleshooting…", disabled=not active_docs)
if question:
    append_chat_message("user", question)
    with st.chat_message("user"):
        st.write(question)

    results = search(question, active_docs)
    catalog_answer = product_specs_answer(question, model_filter, load_products())
    capacity_question = is_battery_capacity_question(question)
    capacity_specs = find_battery_capacity_specs(active_docs) if capacity_question else []
    adapter_power_question = is_adapter_power_question(question)
    adapter_specs = find_adapter_power_specs(active_docs) if adapter_power_question else []
    ask_for_model = (
        model_filter == "All indexed models"
        and not explicit_model_codes(question)
        and needs_exact_model(question)
        and not product_specs_answer(question, model_filter, load_products())
    )
    if ask_for_model:
        results = []
        answer_text = "Choose the exact VivoBook manual family above, or include its model code (for example X1504, X1605, or M1605YA), so I can avoid mixing instructions or specifications across models."
    elif capacity_question and capacity_specs:
        results = [(1.0, doc) for _, doc in capacity_specs]
    elif capacity_question:
        results = []
    if adapter_power_question:
        results = [(1.0, doc) for doc in adapter_specs]

    if capacity_question and not capacity_specs:
        answer_text = "The selected manual does not specify a numeric battery capacity (Wh or mAh). Check the ASUS specifications for your laptop's exact model number, since capacity can vary by configuration."
    elif capacity_question and capacity_specs:
        answer_text = "\n\n".join(
            f"{doc.get('model', doc['source'])}: {value}. [{rank}]"
            for rank, (value, doc) in enumerate(capacity_specs, start=1)
        )
    elif adapter_power_question and adapter_specs:
        answer_text = adapter_power_answer(adapter_specs, model_filter) + " " + "".join(
            f"[{rank}]" for rank in range(1, len(adapter_specs) + 1)
        )
    elif adapter_power_question:
        answer_text = "The selected manual does not state a numeric power-adapter wattage. Check the label on the adapter supplied with your exact laptop model."
    elif catalog_answer:
        results = []
        answer_text = catalog_answer
    elif not results:
        answer_text = unsupported_spec_answer(question, model_filter)
        if not answer_text:
            answer_text = "I couldn't find a relevant answer in this manual. Check that you selected the correct model family, or add the manual for your exact VivoBook model."
    else:
        with st.spinner("Searching the selected manual and preparing an answer…"):
            generated = generate_answer(question, results)
        answer_text = generated or extractive_answer(question, results)
        answer_text = answer_text.strip()

    source_docs = [hit for _, hit in results]
    with st.chat_message("assistant"):
        st.markdown(answer_text)
        render_sources_button(source_docs, "latest")
    append_chat_message("assistant", answer_text, source_docs)

