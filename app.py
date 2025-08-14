import io
import re
import json
import textwrap
import streamlit as st

from typing import List, Tuple

# ---------- Optional imports for file parsing ----------
try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# URL fetching
try:
    import requests
    from bs4 import BeautifulSoup
    from readability import Document
except Exception:
    requests = None
    BeautifulSoup = None
    Document = None

from transformers import pipeline

st.set_page_config(page_title="NLP Toolkit (HF Pipelines) + URL", layout="wide")

st.title("🤗 NLP Toolkit — Transformers Pipelines")
st.caption("情緒分析、問答、摘要、命名實體辨識、零樣本分類｜支援上傳 .txt/.pdf/.docx、貼上文字、與抓取網頁內容（URL）")

# --------------------- Helpers ---------------------
def read_text_from_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        if PyPDF2 is None:
            st.error("未安裝 PyPDF2，請在 requirements.txt 加入 `PyPDF2`。")
            return ""
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
            return "\\n".join(pages)
        except Exception as e:
            st.error(f"PDF 解析失敗：{e}")
            return ""
    if name.endswith(".docx"):
        if docx2txt is None:
            st.error("未安裝 docx2txt，請在 requirements.txt 加入 `docx2txt`。")
            return ""
        try:
            content = uploaded_file.getvalue()
            with open("tmp_docx.docx", "wb") as f:
                f.write(content)
            text = docx2txt.process("tmp_docx.docx") or ""
            return text
        except Exception as e:
            st.error(f"DOCX 解析失敗：{e}")
            return ""
    st.warning("不支援的檔案格式，請上傳 .txt / .pdf / .docx")
    return ""


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 100) -> List[str]:
    text = re.sub(r"\\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        # try to cut on punctuation
        if end < n:
            cut = text.rfind(".", start, end)
            cut = max(cut, text.rfind("。", start, end))
            cut = max(cut, text.rfind("！", start, end))
            cut = max(cut, text.rfind("？", start, end))
            if cut != -1 and cut > start + max_chars * 0.6:
                end = cut + 1
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end)
    return [c for c in chunks if c]


@st.cache_resource(show_spinner=False)
def get_pipeline(task: str, model_id: str = None, tokenizer_id: str = None, device_map: str = "auto"):
    kwargs = {}
    if model_id:
        kwargs["model"] = model_id
    if tokenizer_id:
        kwargs["tokenizer"] = tokenizer_id
    try:
        return pipeline(task, device_map=device_map, **kwargs)
    except TypeError:
        return pipeline(task, **kwargs)


def show_json_download(data, label="下載JSON", filename="result.json"):
    j = json.dumps(data, ensure_ascii=False, indent=2)
    st.download_button(label, j, file_name=filename, mime="application/json")


def is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s.strip(), re.I))


def fetch_url_to_text(url: str, timeout: int = 15) -> Tuple[str, str]:
    "\"\"\"Fetch a URL and return (title, text). Requires requests, BeautifulSoup, readability-lxml.\"\"\""
    if requests is None or BeautifulSoup is None or Document is None:
        raise RuntimeError("缺少 requests/bs4/readability-lxml，請在 requirements.txt 安裝。")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    # Use readability to isolate main content
    doc = Document(r.text)
    title = doc.short_title() or ""
    content_html = doc.summary()
    soup = BeautifulSoup(content_html, "lxml")
    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\\n", strip=True)

    # Fallback if text too short
    if len(text) < 200:
        soup2 = BeautifulSoup(r.text, "lxml")
        for tag in soup2(["script", "style", "noscript"]):
            tag.decompose()
        text2 = soup2.get_text("\\n", strip=True)
        if len(text2) > len(text):
            text = text2
    return title, text


# --------------------- Sidebar ---------------------
st.sidebar.header("⚙️ 設定")

task = st.sidebar.selectbox(
    "選擇任務 (Task)",
    ["Sentiment Analysis", "Question Answering", "Summarization", "Named Entity Recognition", "Zero-shot Classification"],
    index=0
)

lang = st.sidebar.radio("語言/模型選擇", ["中文優先", "英文優先", "多語/通用"], index=0)

# Preselect model IDs
model_map = {
    "Sentiment Analysis": {
        "中文優先": ("uer/roberta-base-finetuned-jd-binary-chinese", None),
        "英文優先": (None, None),
        "多語/通用": ("cardiffnlp/twitter-xlm-roberta-base-sentiment", None),
    },
    "Question Answering": {
        "中文優先": ("uer/roberta-base-chinese-extractive-qa", None),
        "英文優先": (None, None),
        "多語/通用": ("deepset/xlm-roberta-base-squad2", None),
    },
    "Summarization": {
        "中文優先": ("csebuetnlp/mT5_multilingual_XLSum", None),
        "英文優先": ("facebook/bart-large-cnn", None),
        "多語/通用": ("csebuetnlp/mT5_multilingual_XLSum", None),
    },
    "Named Entity Recognition": {
        "中文優先": ("ckiplab/bert-base-chinese-ner", "ckiplab/bert-base-chinese-ner"),
        "英文優先": ("dslim/bert-base-NER", None),
        "多語/通用": ("Davlan/xlm-roberta-base-ner-hrl", None),
    },
    "Zero-shot Classification": {
        "中文優先": ("joeddav/xlm-roberta-large-xnli", None),
        "英文優先": ("facebook/bart-large-mnli", None),
        "多語/通用": ("joeddav/xlm-roberta-large-xnli", None),
    },
}

model_id, tokenizer_id = model_map[task][lang]
st.sidebar.write("**Model:**", model_id or "(default)")
if tokenizer_id:
    st.sidebar.write("**Tokenizer:**", tokenizer_id)

max_new_tokens = st.sidebar.slider("產生/摘要長度（最大字元估計）", 32, 2048, 256) if task == "Summarization" else None

# --------------------- Inputs ---------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    uploaded = st.file_uploader("📁 上傳檔案（.txt / .pdf / .docx）", type=["txt", "pdf", "docx"])
    pasted = st.text_area("或貼上文字 / context", height=160, placeholder="在此貼上長文、文章或問答 context…")
    url = st.text_input("或輸入網址(URL) 抓取內容")
    fetch_btn = st.button("🔎 抓取網址")
    if fetch_btn:
        if not url or not is_url(url):
            st.warning("請輸入有效的 http/https 網址")
        else:
            try:
                title, txt = fetch_url_to_text(url)
                st.session_state["fetched_title"] = title
                st.session_state["fetched_text"] = txt
                st.success(f"已抓取：{title or url}")
            except Exception as e:
                st.error(f"抓取失敗：{e}")

with col_right:
    if task == "Question Answering":
        question = st.text_input("問題 (Question)", value="台灣的首都是哪裡？")
    elif task == "Zero-shot Classification":
        labels_text = st.text_input("候選標籤（以逗號分隔）", value="正面, 負面, 中立, 價格, 服務, 環境")
        multi_label = st.checkbox("多標籤 (一段文字可能屬於多個標籤)", value=True)
    else:
        st.empty()

# Preview fetched content
if "fetched_text" in st.session_state and st.session_state.get("fetched_text"):
    with st.expander("已抓取的內容預覽", expanded=False):
        st.markdown(f"**標題：** {st.session_state.get('fetched_title','(無)')}")
        st.text_area("內容", st.session_state["fetched_text"][:4000], height=200)

# Combine sources
source_text = ""
if uploaded is not None:
    source_text = read_text_from_file(uploaded)
if pasted.strip():
    joiner = "\\n\\n" if source_text else ""
    source_text = (source_text + joiner + pasted.strip())
if st.session_state.get("fetched_text"):
    joiner = "\\n\\n" if source_text else ""
    source_text = (source_text + joiner + st.session_state["fetched_text"])

# --------------------- Run ---------------------
if st.button("🚀 執行"):
    if task in ["Question Answering", "Summarization", "Named Entity Recognition", "Zero-shot Classification"] and not source_text.strip():
        st.warning("請先提供文章/內容來源（上傳 / 貼上 / 抓取網址）。")
        st.stop()

    if task == "Sentiment Analysis" and not (source_text.strip() or pasted.strip() or uploaded or st.session_state.get("fetched_text")):
        st.warning("請提供文字（可上傳檔案、貼上或抓取網址）。")
        st.stop()

    # Build pipeline
    if task == "Sentiment Analysis":
        clf = get_pipeline("sentiment-analysis", model_id, tokenizer_id)
        units = [u.strip() for u in source_text.splitlines() if u.strip()]
        if not units:
            st.warning("未取得可分析的句子，請確認內容。")
            st.stop()
        with st.spinner("分析中…"):
            results = clf(units, truncation=True)
        st.subheader("結果")
        for u, r in zip(units, results):
            st.write(f"- **{r['label']}** ({r.get('score', 0):.3f}) — {u}")
        out = [{"text": u, **r} for u, r in zip(units, results)]
        show_json_download(out, "下載結果 JSON", "sentiment_results.json")

    elif task == "Question Answering":
        qa = get_pipeline("question-answering", model_id, tokenizer_id)
        q = question.strip() if 'question' in locals() else ""
        if not q:
            st.warning("請輸入問題文字。")
            st.stop()
        with st.spinner("回答中…"):
            ans = qa(question=q, context=source_text, handle_impossible_answer=True, max_answer_len=64)
        st.subheader("答案")
        st.markdown(f"**Answer:** {ans.get('answer', '')}  \\n**Score:** {ans.get('score', 0):.3f}  \\n**Start-End:** {ans.get('start')} - {ans.get('end')}")
        show_json_download(ans, "下載結果 JSON", "qa_result.json")

    elif task == "Summarization":
        summarizer = get_pipeline("summarization", model_id, tokenizer_id)
        chunks = chunk_text(source_text, max_chars=1500, overlap=50)
        if not chunks:
            st.warning("無法切分內容，請確認文字來源。")
            st.stop()
        with st.spinner(f"摘要中（共 {len(chunks)} 塊）…"):
            partial = []
            for i, ch in enumerate(chunks, 1):
                res = summarizer(ch, max_length=min(512, max_new_tokens or 256), min_length=16, do_sample=False)
                partial.append(res[0]["summary_text"])
        final_summary = " ".join(partial)
        st.subheader("摘要")
        st.write(final_summary)
        show_json_download({"summary": final_summary, "chunks": partial}, "下載結果 JSON", "summary.json")

    elif task == "Named Entity Recognition":
        ner_pipe = get_pipeline("ner", model_id, tokenizer_id)
        with st.spinner("辨識中…"):
            ents = ner_pipe(source_text, grouped_entities=True, aggregation_strategy="simple")
        st.subheader("命名實體")
        if not ents:
            st.info("沒有偵測到實體。")
        else:
            for e in ents:
                label = e.get("entity_group") or e.get("entity", "")
                score = e.get("score", 0.0)
                word = e.get("word", "")
                st.write(f"- **{label}** ({score:.3f}): {word}")
        show_json_download(ents, "下載結果 JSON", "ner.json")

    elif task == "Zero-shot Classification":
        zsc = get_pipeline("zero-shot-classification", model_id, tokenizer_id)
        candidate_labels = [s.strip() for s in labels_text.split(",") if s.strip()] if 'labels_text' in locals() else []
        if not candidate_labels:
            st.warning("請提供至少一個標籤（以逗號分隔）。")
            st.stop()
        with st.spinner("分類中…"):
            res = zsc(source_text, candidate_labels=candidate_labels, multi_label=bool(multi_label))
        st.subheader("分類結果")
        if isinstance(res, dict) and "labels" in res:
            st.write("**labels** :", res["labels"])
            st.write("**scores** :", [round(x, 3) for x in res["scores"]])
            show_json_download(res, "下載結果 JSON", "zero_shot.json")
        else:
            st.write(res)
            show_json_download(res, "下載結果 JSON", "zero_shot.json")

st.markdown("---")
st.caption("Tips: 若要使用 GPU，請安裝對應的 PyTorch CUDA 版本；如遇到網址限制（需要登入/防爬），可先手動複製內文貼上使用。")
