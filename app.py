# Streamlit + yfinance + Hugging Face：股價與輿情分析儀表板
# -------------------------------------------------------------
# 使用方式：
# 1) 安裝套件：
#    pip install streamlit yfinance plotly pandas xlsxwriter transformers torch sentencepiece
# 2) 執行：
#    streamlit run app.py
# -------------------------------------------------------------

import datetime as dt
from typing import List, Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

# HF
from transformers import pipeline

st.set_page_config(page_title="Stock + Sentiment Dashboard", layout="wide")

# -----------------------------
# Sidebar：參數
# -----------------------------
st.sidebar.title("設定")
mode = st.sidebar.radio("模式", ["單一標的", "多標的比較"], index=0)

if mode == "多標的比較":
    tickers: List[str] = st.sidebar.text_input(
        "輸入代號 (逗號分隔)",
        value="AAPL, TSLA, NVDA",
        help="例如：AAPL, TSLA 或 ^TWII, 2330.TW (台積電)"
    ).replace(" ", "").split(",")
else:
    ticker = st.sidebar.text_input(
        "單一代號",
        value="AAPL",
        help="例如：AAPL、TSLA、NVDA、2330.TW、^GSPC"
    ).strip()
    tickers = [ticker]

# 日期區間
end_date = st.sidebar.date_input("結束日", dt.date.today())
start_default = end_date - dt.timedelta(days=365)
start_date = st.sidebar.date_input("開始日", start_default)

# K 線間隔
interval = st.sidebar.selectbox(
    "取樣頻率 (interval)",
    options=["1d", "1wk", "1mo"],
    index=0,
    help="日/週/月資料"
)

# 移動平均與圖表
ma_1 = st.sidebar.number_input("MA 窗口1", min_value=2, max_value=250, value=20)
ma_2 = st.sidebar.number_input("MA 窗口2", min_value=2, max_value=500, value=50)
show_volume = st.sidebar.checkbox("顯示成交量", value=True)

# 輿情（Hugging Face）
st.sidebar.markdown("---")
st.sidebar.subheader("輿情分析（Hugging Face）")
# 預設使用多語言三分類：negative / neutral / positive
# 其他可用：nlptown/bert-base-multilingual-uncased-sentiment (1~5星) 等
model_name = st.sidebar.text_input(
    "模型名稱",
    value="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    help="建議多語言模型以便中英並用"
)
max_items_news = st.sidebar.slider("分析最新新聞(筆)", 0, 50, 20, help="0 代表不抓新聞")
user_text = st.sidebar.text_area("或自行輸入待分析文字(每行一則)")


# -----------------------------
# 工具函式
# -----------------------------

@st.cache_data(ttl=3600)
def _valid_range(s: dt.date, e: dt.date) -> bool:
    return s < e

@st.cache_data(ttl=3600)
def load_prices(_tickers: List[str], start: dt.date, end: dt.date, interval: str, ma1: int, ma2: int) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for t in _tickers:
        try:
            df = yf.download(
                t, start=start, end=end + dt.timedelta(days=1), interval=interval,
                auto_adjust=True, progress=False
            )
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                df["MA1"] = df["Close"].rolling(ma1).mean()
                df["MA2"] = df["Close"].rolling(ma2).mean()
                out[t] = df
        except Exception as ex:
            st.warning(f"下載 {t} 失敗：{ex}")
    return out

@st.cache_data(ttl=900)
def fetch_news(_tickers: List[str], limit_per_ticker: int = 20) -> pd.DataFrame:
    rows = []
    for t in _tickers:
        try:
            news_list = yf.Ticker(t).news or []
            for item in news_list[:limit_per_ticker]:
                title = item.get("title", "")
                link = item.get("link", "")
                provider = item.get("publisher", "")
                ts = item.get("providerPublishTime")
                ts = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Taipei") if ts else None
                rows.append({"ticker": t, "time": ts, "provider": provider, "title": title, "link": link})
        except Exception as ex:
            st.warning(f"抓取 {t} 新聞失敗：{ex}")
    df = pd.DataFrame(rows).sort_values("time", ascending=False).reset_index(drop=True)
    return df

@st.cache_resource(show_spinner=False)
def load_pipeline(name: str):
    # device="cpu" 可在無 GPU 的環境下運行；若使用 CUDA：device=0
    try:
        return pipeline("sentiment-analysis", model=name, device=-1)
    except Exception as ex:
        st.error(f"載入模型失敗：{ex}")
        st.stop()

@st.cache_data(ttl=600)
def analyze_texts(texts: List[str], model_name: str) -> pd.DataFrame:
    if not texts:
        return pd.DataFrame(columns=["text", "label", "score"])
    clf = load_pipeline(model_name)
    results = clf(texts, truncation=True)
    # 統一欄位
    rows = []
    for t, r in zip(texts, results):
        label = r.get("label", "")
        score = float(r.get("score", 0.0))
        rows.append({"text": t, "label": label, "score": score})
    return pd.DataFrame(rows)


# -----------------------------
# 主區塊：標題
# -----------------------------

st.title("📈 股價走勢 + 🗞️ 輿情分析 儀表板")

if not _valid_range(start_date, end_date):
    st.sidebar.error("開始日需早於結束日")
    st.stop()

# 載入價格資料
data = load_prices(tickers, start_date, end_date, interval, ma_1, ma_2)
if not data:
    st.info("查無價格資料，請更換代號或時間區間。")
    st.stop()

# -----------------------------
# 價格視圖
# -----------------------------

if mode == "多標的比較":
    st.subheader("多標的走勢比較 (基準=100)")
    norm_df = pd.DataFrame()
    for t, df in data.items():
        if df.empty:
            continue
        series = (df["Close"] / df["Close"].iloc[0]) * 100
        norm_df[t] = series
    st.line_chart(norm_df)
else:
    t = tickers[0]
    df = data.get(t)
    st.subheader(f"{t} 價格走勢（{interval}）")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA1"], name=f"MA{ma_1}", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA2"], name=f"MA{ma_2}", mode="lines"))
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="日期", yaxis_title="價格")
    st.plotly_chart(fig, use_container_width=True)

    if show_volume:
        vol = go.Figure()
        vol.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
        vol.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(vol, use_container_width=True)

    # 指標卡片
    col1, col2, col3, col4 = st.columns(4)
    last_close = df["Close"].iloc[-1]
    ret = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100
    hi_ = df["High"].max()
    lo_ = df["Low"].min()
    vol_std = df["Close"].pct_change().std() * (252 ** 0.5) * 100
    col1.metric("最新收盤", f"{last_close:,.2f}")
    col2.metric("區間報酬(%)", f"{ret:,.2f}")
    col3.metric("區間最高", f"{hi_:,.2f}")
    col4.metric("區間最低", f"{lo_:,.2f}")
    st.caption(f"年化波動率(近似)：{vol_std:.2f}%")

    with st.expander("查看原始資料表 / 下載 CSV"):
        st.dataframe(df)
        csv = df.to_csv(index=True).encode("utf-8-sig")
        st.download_button(
            label="下載 CSV", data=csv,
            file_name=f"{t}_{start_date}_{end_date}_{interval}.csv", mime="text/csv"
        )

# -----------------------------
# 輿情分析區：來自 yfinance 新聞 + 自訂文字
# -----------------------------

st.markdown("---")
st.header("🗞️ 輿情分析 (Hugging Face)")

news_df = pd.DataFrame()
if max_items_news and max_items_news > 0:
    news_df = fetch_news(tickers, limit_per_ticker=max_items_news)
    if not news_df.empty:
        st.subheader("最新新聞標題")
        st.dataframe(news_df[["ticker", "time", "provider", "title"]])

# 待分析文字集合：新聞標題 + 使用者自訂
texts: List[str] = []
if not news_df.empty:
    texts.extend(news_df["title"].tolist())
if user_text.strip():
    texts.extend([line.strip() for line in user_text.splitlines() if line.strip()])

if texts:
    with st.spinner("模型分析中…"):
        df_pred = analyze_texts(texts, model_name)

    # 若是 1~5 星模型，將 label 正規化為 Negative/Neutral/Positive
    def normalize_label(label: str) -> str:
        l = label.lower()
        # 常見：'NEGATIVE'/'NEUTRAL'/'POSITIVE' 或 '1 star'...'5 stars'
        if "neg" in l or l.startswith("1") or l.startswith("2"):
            return "negative"
        if "neu" in l or l.startswith("3"):
            return "neutral"
        return "positive"

    df_pred["sentiment"] = df_pred["label"].map(normalize_label)

    st.subheader("情緒分佈")
    counts = df_pred["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig_bar = px.bar(counts, x="sentiment", y="count")
    fig_bar.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("逐則結果")
    show_cols = ["text", "label", "score", "sentiment"]
    st.dataframe(df_pred[show_cols])

    # 若有新聞資料，合併回來源表
    if not news_df.empty:
        merged = news_df.copy()
        merged = merged.join(df_pred[["label", "score", "sentiment"]], how="left")
        with st.expander("新聞情緒明細與下載"):
            st.dataframe(merged[["time", "ticker", "provider", "title", "label", "score", "sentiment"]])
            csv2 = merged.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下載新聞情緒 CSV", csv2, "news_sentiment.csv", "text/csv")
else:
    st.info("沒有可分析的文字。請在側欄輸入文字，或開啟『分析最新新聞』。")

# -----------------------------
# 批次下載（多標的）
# -----------------------------

with st.expander("批次下載（所有輸入的標的）"):
    if data:
        from io import BytesIO
        buff = BytesIO()
        with pd.ExcelWriter(buff, engine="xlsxwriter") as writer:
            for t, df in data.items():
                if df.empty:
                    continue
                df.to_excel(writer, sheet_name=t[:31])
        st.download_button(
            label="下載多標的 Excel",
            data=buff.getvalue(),
            file_name=f"prices_{start_date}_{end_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# -----------------------------
# 說明
# -----------------------------

st.markdown(
    """
    **說明**  
    - 資料來源：`yfinance`（Yahoo Finance）；新聞是以 `yf.Ticker(...).news` 抓取近況標題（有些標的可能無新聞）。  
    - 輿情分析：預設使用 `cardiffnlp/twitter-xlm-roberta-base-sentiment`（多語言 3 類別），可在側欄更換模型。  
    - 若你的環境有 GPU，可修改 `load_pipeline` 裡的 `device` 參數以加速推論。  
    - 企業應用建議：可將資料來源擴充為社群貼文、論壇、客服回饋，並以排程批次寫入資料庫，再由此 App 做儀表可視化。  
    """
)
