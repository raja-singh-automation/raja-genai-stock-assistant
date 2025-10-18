import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="GenAI Stock Assistant",
    page_icon="📊",
    layout="wide"
)

# Inject CSS for modern UI
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .card {
        padding: 18px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .buy {color: green; font-weight: bold;}
    .hold {color: orange; font-weight: bold;}
    .sell {color: red; font-weight: bold;}
    .pos {color: green; font-weight: bold;}
    .neg {color: red; font-weight: bold;}
    .neu {color: gray; font-weight: bold;}
    h3 {margin-bottom: 8px;}
    </style>
""", unsafe_allow_html=True)

st.title("📊 GenAI Stock Assistant")
st.caption("AI-powered recommendations + News Sentiment + Earnings Chatbot (RAG)")

# -----------------------------
# Stock Data Functions
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="6mo")
        if hist.empty:
            return None

        # Technical indicators
        hist["RSI"] = ta.momentum.RSIIndicator(hist["Close"]).rsi()
        hist["SMA50"] = hist["Close"].rolling(50).mean()
        hist["SMA200"] = hist["Close"].rolling(200).mean()

        return {
            "Ticker": ticker,
            "Name": info.get("shortName"),
            "Sector": info.get("sector", "Others"),
            "Price": info.get("currentPrice"),
            "PE": info.get("trailingPE"),
            "EPS": info.get("trailingEps"),
            "RSI": float(hist["RSI"].iloc[-1]) if "RSI" in hist else None,
            "SMA50": float(hist["SMA50"].iloc[-1]) if "SMA50" in hist else None,
            "SMA200": float(hist["SMA200"].iloc[-1]) if "SMA200" in hist else None,
            "History": hist,
            "News": stock.news[:3] if stock.news else []
        }
    except Exception as e:
        st.warning(f"Data fetch failed for {ticker}: {e}")
        return None

def recommend(stock: dict):
    signals = []
    if stock.get("PE") and stock["PE"] < 20:
        signals.append("Good P/E")
    if stock.get("EPS") and stock["EPS"] > 0:
        signals.append("Positive EPS")
    if stock.get("RSI") and stock["RSI"] > 60:
        signals.append("Bullish RSI")
    if stock.get("SMA50") and stock.get("SMA200") and stock["SMA50"] > stock["SMA200"]:
        signals.append("Golden cross")

    score = len(signals)
    if score >= 3:
        return "BUY", signals
    elif score == 2:
        return "HOLD", signals
    else:
        return "SELL", signals

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    default_tickers = ["INFY.NS", "TCS.NS", "RELIANCE.NS", "YESBANK.NS", "SUZLON.NS"]
    tickers = st.text_area("Tickers (comma-separated; use .NS for NSE)",
                           value=",".join(default_tickers))
    tickers = [t.strip() for t in tickers.split(",") if t.strip()]

    price_min, price_max = st.slider("Price range (₹)", 1, 20000, (10, 5000), 1)
    show_charts = st.checkbox("Show candlestick charts", value=False)

    st.markdown("---")

    # ✅ Sector filter
    sector_filter = st.selectbox(
        "Filter by Sector",
        options=["All Sectors", "IT", "Banking", "Energy", "Pharma", "Auto", "Others"]
    )

    # ✅ Sorting
    sort_by = st.selectbox(
        "Sort stocks by",
        options=["Recommendation", "Price", "P/E", "EPS", "RSI"]
    )

tab1, tab2 = st.tabs(["📈 Recommendations", "🗂️ Chat with Earnings (RAG)"])

# -----------------------------
# Tab 1: Modern Recommendations + Sentiment
# -----------------------------
with tab1:
    st.subheader("✨ Stock Recommendations with News Sentiment")

    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)

    data_rows = [fetch_stock_data(t) for t in tickers]
    # ✅ Apply price filter
    data_rows = [d for d in data_rows if d and d.get("Price") and price_min <= d["Price"] <= price_max]

    # ✅ Apply sector filter
    if sector_filter != "All Sectors":
        data_rows = [d for d in data_rows if d.get("Sector") and sector_filter.lower() in d["Sector"].lower()]

    # ✅ Apply sorting
    if sort_by == "Price":
        data_rows = sorted(data_rows, key=lambda x: x.get("Price") or 0, reverse=True)
    elif sort_by == "P/E":
        data_rows = sorted(data_rows, key=lambda x: x.get("PE") or 9999)  # undervalued first
    elif sort_by == "EPS":
        data_rows = sorted(data_rows, key=lambda x: x.get("EPS") or 0, reverse=True)
    elif sort_by == "RSI":
        data_rows = sorted(data_rows, key=lambda x: x.get("RSI") or 0, reverse=True)
    elif sort_by == "Recommendation":
        order = {"BUY": 0, "HOLD": 1, "SELL": 2}
        data_rows = sorted(data_rows, key=lambda x: order[recommend(x)[0]])

    if not data_rows:
        st.info("No stocks found in this selection.")
    else:
        # ✅ Summary metrics
        buy_count = sum(recommend(stock)[0] == "BUY" for stock in data_rows)
        hold_count = sum(recommend(stock)[0] == "HOLD" for stock in data_rows)
        sell_count = sum(recommend(stock)[0] == "SELL" for stock in data_rows)

        c1, c2, c3 = st.columns(3)
        c1.metric("✅ BUY", buy_count)
        c2.metric("⚖️ HOLD", hold_count)
        c3.metric("❌ SELL", sell_count)

        st.markdown("---")

        # ✅ Stock cards grid with safe handling
        for i, stock in enumerate(data_rows):
            action, signals = recommend(stock)

            # Safe news extraction
            if stock.get("News") and len(stock["News"]) > 0:
                first_news = stock["News"][0]
                news_headline = first_news.get("title", "No title")
            else:
                news_headline = "No news found"

            # Sentiment safely
            try:
                sentiment = finbert(news_headline)[0]["label"].lower()
            except Exception:
                sentiment = "neutral"
            sentiment_html = {
                "positive": "<span class='pos'>Positive</span>",
                "negative": "<span class='neg'>Negative</span>",
                "neutral": "<span class='neu'>Neutral</span>",
            }.get(sentiment, sentiment)

            # Safe values
            price = stock.get("Price", "NA")
            pe = stock.get("PE", "NA")
            eps = stock.get("EPS", "NA")
            rsi = round(stock["RSI"], 2) if stock.get("RSI") else "NA"

            if i % 2 == 0:
                cols = st.columns(2)
            with cols[i % 2]:
                st.markdown(f"<div class='card'>"
                            f"<h3>{stock.get('Name','Unknown')} ({stock.get('Ticker','-')})</h3>"
                            f"<p>💰 Price: ₹{price}</p>"
                            f"<p>📊 P/E: {pe}, EPS: {eps}</p>"
                            f"<p>📈 RSI: {rsi}</p>"
                            f"<p>🔍 Signals: {', '.join(signals) if signals else 'No strong signals'}</p>"
                            f"<p>✅ Recommendation: <span class='{action.lower()}'>{action}</span></p>"
                            f"<p>📰 News: {news_headline}</p>"
                            f"<p>📌 Sentiment: {sentiment_html}</p>"
                            f"</div>", unsafe_allow_html=True)

                if show_charts and stock.get("History") is not None and not stock["History"].empty:
                    hist = stock["History"]
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'], high=hist['High'],
                        low=hist['Low'], close=hist['Close'],
                        name="Price"
                    ))
                    if "SMA50" in hist:
                        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color='blue', width=1), name="SMA50"))
                    if "SMA200" in hist:
                        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color='orange', width=1), name="SMA200"))
                    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab 2: Chat with Earnings (RAG)
# -----------------------------
with tab2:
    st.subheader("🗂️ Chat with Earnings Reports")

    if "rag_index" not in st.session_state:
        st.session_state.rag_index = None
        st.session_state.rag_docs = []
        st.session_state.embedder_name = "all-MiniLM-L6-v2"
        st.session_state.embedder = SentenceTransformer(
            st.session_state.embedder_name, device="cpu"
        )

    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    build = st.button("📚 Build Knowledge Base")

    if build and uploaded:
        texts, meta = [], []
        for up in uploaded:
            reader = PdfReader(up)
            for i, page in enumerate(reader.pages):
                t = (page.extract_text() or "").strip()
                if t:
                    CHUNK, OVERLAP = 800, 120
                    for start in range(0, len(t), CHUNK - OVERLAP):
                        chunk = t[start:start + CHUNK]
                        if len(chunk) > 100:
                            texts.append(chunk)
                            meta.append({"source": up.name, "page": i + 1})

        if texts:
            embeddings = st.session_state.embedder.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True
            )
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            st.session_state.rag_index = index
            st.session_state.rag_docs = [{"text": t, "meta": m} for t, m in zip(texts, meta)]
            st.success(f"Knowledge base built with {len(texts)} chunks.")
        else:
            st.error("No extractable text found.")

    if st.session_state.rag_index:
        query = st.text_input("Ask a question:")
        if st.button("🔎 Ask"):
            q_emb = st.session_state.embedder.encode([query],
                        normalize_embeddings=True, convert_to_numpy=True)
            D, I = st.session_state.rag_index.search(q_emb, 3)
            retrieved = [st.session_state.rag_docs[i] for i in I[0]]
            context = "\n\n".join([f"[{r['meta']['source']} p.{r['meta']['page']}] {r['text']}" for r in retrieved])
            
            qa = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
            
            prompt = f"Answer the question using only this CONTEXT:\n{context}\nQUESTION: {query}\nANSWER:"
            out = qa(prompt, max_length=256)[0]["generated_text"]
            st.write("### 💬 Answer")
            st.write(out)
            with st.expander("📑 Sources"):
                for r in retrieved:
                    st.markdown(f"- {r['meta']['source']} (page {r['meta']['page']})")
