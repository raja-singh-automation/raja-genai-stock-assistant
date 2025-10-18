# 📊 GenAI Stock Assistant

An **AI-powered Stock Recommendation Assistant** built with Streamlit.  

## 🚀 Features
- **📈 Stock Recommendations**  
  - Fetches live NSE/BSE stock data using `yfinance`  
  - Computes fundamentals (P/E, EPS) and technicals (RSI, SMA50, SMA200)  
  - Recommends **BUY / HOLD / SELL**  
- **📰 News Sentiment (FinBERT)**  
  - Analyzes latest stock headlines with FinBERT sentiment analysis  
- **🗂️ Chat with Earnings Reports (RAG)**  
  - Upload earnings report PDFs  
  - Ask questions, get answers grounded in documents  

## 🛠️ Tech Stack
- Streamlit  
- Hugging Face Transformers & Sentence Transformers  
- FAISS for semantic search  
- Plotly for charts  
- yfinance for market data  

## 🌐 Deployment
Deployed on **Streamlit Community Cloud**.  

Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
