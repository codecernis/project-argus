# Project Argus: Quantitative Market Analysis Pipeline

### Project Overview

Project Argus is a containerised, machine-learning-driven backtesting engine that watches Bitcoin like a hawk with trust issues. Built on a Walk-Forward Optimization (WFO) framework, it was created for educational purposes and personal research, because nothing says "fun weekend/CNY project" like teaching a computer to lose money more efficiently than you would yourself.

### Core Philosophy

Argus exists to solve the "Regime Change" problem. The cruel truth is that your beautifully-tuned 2017 model has no idea that the 2026 market is a completely different animal that eats 2017 models for breakfast. Argus combats this with a rolling training window, ensuring the model is perpetually catching up to current market chaos rather than confidently applying yesterday's wisdom to tomorrow's losses. Is it better to buy and hold instead of using Argus? Spoiler alert, yes.

### The Technology Stack

* **Language:** Python 3.11
* **Machine Learning:** Scikit-Learn (Random Forest Classifier, because if one bad decision tree is wrong, surely a whole forest of them will average out to something useful)
* **Technical Analysis:** `pandas-ta-classic` (RSI, EMA, Bollinger Bands, MACD, ATR a.k.a the full alphabet soup)
* **Infrastructure:** Docker & Docker Compose (so it fails consistently across all machines)
* **Data Source:** Yahoo Finance API (`yfinance`) (free, reliable, and appropriately humbling)

### System Architecture and Logic

1. Feature Engineering: Beyond staring at price charts and crying, Argus uses the Average True Range (ATR) to formally quantify market "chaos" and the MACD Histogram to gauge momentum. This allows the model to distinguish between a healthy trend and a violent fake-out — something human traders have been failing to do since 1637.
2. Walk-Forward Optimization: The system employs a 365-day rolling training window with a 30-day step size, essentially firing and rehiring its own brain every month. This prevents data leakage and ensures the model isn't still fighting the last war when the next one starts.
3. Risk-First Backtesting: The backtester dutifully accounts for 0.1% exchange fees and 0.2% round-trip slippage, because ignoring friction costs is how you go from "incredible backtest" to "deeply confusing live results." It then calculates institutional-grade metrics like the Sharpe Ratio and Maximum Drawdown to deliver a dose of realism directly to your optimism.

### Historical Performance (Simulation)

In a decade-long simulation on BTC-USD, two strategies went head-to-head:

* **Buy & Hold Strategy:** Achieved significantly higher absolute returns, but only if you were emotionally prepared to watch your portfolio crater 83% at some point and somehow not sell everything in a blind panic at 3am. Most people are not.
* **Argus Strategy:** Delivered a respectable ~1,000% return while keeping the maximum drawdown to a comparatively civilised 60%. You still lost more than half your money on paper at the worst point, but in crypto terms, that practically counts as stability.

The result is a lower Sharpe Ratio than a simple hold strategy, but a significantly smoother equity curve that is much easier on the human nervous system.

### Installation and Usage

To run the simulation in a fully isolated environment (so it can't touch anything important), make sure Docker is installed and run:

```bash
docker compose up --build

```

### Disclaimer

This project is strictly for research and entertainment purposes. It is emphatically not financial advice. If you choose to deploy a live trading bot based on these scripts, please be aware that past performance is not indicative of future results — particularly in an asset class that has, on multiple occasions, moved 20% in a single afternoon because someone famous posted a meme.
