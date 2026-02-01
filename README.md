
# 🛡️ Network Traffic Anomaly Detector (NTAD)

### 📝 Project Description
This project is an **AI-powered Security System** designed to protect networks. It works by capturing and analyzing network traffic in real-time. Using **Machine Learning**, the system "learns" the pattern of normal traffic and can instantly detect suspicious activities or cyber-attacks (like DDoS or Port Scanning) that deviate from this norm.

---

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/AI-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📌 Project Overview
**Network Traffic Anomaly Detector** is an intelligent Intrusion Detection System (IDS) designed to identify malicious network traffic using Machine Learning algorithms. It analyzes network flow data to distinguish between legitimate user behavior and cyber-attacks.

### 🎯 Key Goals
- Detect **Zero-day attacks** using anomaly detection.
- Classify attacks like **DDoS**, **Port Scanning**, and **Brute Force**.
- Provide a real-time visualization dashboard for network admins.

---

## 🛠️ Technologies Used

| Category | Tools & Libraries |
| :--- | :--- |
| **Language** | Python 3.x |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost, TensorFlow |
| **Packet Analysis** | Scapy, PyShark |
| **Visualization** | Streamlit, Matplotlib |

---

## 📂 Project Structure

```text
Network-Traffic-Analyzer/
├── data/                  # Dataset files (e.g., CIC-IDS2017)
├── notebooks/             # Jupyter Notebooks for EDA & Training
├── app.py                 # Dashboard application (Streamlit)
├── main.py                # Main Entry Point
├── file_module.py         # Data handling logic
├── live_module.py         # Real-time sniffing logic
└── requirements.txt       # Project dependencies

```

## 🚀 Getting Started

### 1. Prerequisites

Clone the repository and install the required libraries:

bash
git clone [https://github.com/iraj07/Network-Traffic-Analyzer.git](https://github.com/iraj07/Network-Traffic-Analyzer.git)
cd Network-Traffic-Analyzer
pip install -r requirements.txt



### 2. Running the Project

To start the analysis engine and process network data:

bash
python main.py



To launch the monitoring dashboard:

bash
streamlit run app.py





## 📊 Evaluation & Results

*Performance metrics on the test dataset:*

| Algorithm | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| **Random Forest** | 99.1% | 0.99 | 0.98 | 0.99 |
| **XGBoost** | 98.5% | 0.98 | 0.97 | 0.98 |
| **Autoencoder** | 96.0% | 0.95 | 0.96 | 0.95 |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

