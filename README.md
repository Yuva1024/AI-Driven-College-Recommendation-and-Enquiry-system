# 🎓 TNEA College Predictor

An intelligent web application that predicts the best engineering colleges for students based on their TNEA (Tamil Nadu Engineering Admissions) counselling data. Built with Flask, Machine Learning, and Google Gemini AI.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3+-green?logo=flask&logoColor=white)
![Gemini AI](https://img.shields.io/badge/Gemini-AI%20Powered-orange?logo=google&logoColor=white)

## ✨ Features

- **🔍 College Prediction** — ML-powered predictions based on rank, cutoff marks, category, branch preference, and district
- **📊 Filter & Rank Algorithm** — High-performance rule-based engine using historical cutoff data
- **💰 Fee Details** — Real-time fee information with annual tuition calculations
- **📈 Placement Info** — Placement statistics sourced from datasets, web scraping, and Gemini AI
- **🤖 AI Chatbot** — RAG-enabled chatbot powered by Google Gemini for college-related queries
- **👤 User Authentication** — Register, login, and track prediction history
- **📜 Prediction History** — View and manage past predictions in your dashboard

## 🛠️ Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| Backend         | Flask (Python)                      |
| ML Models       | Scikit-learn, Joblib                |
| AI / LLM        | Google Gemini API                   |
| Database        | SQLite + SQLAlchemy                 |
| Fuzzy Matching  | RapidFuzz                           |
| Web Scraping    | BeautifulSoup, Requests             |
| Chat History    | LangChain (SQLChatMessageHistory)   |
| Frontend        | Jinja2, Bootstrap 5                 |

## 📁 Project Structure

```
TNEA COLLEGE/
├── flaskappnew.py                         # Main Flask application
├── requirements.txt                       # Python dependencies
├── complete_engineering_colleges_dataset.csv  # College dataset (~2MB)
├── college_details_generated.csv          # Generated college details
├── init_db_run.py                         # Database initialization script
├── .env                                   # Environment variables (not tracked)
├── saved_models/                          # ML models (not tracked - too large)
│   ├── ensemble_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── target_encoder.pkl
│   └── feature_columns.pkl
├── templates/                             # HTML templates
│   ├── base.html                          # Base layout
│   ├── index.html                         # Prediction form + results
│   ├── result.html                        # Detailed results page
│   ├── chatbot.html                       # AI Chatbot interface
│   ├── dashboard.html                     # User dashboard
│   ├── history.html                       # Prediction history
│   ├── login.html                         # Login page
│   ├── register.html                      # Registration page
│   └── home.html                          # Landing page
└── tests/
    ├── test_fuzz.py / test_fuzz2.py       # Fuzzy matching tests
    ├── test_load.py                       # Load tests
    ├── test_routes.py                     # Route tests
    └── test_gemini_api.py                 # Gemini API tests
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Google Gemini API Key ([Get one here](https://aistudio.google.com/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/tnea-college-predictor.git
   cd tnea-college-predictor
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   FLASK_SECRET_KEY=your_secret_key_here
   ```

5. **Set up ML models**

   > ⚠️ The trained ML model files are not included in the repository due to size limits (~866 MB). You will need to train the models or obtain them separately.

6. **Run the application**
   ```bash
   python flaskappnew.py
   ```

   The app will be available at `http://localhost:5000`

## 📸 Screenshots

| Prediction Form | Results Page | AI Chatbot |
|:---:|:---:|:---:|
| Enter rank, cutoff & preferences | View matched colleges with probabilities | Ask college-related questions |

## 🔧 Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for AI features | Yes |
| `FLASK_SECRET_KEY` | Secret key for Flask sessions | Optional |

## 📄 License

This project is for educational purposes — built as a final year college project.

## 🙏 Acknowledgements

- [TNEA](https://www.tneaonline.org/) for admission data
- [Google Gemini](https://ai.google.dev/) for AI capabilities
- [Shiksha](https://www.shiksha.com/), [CollegeDunia](https://collegedunia.com/) for college data
