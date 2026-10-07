# 📄 ATS Resume Checker

A Streamlit app that scores a resume for Applicant Tracking System (ATS) compatibility and suggests concrete improvements, powered by Google's Gemini Flash model.

## Features

- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** for a tailored keyword match
- Overall **ATS score (0-100)** plus breakdown: formatting, keywords, experience, education, readability
- Strengths, weaknesses and **missing keywords**
- **Prioritised improvements** (High / Medium / Low) with example rewrites
- Download the report as JSON

## Project structure

```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Run locally

1. Get a free Gemini API key at <https://aistudio.google.com/apikey>.
2. Install and run:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

3. Provide your API key in one of three ways:
   - Paste it into the sidebar field when the app opens, **or**
   - Set an environment variable: `export GEMINI_API_KEY="your-key"` (Windows PowerShell: `$env:GEMINI_API_KEY="your-key"`), **or**
   - Create `.streamlit/secrets.toml` containing:
     ```toml
     GEMINI_API_KEY = "your-key"
     ```

> Never commit your API key or `secrets.toml` to GitHub.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (do **not** include secrets).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main` and main file `app.py`.
4. Open **Advanced settings** → **Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Configuration

The model list is in `app.py` (`FALLBACK_MODELS`). The default is `gemini-3.5-flash`; if your key does not have access to it, pick `gemini-2.5-flash` in the sidebar.

## Notes and limitations

- The score is an AI estimate based on ATS best practices, not the output of a real ATS. Use it as guidance.
- Scanned/image-only PDFs cannot be read (an ATS can't read them either). Export a text-based PDF or DOCX.
- Resume text is sent to the Gemini API for analysis; this app does not store it.

## Tech stack

Streamlit · google-genai (Gemini Flash) · pypdf · python-docx
