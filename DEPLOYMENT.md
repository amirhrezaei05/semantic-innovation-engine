# Deployment guide — Streamlit Community Cloud

## 1. Prepare the GitHub repository

1. Create a new **private or public** GitHub repository named `semantic-innovation-engine`.
2. Extract this project and upload the contents of this folder to the repository root (the repository root must contain `app.py`, `data_loader.py`, `innovation_engine.py`, `requirements.txt`, and the complete `data/` folder).
3. Verify that GitHub shows `data/hand_hygiene.xlsx` and `data/antimicrobial_stewardship.xlsx` in the repository. Also commit `data/README.md` so the folder is visible.
4. Commit and push the files.

**Important:** Do not upload only `app.py`. Streamlit runs from the GitHub repository; it cannot access Excel files that remain only on your computer. The app now displays an actionable error if either workbook is missing.

## 2. Deploy

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select **Create app**.
3. Choose repository `semantic-innovation-engine`, branch `main`, and main file path `app.py`.
4. Deploy. The first build downloads Python dependencies and the multilingual Sentence Transformer model; this can take several minutes.

## 3. Configure the optional LLM provider

The application works without an LLM provider. To enable scientific synthesis, open the app's **Advanced settings / Secrets** panel and add:

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
# Optional: EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

For an OpenAI-compatible provider, replace the base URL and model with the provider's documented values. Never commit secrets to GitHub.

## 4. Resource notes

- The app downloads `paraphrase-multilingual-MiniLM-L12-v2` on first startup and caches it during the running app process.
- The first cold start can be slow. If the free deployment instance runs out of memory, use a larger deployment tier or move embedding generation to a separate service.
- Streamlit Community Cloud is suitable for a prototype. For a persistent production deployment, add authentication, durable storage, observability, and a separate backend before handling sensitive or clinical data.

## 5. Local verification

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```
