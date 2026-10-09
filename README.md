# VivoBook Manual Assistant

A local RAG chatbot for ASUS VivoBook manuals. It extracts pages from PDF manuals, splits them into overlapping passages, searches with BM25, and answers with supporting page references. If Ollama is running locally, the app can synthesize a conversational answer; otherwise it uses an extractive answer.

## Run it

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Deploy and share a link

This app can be hosted on [Streamlit Community Cloud](https://share.streamlit.io/). Upload the project to a GitHub repository, then create an app using `app.py` as the entrypoint. Keep `requirements.txt`, `.streamlit/config.toml`, `data/laptops.json`, and the PDFs in `data/manuals/` in the repository so the hosted app has the same product catalog and manual library. Do not upload `.venv/`, `logs/`, or secrets. After deployment, share the `*.streamlit.app` URL; the local `127.0.0.1` URL only works on the computer running the app.

The optional Ollama integration uses a local service and will not be available on Community Cloud unless you configure a separately hosted model endpoint. Manual retrieval and the extractive fallback work without Ollama.

The included English manuals cover these model families:

- E25357: VivoBook X1404, X1504, X1704
- E25362: VivoBook X1405, X1505, X1605
- E25361: VivoBook M1605YA
- E25372: VivoBook Go 15 E510
- E19281: VivoBook Flip 14 TP401
- E15273: VivoBook X412, X512
- E25354: VivoBook S14/S15/S16 (S5406/S5506/S5606 family)

Manuals are sourced from ASUS support downloads. The exact models covered are also shown in the model selector. ASUS has many VivoBook generations and hardware variants, so this collection cannot represent every VivoBook ever sold. Add an English manual PDF in the sidebar for a specific model family; keep its model number in the filename, then choose **Save and index manuals**.

## Optional local answer generation

Install [Ollama](https://ollama.com/), pull a model such as `llama3.2`, and start its local service. The app uses `http://localhost:11434` and model `llama3.2` by default. Override with `OLLAMA_HOST` and `OLLAMA_MODEL`. Without Ollama, retrieval and page citations still work.

## Notes

- This is retrieval augmented generation, not a newly trained foundation model.
- Use the manual matching the full model suffix printed on the laptop; features vary by configuration.
- Retrieval uses lexical BM25 and works without paid APIs or GPU setup.

