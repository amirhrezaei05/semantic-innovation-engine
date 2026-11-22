# Semantic Innovation Engine

A Streamlit research prototype for mathematically guided exploration and recombination of scientific ideas. It preserves the original Persian hand-hygiene and antimicrobial-stewardship datasets and adds semantic graph geometry, spectral analysis, graph-geodesic approximations, entropic optimal transport, evolutionary novelty search, MAP-Elites, Pareto selection, DPP diversity selection, and optional LLM-based scientific synthesis.

## Features

- Persian/English field normalization for Goal, Approach, Concept, and Idea.
- Multilingual Sentence Transformer embeddings.
- Single-view or weighted multi-view representations.
- Locally scaled k-nearest-neighbor semantic graph.
- Spectral graph coordinates and shortest-path graph-geodesic approximation.
- Entropic optimal transport between the two supplied idea domains.
- Candidate generation by semantic crossover, direction transfer, and mutation.
- Novelty/relevance analysis, Pareto front, MAP-Elites archive, and greedy DPP shortlist.
- Optional OpenAI-compatible LLM synthesis with scientific assumptions, risks, and validation plan.
- CSV export for transport plans and evolutionary populations.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Windows quick start

Run `run_windows.bat` from this folder, or follow the commands above manually. Python 3.11 is recommended.

## Deploy

See [DEPLOYMENT.md](DEPLOYMENT.md) for Streamlit Community Cloud deployment instructions.

## Scientific limitations

This is an exploratory research tool, not a validated innovation predictor. Embedding distance is not proof of scientific novelty. Optimal transport finds a mathematical coupling, not a causal mechanism transfer. Graph shortest paths approximate distances on the constructed graph, not exact geodesics on a known semantic manifold. The current source corpus contains only 67 records, so geometric estimates are sensitive to the encoder, language, field weights, and sampling. All generated hypotheses require literature review, domain-expert assessment, and empirical validation. Do not use the prototype for clinical decisions or upload identifiable patient data.
