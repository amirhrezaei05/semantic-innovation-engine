from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from data_loader import resolve_dataset_path

from innovation_engine import (
    analyze_candidates,
    build_semantic_graph,
    generate_vector_population,
    geodesic_distance_matrix,
    sinkhorn_transport,
    spectral_coordinates,
    transport_pairs,
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
def config_value(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default)


MODEL_NAME = config_value("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
FIELDS = ["goal", "approach", "concept", "idea"]
FIELD_LABELS = {"goal": "هدف / Goal", "approach": "رویکرد / Approach", "concept": "مفهوم / Concept", "idea": "ایده / Idea"}
ALIASES = {
    "goal": ["هدف", "goal", "objective", "problem"],
    "approach": ["رویکرد", "کارکرد/رویکرد", "approach", "attitude", "strategy"],
    "concept": ["مفهوم", "concept", "mechanism", "category"],
    "idea": ["ایده", "ث", "idea", "description", "solution"],
}

st.set_page_config(page_title="Semantic Innovation Engine", page_icon="🧬", layout="wide")


def clean_cell(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in raw.columns:
        key = clean_cell(col).lower()
        for target, aliases in ALIASES.items():
            if key in [str(alias).strip().lower() for alias in aliases]:
                mapping[col] = target
                break
    out = pd.DataFrame(index=raw.index)
    for field in FIELDS:
        cols = [col for col, target in mapping.items() if target == field]
        out[field] = raw[cols[0]].map(clean_cell) if cols else ""
    if not out["idea"].astype(str).str.len().gt(0).any():
        candidates = [col for col in raw.columns if col not in mapping]
        if candidates:
            out["idea"] = raw[candidates[-1]].map(clean_cell)
    for field in FIELDS:
        if field not in out:
            out[field] = ""
    out["source_row"] = np.arange(2, len(out) + 2)
    out["duplicate_idea"] = out["idea"].str.casefold().duplicated(keep=False) & out["idea"].ne("")
    out["full_text"] = out.apply(
        lambda row: " | ".join(f"{FIELD_LABELS[field]}: {row[field]}" for field in FIELDS if row[field]), axis=1
    )
    return out[out["full_text"].str.len() > 0].reset_index(drop=True)


@st.cache_resource(show_spinner="Loading multilingual embedding model...")
def get_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_data(show_spinner="Encoding ideas...")
def encode_texts(texts: tuple[str, ...]) -> np.ndarray:
    return get_model().encode(list(texts), normalize_embeddings=True, show_progress_bar=False).astype("float32")


def encode_dataset(df: pd.DataFrame, mode: str, weights: dict[str, float]) -> np.ndarray:
    if mode == "Single-view: complete idea":
        return encode_texts(tuple(df["full_text"].tolist()))
    blocks = []
    total = sum(max(0.0, weights[field]) for field in FIELDS)
    if total <= 0:
        weights = {field: 1.0 for field in FIELDS}
        total = float(len(FIELDS))
    for field in FIELDS:
        texts = df[field].replace("", "(not supplied)").tolist()
        vectors = encode_texts(tuple(texts))
        blocks.append(vectors * np.sqrt(max(0.0, weights[field]) / total))
    combined = np.concatenate(blocks, axis=1)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    return combined / np.maximum(norms, 1e-10)


def load_builtin(name: str) -> pd.DataFrame:
    workbook = resolve_dataset_path(APP_DIR, name)
    return normalize_columns(pd.read_excel(workbook, engine="openpyxl"))


def call_synthesis(candidate, df, vectors, api_key, base_url, model_name, temperature, perspective):
    from openai import OpenAI
    z = np.asarray(candidate["vector"]).reshape(1, -1)
    sims = cosine_similarity(z, vectors)[0]
    nearest = np.argsort(sims)[::-1][:6]
    anchors = []
    for idx in nearest:
        row = df.iloc[int(idx)]
        anchors.append({field: row[field] for field in FIELDS} | {"similarity": round(float(sims[idx]), 3)})
    client = OpenAI(api_key=api_key or "local-compatible", base_url=base_url)
    prompt = f"""You are an interdisciplinary scientific invention researcher.
The task is to formulate one scientifically explicit candidate inspired by a mathematical search operation.

Search operation: {candidate['method']}
Perspective: {perspective}
Nearest source ideas (anchors, not templates):
{json.dumps(anchors, ensure_ascii=False, indent=2)}

Requirements:
- Do not merely paraphrase or concatenate source ideas.
- Transfer an underlying mechanism, mathematical principle, control strategy, biological process, or information-processing idea.
- Explain the causal/mechanistic chain connecting the source principle to the target problem.
- State what is established science versus speculative extrapolation.
- Do not claim patentability, global novelty, clinical efficacy, or safety without evidence.
- For healthcare ideas, include privacy, human-factors, clinical-governance, and safety considerations where relevant.

Return a structured response with: title, target_problem, transferred_mechanism, proposed_idea, why_it_may_be_different, scientific_basis, speculative_assumptions, implementation_steps (3-5), risks_and_safeguards, validation_experiment, measurable_outcomes, and novelty_verification_queries.
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": "Be rigorous, inventive, explicit about uncertainty, and avoid unsupported claims."},
                  {"role": "user", "content": prompt}],
        temperature=float(temperature),
        top_p=0.95,
    )
    return response.choices[0].message.content or "No content returned."


def main():
    st.title("🧬 Semantic Innovation Engine")
    st.caption("A mathematical laboratory for semantic geometry, cross-domain transfer, novelty search, and scientific hypothesis generation.")

    with st.sidebar:
        st.header("1 · Knowledge base")
        source_mode = st.radio("Dataset", ["Hand hygiene", "Antimicrobial stewardship", "Upload Excel"])
        if source_mode == "Upload Excel":
            uploaded = st.file_uploader("Upload an .xlsx workbook", type=["xlsx"])
            if uploaded is None:
                st.info("Upload a workbook to begin.")
                return
            df = normalize_columns(pd.read_excel(uploaded))
        else:
            try:
                df = load_builtin(source_mode)
            except FileNotFoundError as exc:
                st.error("A bundled dataset is missing from this deployment.")
                st.code(str(exc))
                st.info("In GitHub, upload the complete data/ folder beside app.py, including hand_hygiene.xlsx and antimicrobial_stewardship.xlsx, then reboot the Streamlit app.")
                st.stop()
        st.caption(f"{len(df)} source ideas loaded")
        if len(df) < 3:
            st.error("At least three non-empty source ideas are required for mathematical exploration.")
            return
        st.divider()
        st.header("2 · Semantic representation")
        view_mode = st.radio("Embedding geometry", ["Single-view: complete idea", "Multi-view: weighted fields"])
        weights = {}
        if view_mode.startswith("Multi-view"):
            st.caption("Weights control each field's contribution to the concatenated representation.")
            defaults = {"goal": 1.0, "approach": 1.0, "concept": 1.2, "idea": 1.3}
            for field in FIELDS:
                weights[field] = st.slider(FIELD_LABELS[field], 0.0, 2.0, defaults[field], 0.1)
        else:
            weights = {field: 1.0 for field in FIELDS}
        st.divider()
        st.header("3 · Discovery parameters")
        graph_k = st.slider("Semantic graph neighbors (k)", 2, min(12, max(2, len(df)-1)), min(6, max(2, len(df)-1)))
        population_size = st.slider("Evolutionary candidate population", 20, 300, 100, 10)
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)
        st.divider()
        st.header("4 · LLM synthesis (optional)")
        use_llm = st.checkbox("Enable scientific idea synthesis", value=False)
        base_url = st.text_input("OpenAI-compatible base URL", config_value("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        model_name = st.text_input("Chat model", config_value("OPENAI_MODEL", "gpt-4o-mini"))
        api_key = st.text_input("API key", config_value("OPENAI_API_KEY", ""), type="password")
        temperature = st.slider("Creativity temperature", 0.0, 1.5, 1.15, 0.05)

    with st.spinner("Constructing semantic representations..."):
        vectors = encode_dataset(df, view_mode, weights)
    dataset_hash = hashlib.sha256((view_mode + "|" + "|".join(df.full_text)).encode()).hexdigest()
    st.success(f"{len(df)} ideas represented in a {vectors.shape[1]}-dimensional semantic space · {MODEL_NAME}")

    tab_space, tab_transport, tab_evolution, tab_synthesis, tab_method = st.tabs([
        "Semantic Space", "Cross-domain Transport", "Evolution Lab", "Scientific Synthesis", "Mathematical Notes"
    ])

    with tab_space:
        st.subheader("Semantic graph and spectral geometry")
        graph = build_semantic_graph(vectors, k=graph_k)
        spectral = spectral_coordinates(graph, dimensions=2)
        geo = geodesic_distance_matrix(graph)
        coords = spectral["coordinates"]
        if coords.shape[1] < 2:
            coords = PCA(n_components=2, random_state=42).fit_transform(vectors)
        plot_df = pd.DataFrame({
            "x": coords[:, 0], "y": coords[:, 1],
            "Idea": df["idea"].replace("", "(idea text missing)"),
            "Approach": df["approach"], "Concept": df["concept"],
            "Graph component": graph["component_labels"].astype(str),
        })
        fig = px.scatter(plot_df, x="x", y="y", color="Graph component", hover_data=["Idea", "Approach", "Concept"],
                         title="Spectral embedding of the semantic k-nearest-neighbor graph")
        st.plotly_chart(fig, use_container_width=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Connected components", graph["components"])
        m2.metric("Graph edges", int(np.count_nonzero(graph["weights"]) // 2))
        m3.metric("Mean finite geodesic", f"{np.mean(geo[np.isfinite(geo)]):.3f}")
        m4.metric("Spectral dimensions", coords.shape[1])
        st.caption("Spectral coordinates are a graph-based visualization. Geodesic distances are shortest paths on the constructed kNN graph, not exact distances on a known semantic manifold.")
        with st.expander("Inspect graph spectrum"):
            eig = pd.DataFrame({"Eigenvalue index": np.arange(len(spectral["eigenvalues"])), "Eigenvalue": spectral["eigenvalues"]})
            st.line_chart(eig.set_index("Eigenvalue index"))
        with st.expander("Source ideas"):
            st.dataframe(df[["source_row", *FIELDS, "duplicate_idea"]], use_container_width=True, hide_index=True)

    with tab_transport:
        st.subheader("Mechanism-level cross-domain transfer")
        st.write("Optimal transport finds a low-cost coupling between two idea distributions. It identifies candidate correspondences; it does not prove that mechanisms are causally transferable.")
        if source_mode == "Upload Excel":
            st.info("For this tab, choose one of the two built-in datasets as the target domain, or use the uploaded workbook as the source domain.")
            target_name = st.selectbox("Target domain", ["Hand hygiene", "Antimicrobial stewardship"])
            target_df = load_builtin(target_name)
            target_vectors = encode_dataset(target_df, view_mode, weights)
            source_df, source_vectors = df, vectors
        else:
            other_name = "Antimicrobial stewardship" if source_mode == "Hand hygiene" else "Hand hygiene"
            target_df = load_builtin(other_name)
            target_vectors = encode_dataset(target_df, view_mode, weights)
            source_df, source_vectors = df, vectors
            st.caption(f"Source: {source_mode} · Target: {other_name}")
        reg = st.slider("Entropy regularization (ε)", 0.01, 0.5, 0.08, 0.01)
        if st.button("Compute optimal transport", type="primary", key="run_ot"):
            result = sinkhorn_transport(source_vectors, target_vectors, regularization=reg)
            st.session_state["ot_result"] = result
            st.session_state["ot_source_name"] = source_mode
            st.session_state["ot_target_name"] = other_name if source_mode != "Upload Excel" else target_name
        result = st.session_state.get("ot_result")
        if result is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Transport cost", f"{result['transport_cost']:.4f}")
            c2.metric("Source marginal error", f"{result['source_marginal_error']:.2e}")
            c3.metric("Target marginal error", f"{result['target_marginal_error']:.2e}")
            pairs = transport_pairs(result, top_n=15)
            pair_rows = []
            for pair in pairs:
                srow = source_df.iloc[pair["source_index"]]
                trow = target_df.iloc[pair["target_index"]]
                pair_rows.append({
                    "Source mechanism / idea": srow["concept"] or srow["idea"],
                    "Target mechanism / idea": trow["concept"] or trow["idea"],
                    "Transport mass": round(pair["transport_mass"], 5),
                    "Semantic cost": round(pair["semantic_cost"], 4),
                    "Source row": int(srow["source_row"]), "Target row": int(trow["source_row"]),
                })
            st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, hide_index=True)
            st.download_button("Download transport plan (.csv)", pd.DataFrame(result["plan"]).to_csv(index=False).encode("utf-8-sig"),
                               file_name="optimal_transport_plan.csv", mime="text/csv")

    with tab_evolution:
        st.subheader("Novelty search · MAP-Elites · Pareto selection · DPP")
        st.write("Generate a population through semantic crossover, direction transfer, and mutation. Then preserve candidates that balance novelty and relevance.")
        if st.button("Evolve candidate population", type="primary"):
            population, metadata = generate_vector_population(vectors, count=int(population_size), seed=int(seed))
            analysis = analyze_candidates(vectors, population)
            st.session_state["population"] = population
            st.session_state["population_metadata"] = metadata
            st.session_state["population_analysis"] = analysis
            st.session_state["population_hash"] = dataset_hash
        population = st.session_state.get("population")
        analysis = st.session_state.get("population_analysis")
        if population is not None and analysis is not None and st.session_state.get("population_hash") == dataset_hash:
            metadata = st.session_state["population_metadata"]
            results = pd.DataFrame({
                "Candidate": np.arange(1, len(population) + 1),
                "Operation": [m["method"] for m in metadata],
                "Novelty": analysis["novelty"],
                "Relevance": analysis["relevance"],
                "Quality heuristic": analysis["quality"],
            })
            pareto = set(analysis["pareto_indices"].tolist())
            elite = set(analysis["map_elites"].values())
            dpp = set(analysis["dpp_indices"])
            results["Pareto"] = [i in pareto for i in range(len(results))]
            results["MAP-Elites"] = [i in elite for i in range(len(results))]
            results["DPP"] = [i in dpp for i in range(len(results))]
            st.dataframe(results.sort_values("Quality heuristic", ascending=False), use_container_width=True, hide_index=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Pareto candidates", len(pareto))
            c2.metric("Occupied MAP-Elites cells", len(elite))
            c3.metric("DPP diverse shortlist", len(dpp))
            fig = px.scatter(results, x="Novelty", y="Relevance", color="Operation", hover_data=["Candidate", "Quality heuristic"],
                             title="Novelty–relevance objective space")
            st.plotly_chart(fig, use_container_width=True)
            shortlist = sorted(dpp)
            chosen = st.selectbox("Candidate to inspect", shortlist, format_func=lambda i: f"Candidate {i+1} · {metadata[i]['method']}")
            st.write("**Source ideas used:**")
            for idx in metadata[chosen]["source_indices"]:
                st.write(f"- {df.iloc[idx]['full_text']}")
            st.caption("Novelty is mean cosine distance to the source archive. Relevance is cosine alignment to the source centroid. These are geometric proxies, not validated innovation measurements.")
            export = results.copy()
            export_path = export.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Download evolutionary population (.csv)", export_path, "evolution_population.csv", "text/csv")
            st.session_state["selected_candidate_index"] = int(chosen)
        else:
            st.caption("Run evolution to populate the candidate archive.")

    with tab_synthesis:
        st.subheader("Scientific hypothesis reconstruction")
        population = st.session_state.get("population")
        analysis = st.session_state.get("population_analysis")
        population_is_current = st.session_state.get("population_hash") == dataset_hash
        if population is None or analysis is None or not population_is_current:
            st.info("First generate a population for the currently selected dataset and representation in the Evolution Lab.")
        elif not use_llm:
            st.info("Enable LLM synthesis in the sidebar to reconstruct a candidate as a structured scientific proposal.")
        else:
            dpp_ids = analysis["dpp_indices"]
            selected = st.selectbox("Candidate", dpp_ids, format_func=lambda i: f"Candidate {i+1} · novelty {analysis['novelty'][i]:.3f}")
            perspectives = ["Physics and control theory", "Molecular and cellular biology", "Information theory and computation",
                            "Evolutionary systems and game theory", "Hardware and human factors", "Radical assumption inversion"]
            perspective = st.selectbox("Intellectual perspective", perspectives)
            if st.button("Synthesize scientific hypothesis", type="primary"):
                if not api_key and "api.openai.com" in base_url:
                    st.error("Provide an API key or configure an OpenAI-compatible provider.")
                else:
                    candidate = {"vector": population[selected], "method": st.session_state["population_metadata"][selected]["method"]}
                    try:
                        with st.spinner("Constructing a mechanism-level hypothesis..."):
                            proposal = call_synthesis(candidate, df, vectors, api_key, base_url, model_name, temperature, perspective)
                        st.session_state["last_proposal"] = proposal
                    except Exception as exc:
                        st.error(f"Synthesis failed: {exc}")
            if st.session_state.get("last_proposal"):
                st.markdown(st.session_state["last_proposal"])
                st.download_button("Download hypothesis (.md)", st.session_state["last_proposal"], "scientific_hypothesis.md", "text/markdown")

    with tab_method:
        st.subheader("Mathematical methods and interpretation")
        st.markdown(r"""
**1. Multi-view representation.** For field embeddings \(e^{(k)}\), the representation is a weighted concatenation
\[
z_i=[\sqrt{w_1}e_i^{(1)};\ldots;\sqrt{w_m}e_i^{(m)}],\qquad \sum_k w_k=1.
\]

**2. Locally scaled semantic graph.** For k-nearest-neighbor edges,
\[
W_{ij}=\exp\left(-\frac{d_{ij}^2}{\sigma_i\sigma_j}\right),
\]
with symmetric weights. The normalized graph Laplacian is \(L=I-D^{-1/2}WD^{-1/2}\).

**3. Spectral coordinates and graph geodesics.** Eigenvectors of \(L\) provide a low-dimensional graph embedding. Shortest paths on edge costs \(-\log(W_{ij})\) approximate graph geodesics; they are not exact manifold geodesics.

**4. Entropic optimal transport.** We solve a balanced transport problem with uniform empirical masses and cosine cost, regularized by entropy. Sinkhorn iterations produce a coupling between source and target idea sets.

**5. Novelty search and quality-diversity.** Candidate novelty is mean distance to the k nearest archived ideas. MAP-Elites keeps the best candidate per novelty–relevance cell. Pareto selection preserves non-dominated candidates.

**6. DPP selection.** A quality-weighted cosine kernel encourages a shortlist that is both high-quality under the proxy and diverse. The implementation uses greedy log-determinant selection.

**Scientific caution.** Embedding distance is not scientific novelty. Semantic geometry depends on the encoder, language, field weights, dataset composition, and preprocessing. Generated candidates require literature review, expert assessment, and empirical validation.
""")
        st.warning("The current datasets contain 67 source records. This is enough to prototype the algorithms, not enough to learn a reliable universal model of innovation. Treat outputs as research hypotheses.")


if __name__ == "__main__":
    main()
