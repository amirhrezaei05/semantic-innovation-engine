from __future__ import annotations

import hashlib
import json
import os
import time
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



# Network behaviour for the LLM gateway. These are *per-phase* limits, not one big total.
# READ = max silence between two chunks of the response; with streaming it only has to cover the
# wait for the first token, not the whole generation.
GATEWAY_CONNECT_TIMEOUT_S = 15.0
GATEWAY_READ_TIMEOUT_S = 90.0


def _timeout(connect: float | None = None, read: float | None = None):
    # openai.Timeout exists in old (1.x) and new SDK versions, so no direct HTTP-library import is needed.
    from openai import Timeout

    c = GATEWAY_CONNECT_TIMEOUT_S if connect is None else connect
    r = GATEWAY_READ_TIMEOUT_S if read is None else read
    return Timeout(connect=c, read=r, write=15.0, pool=15.0)


def _clean_gateway_config(api_key, base_url, model_name):
    """Validate and normalise gateway settings before any network request."""
    api_key = (api_key or "").strip()
    base_url = (base_url or "").strip().rstrip("/")
    model_name = (model_name or "").strip()

    if not api_key:
        raise ValueError("API key is missing. Configure OPENAI_API_KEY in Streamlit Secrets.")
    if not base_url:
        raise ValueError("API base URL is missing. Configure OPENAI_BASE_URL.")
    if not model_name:
        raise ValueError("Model name is missing. Configure OPENAI_MODEL.")
    # ArvanCloud's documented OpenAI SDK example uses this key format.
    if not api_key.startswith("apikey "):
        raise ValueError(
            "The ArvanCloud API key should include its 'apikey ' prefix. "
            "Copy the complete key from your ArvanCloud AI gateway."
        )
    return api_key, base_url, model_name


def _make_client(api_key, base_url):
    from openai import OpenAI

    # max_retries=0: the SDK's silent retry used to re-send the whole request after a timeout,
    # doubling the wait (and the generation cost) with no feedback in the UI.
    return OpenAI(api_key=api_key, base_url=base_url, timeout=_timeout(), max_retries=0)


def _translate_openai_error(exc: Exception) -> Exception:
    """Map OpenAI-SDK exceptions to short, actionable messages."""
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        NotFoundError,
        RateLimitError,
    )

    if isinstance(exc, AuthenticationError):
        return RuntimeError("ArvanCloud authentication failed. Check the complete API key and its 'apikey ' prefix.")
    if isinstance(exc, NotFoundError):
        return RuntimeError("The requested model or gateway endpoint was not found. Check OPENAI_BASE_URL and OPENAI_MODEL.")
    if isinstance(exc, BadRequestError):
        return RuntimeError(
            "The provider rejected the request. Check the model name and supported request parameters. "
            f"Details: {exc}"
        )
    if isinstance(exc, RateLimitError):
        return RuntimeError("ArvanCloud rate limit or account quota reached. Check your gateway quota and try again later.")
    if isinstance(exc, APITimeoutError):  # must come before APIConnectionError (it is a subclass)
        return RuntimeError(
            "The gateway accepted the connection but sent nothing for "
            f"{GATEWAY_READ_TIMEOUT_S:.0f}s. The model may be overloaded or a slow 'reasoning' model. "
            "Use 'Test gateway connection' below, or try a faster model."
        )
    if isinstance(exc, APIConnectionError):
        return RuntimeError(
            "Could not connect to the ArvanCloud gateway. Check the base URL and network availability "
            "(the host running Streamlit must be able to reach it)."
        )
    if isinstance(exc, APIStatusError):
        return RuntimeError(f"Provider returned HTTP {exc.status_code}. Details: {exc}")
    return exc


def ping_gateway(api_key, base_url, model_name) -> float:
    """Tiny request that separates 'gateway unreachable / wrong config' from 'model is just slow'."""
    from openai import OpenAIError

    api_key, base_url, model_name = _clean_gateway_config(api_key, base_url, model_name)
    client = _make_client(api_key, base_url).with_options(timeout=_timeout(connect=20.0, read=20.0))
    started = time.time()
    try:
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=8,
        )
    except OpenAIError as exc:
        raise _translate_openai_error(exc) from exc
    return time.time() - started


def stream_synthesis(
    candidate,
    df,
    vectors,
    api_key,
    base_url,
    model_name,
    temperature,
    perspective,
):
    """Generator yielding the hypothesis text as the model produces it.

    Streaming means the person sees output immediately instead of a spinner for the whole
    generation, and the timeout only has to cover time-to-first-token / gaps between chunks.
    Use with ``st.write_stream``.
    """
    from openai import OpenAIError

    api_key, base_url, model_name = _clean_gateway_config(api_key, base_url, model_name)

    # Find the six nearest original ideas to the generated candidate.
    z = np.asarray(candidate["vector"], dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(z, vectors)[0]
    nearest = np.argsort(sims)[::-1][:6]

    anchors = []
    for idx in nearest:
        row = df.iloc[int(idx)]
        anchors.append({field: row[field] for field in FIELDS} | {"similarity": round(float(sims[idx]), 3)})

    prompt = f"""
You are an interdisciplinary scientific invention researcher.

Your task is to formulate one scientifically explicit research hypothesis
inspired by a mathematical semantic-search operation.

SEARCH OPERATION:
{candidate['method']}

INTELLECTUAL PERSPECTIVE:
{perspective}

NEAREST SOURCE IDEAS:
{json.dumps(anchors, ensure_ascii=False, indent=2)}

REQUIREMENTS:

1. Do not merely paraphrase, combine, or concatenate the source ideas.
2. Identify an underlying mechanism, mathematical principle, biological
   process, control strategy, or information-processing principle.
3. Transfer that principle into a meaningfully different target problem.
4. Explain the causal chain connecting the transferred mechanism
   to the proposed application.
5. Distinguish established scientific knowledge from speculation.
6. Explain why the proposed mechanism may be different from the source ideas.
7. Do not claim patentability, global novelty, clinical efficacy,
   or safety without evidence.
8. For healthcare applications, discuss privacy, human factors,
   clinical governance, and safety where relevant.
9. Make the proposal technically specific rather than generic.

RETURN THESE SECTIONS:

# Title
# Target Problem
# Transferred Mechanism
# Proposed Idea
# Why It May Be Different
# Scientific Basis
# Speculative Assumptions
# Implementation Steps (3-5 steps)
# Risks and Safeguards
# Validation Experiment
# Measurable Outcomes
# Novelty Verification Queries
"""

    client = _make_client(api_key, base_url)
    produced = False
    finish_reason = None
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Be rigorous, inventive, technically explicit, and transparent about scientific uncertainty.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
            max_tokens=3000,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            piece = getattr(choice.delta, "content", None)
            if piece:
                produced = True
                yield piece
            if choice.finish_reason:
                finish_reason = choice.finish_reason
    except OpenAIError as exc:
        raise _translate_openai_error(exc) from exc

    if not produced:
        raise RuntimeError(
            "The model returned an empty response. Try another model or check the gateway configuration."
        )
    if finish_reason == "length":
        yield "\n\n> **Note:** The response reached the token limit and may be incomplete."


def call_synthesis(candidate, df, vectors, api_key, base_url, model_name, temperature, perspective):
    """Non-streaming convenience wrapper (kept for backwards compatibility / tests)."""
    return "".join(
        stream_synthesis(candidate, df, vectors, api_key, base_url, model_name, temperature, perspective)
    ).strip()



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

        st.caption("Configure your ArvanCloud AI gateway.")
        
        base_url = st.text_input(
            "ArvanCloud gateway URL",
            config_value("OPENAI_BASE_URL", ""),
        )
        
        model_name = st.text_input(
            "Model identifier",
            config_value("OPENAI_MODEL", ""),
        )
        
        api_key = st.text_input(
            "ArvanCloud API key",
            config_value("OPENAI_API_KEY", ""),
            type="password",
            help="Enter the complete key, including the 'apikey ' prefix.",
        )
        
        temperature = st.slider(
            "Creativity temperature",
            0.0,
            1.5,
            0.9,
            0.05,
        )


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
            c_run, c_test = st.columns([1, 1])
            run_clicked = c_run.button("Synthesize scientific hypothesis", type="primary")
            test_clicked = c_test.button("Test gateway connection")

            if test_clicked:
                try:
                    with st.spinner("Sending a tiny test request..."):
                        seconds = ping_gateway(api_key, base_url, model_name)
                    st.success(f"Gateway reachable and model answered in {seconds:.1f}s.")
                except Exception as exc:
                    st.error(f"Gateway test failed: {exc}")

            generated_now = False
            if run_clicked:
                if not api_key and "api.openai.com" in base_url:
                    st.error("Provide an API key or configure an OpenAI-compatible provider.")
                else:
                    candidate = {"vector": population[selected], "method": st.session_state["population_metadata"][selected]["method"]}
                    started = time.time()
                    try:
                        st.caption("Waiting for the model... text appears as soon as the first tokens arrive.")
                        proposal = st.write_stream(
                            stream_synthesis(candidate, df, vectors, api_key, base_url, model_name, temperature, perspective)
                        )
                        st.session_state["last_proposal"] = str(proposal).strip()
                        st.caption(f"Generated in {time.time() - started:.1f}s")
                        generated_now = True
                    except Exception as exc:
                        st.error(f"Synthesis failed after {time.time() - started:.0f}s: {exc}")
            if st.session_state.get("last_proposal"):
                if not generated_now:  # when just generated, write_stream already displayed it
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
