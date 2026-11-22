"""Mathematical discovery primitives for Semantic Innovation Lab.

These algorithms are exploration heuristics. They do not certify scientific novelty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.linalg import eigh
from sklearn.metrics.pairwise import cosine_similarity

EPS = 1e-10


def _rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError("Expected a 1D or 2D embedding array.")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, EPS)


def cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    x = _rows(x)
    return np.clip(1.0 - x @ x.T, 0.0, 2.0)


def build_semantic_graph(embeddings: np.ndarray, k: int = 6) -> dict[str, Any]:
    """Build a symmetric locally-scaled kNN graph from cosine distances."""
    x = _rows(embeddings)
    n = len(x)
    if n < 2:
        raise ValueError("At least two ideas are required to build a graph.")
    k = int(np.clip(k, 1, n - 1))
    dist = cosine_distance_matrix(x)
    np.fill_diagonal(dist, np.inf)
    neighbors = np.argsort(dist, axis=1)[:, :k]
    local_scale = np.maximum(np.take_along_axis(dist, neighbors[:, -1:], axis=1).ravel(), 1e-4)
    weights = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in neighbors[i]:
            denom = max(local_scale[i] * local_scale[j], 1e-8)
            weights[i, j] = np.exp(-(dist[i, j] ** 2) / denom)
    weights = np.maximum(weights, weights.T)
    np.fill_diagonal(weights, 0.0)
    graph = csr_matrix(weights)
    component_count, component_labels = connected_components(graph, directed=False)
    return {
        "weights": weights,
        "graph": graph,
        "distances": dist,
        "neighbors": neighbors,
        "components": int(component_count),
        "component_labels": component_labels,
        "k": k,
    }


def spectral_coordinates(graph: dict[str, Any], dimensions: int = 2) -> dict[str, np.ndarray]:
    """Normalized-Laplacian spectral coordinates and diffusion coordinates."""
    w = np.asarray(graph["weights"], dtype=float)
    degree = w.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, EPS))
    normalized_adj = inv_sqrt[:, None] * w * inv_sqrt[None, :]
    laplacian = np.eye(len(w)) - normalized_adj
    vals, vecs = eigh(laplacian)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    d = min(int(dimensions), max(1, len(w) - 1))
    # Skip the trivial first eigenvector when possible.
    start = 1 if len(w) > 2 else 0
    spec = vecs[:, start:start + d]
    adjacency_eigenvalues = np.clip(1.0 - vals[start:start + d], -1.0, 1.0)
    diffusion = spec * np.power(adjacency_eigenvalues, 2)[None, :]
    return {
        "coordinates": spec,
        "diffusion_coordinates": diffusion,
        "eigenvalues": vals,
        "laplacian": laplacian,
    }


def geodesic_distance_matrix(graph: dict[str, Any]) -> np.ndarray:
    """Shortest-path distances on the semantic kNN graph; inf means disconnected."""
    w = np.asarray(graph["weights"], dtype=float)
    cost = np.zeros_like(w)
    mask = w > 0
    cost[mask] = -np.log(np.maximum(w[mask], EPS)) + 1e-6
    return shortest_path(csr_matrix(cost), directed=False, unweighted=False)


def local_novelty_scores(candidates: np.ndarray, archive: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean cosine distance to k nearest archive points (larger = more unusual)."""
    c, a = _rows(candidates), _rows(archive)
    if len(a) == 0:
        return np.ones(len(c))
    d = np.clip(1.0 - c @ a.T, 0.0, 2.0)
    kk = min(max(1, int(k)), len(a))
    return np.sort(d, axis=1)[:, :kk].mean(axis=1)


def relevance_scores(candidates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cosine alignment to a reference centroid or target embedding."""
    c = _rows(candidates)
    r = _rows(reference).mean(axis=0, keepdims=True)
    r = _rows(r)
    return np.clip((c @ r.T).ravel(), -1.0, 1.0)


def sinkhorn_transport(
    source: np.ndarray,
    target: np.ndarray,
    regularization: float = 0.08,
    iterations: int = 300,
) -> dict[str, Any]:
    """Entropically regularized balanced OT between two finite embedding sets."""
    x, y = _rows(source), _rows(target)
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Source and target domains must both contain ideas.")
    reg = max(float(regularization), 1e-4)
    cost = np.clip(1.0 - x @ y.T, 0.0, 2.0)
    log_k = -cost / reg
    a = np.full(len(x), 1.0 / len(x))
    b = np.full(len(y), 1.0 / len(y))
    # Log-domain Sinkhorn updates are stable for small regularization values.
    log_a, log_b = np.log(a), np.log(b)
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    for _ in range(int(iterations)):
        row_lse = np.logaddexp.reduce(log_k + v[None, :], axis=1)
        u = log_a - row_lse
        col_lse = np.logaddexp.reduce(log_k + u[:, None], axis=0)
        v = log_b - col_lse
    plan = np.exp(log_k + u[:, None] + v[None, :])
    plan /= max(plan.sum(), EPS)
    return {
        "plan": plan,
        "cost": cost,
        "transport_cost": float(np.sum(plan * cost)),
        "source_marginal_error": float(np.max(np.abs(plan.sum(axis=1) - a))),
        "target_marginal_error": float(np.max(np.abs(plan.sum(axis=0) - b))),
        "regularization": reg,
    }


def transport_pairs(result: dict[str, Any], top_n: int = 12) -> list[dict[str, float | int]]:
    plan = np.asarray(result["plan"])
    cost = np.asarray(result["cost"])
    flat = np.argsort(plan.ravel())[::-1][:top_n]
    pairs = []
    for index in flat:
        i, j = np.unravel_index(index, plan.shape)
        pairs.append({"source_index": int(i), "target_index": int(j),
                      "transport_mass": float(plan[i, j]), "semantic_cost": float(cost[i, j])})
    return pairs


def generate_vector_population(
    embeddings: np.ndarray,
    count: int = 100,
    seed: int = 42,
    noise_scale: float = 0.12,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Create candidate vectors using crossover, direction transfer and mutation."""
    x = _rows(embeddings)
    if len(x) < 3:
        raise ValueError("At least three source ideas are required for evolution.")
    rng = np.random.default_rng(seed)
    population, metadata = [], []
    methods = ["cross-domain interpolation", "semantic direction transfer", "manifold-neighborhood mutation"]
    for _ in range(int(count)):
        method = methods[int(rng.integers(0, len(methods)))]
        if method == methods[0]:
            i, j = rng.choice(len(x), 2, replace=False)
            alpha = float(rng.uniform(0.2, 0.8))
            z = alpha * x[i] + (1 - alpha) * x[j]
            sources = [int(i), int(j)]
        elif method == methods[1]:
            i, j, q = rng.choice(len(x), 3, replace=False)
            alpha = float(rng.uniform(0.25, 0.9))
            z = x[i] + alpha * (x[j] - x[q])
            sources = [int(i), int(j), int(q)]
        else:
            i = int(rng.integers(0, len(x)))
            z = x[i] + rng.normal(0, noise_scale, size=x.shape[1])
            sources = [i]
        z = z / max(np.linalg.norm(z), EPS)
        population.append(z)
        metadata.append({"method": method, "source_indices": sources})
    return np.vstack(population), metadata


def pareto_front(objectives: np.ndarray) -> np.ndarray:
    """Indices of non-dominated rows; all objectives are maximized."""
    values = np.asarray(objectives, dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for i, point in enumerate(values):
        dominates_i = np.all(values >= point, axis=1) & np.any(values > point, axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            keep[i] = False
    return np.flatnonzero(keep)


def map_elites(
    vectors: np.ndarray,
    novelty: np.ndarray,
    relevance: np.ndarray,
    fitness: np.ndarray,
    bins: int = 6,
) -> dict[tuple[int, int], int]:
    """Quality-diversity archive over novelty and relevance descriptors."""
    n = max(2, int(bins))
    nov = np.clip(np.asarray(novelty), 0.0, 1.0)
    rel = np.clip((np.asarray(relevance) + 1.0) / 2.0, 0.0, 1.0)
    fit = np.asarray(fitness)
    archive: dict[tuple[int, int], int] = {}
    for i in range(len(fit)):
        cell = (min(n - 1, int(nov[i] * n)), min(n - 1, int(rel[i] * n)))
        if cell not in archive or fit[i] > fit[archive[cell]]:
            archive[cell] = int(i)
    return archive


def dpp_greedy_select(vectors: np.ndarray, quality: np.ndarray, n_select: int = 10) -> list[int]:
    """Greedy MAP-style DPP subset selection using a PSD cosine kernel."""
    x = _rows(vectors)
    q = np.maximum(np.asarray(quality, dtype=float), 1e-6)
    similarity = x @ x.T  # Gram matrix is positive semidefinite; retain negative cosines.
    kernel = np.outer(q, q) * similarity
    kernel += np.eye(len(x)) * 1e-8
    selected: list[int] = []
    remaining = set(range(len(x)))
    target = min(max(0, int(n_select)), len(x))
    while remaining and len(selected) < target:
        best_idx, best_gain = None, -np.inf
        for idx in remaining:
            trial = selected + [idx]
            sign, logdet = np.linalg.slogdet(kernel[np.ix_(trial, trial)])
            gain = logdet if sign > 0 else -np.inf
            if gain > best_gain:
                best_idx, best_gain = idx, gain
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def analyze_candidates(
    embeddings: np.ndarray,
    candidates: np.ndarray,
    novelty_k: int = 5,
    bins: int = 6,
) -> dict[str, Any]:
    """Score candidates and construct Pareto, MAP-Elites and DPP selections."""
    x, c = _rows(embeddings), _rows(candidates)
    novelty = local_novelty_scores(c, x, k=novelty_k)
    relevance = relevance_scores(c, x)
    # Normalize novelty for descriptor/scoring only; retain raw cosine distance too.
    nov01 = novelty / max(float(np.max(novelty)), EPS)
    rel01 = (relevance + 1.0) / 2.0
    quality = 0.55 * nov01 + 0.45 * rel01
    objectives = np.column_stack([nov01, rel01])
    front = pareto_front(objectives)
    elites = map_elites(c, nov01, relevance, quality, bins=bins)
    diverse = dpp_greedy_select(c, quality, n_select=min(12, len(c)))
    return {
        "novelty": novelty,
        "novelty_normalized": nov01,
        "relevance": relevance,
        "quality": quality,
        "pareto_indices": front,
        "map_elites": elites,
        "dpp_indices": diverse,
    }
