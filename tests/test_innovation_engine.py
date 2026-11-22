import numpy as np
from innovation_engine import (
    build_semantic_graph, spectral_coordinates, geodesic_distance_matrix,
    sinkhorn_transport, generate_vector_population, analyze_candidates,
    pareto_front, dpp_greedy_select,
)


def sample_vectors(n=12, d=8, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_graph_and_spectral_shapes():
    x = sample_vectors()
    graph = build_semantic_graph(x, k=4)
    result = spectral_coordinates(graph, dimensions=2)
    assert graph["weights"].shape == (12, 12)
    assert result["coordinates"].shape[0] == 12
    assert np.allclose(graph["weights"], graph["weights"].T)


def test_geodesic_nonnegative_or_infinite():
    x = sample_vectors()
    graph = build_semantic_graph(x, k=4)
    distances = geodesic_distance_matrix(graph)
    finite = distances[np.isfinite(distances)]
    assert np.all(finite >= 0)


def test_sinkhorn_marginals():
    x, y = sample_vectors(7), sample_vectors(5, seed=9)
    result = sinkhorn_transport(x, y, regularization=0.15)
    assert np.isclose(result["plan"].sum(), 1.0, atol=1e-6)
    assert result["source_marginal_error"] < 1e-5
    assert result["target_marginal_error"] < 1e-5


def test_evolution_and_selection():
    x = sample_vectors()
    candidates, _ = generate_vector_population(x, count=30, seed=10)
    result = analyze_candidates(x, candidates)
    assert candidates.shape == (30, x.shape[1])
    assert len(result["pareto_indices"]) >= 1
    assert len(result["dpp_indices"]) == 12
    assert len(set(result["dpp_indices"])) == 12


def test_pareto_front_and_dpp():
    obj = np.array([[1, 0], [0, 1], [0.5, 0.5], [0.2, 0.2]])
    front = set(pareto_front(obj).tolist())
    assert front == {0, 1, 2}
    selected = dpp_greedy_select(sample_vectors(8), np.ones(8), n_select=4)
    assert len(selected) == 4
