"""Numerical evidence, independent of instruction text and external solvers."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "skills/differentiate-seismic-solver/scripts/numerics.py"
SPEC = importlib.util.spec_from_file_location("numerics", SCRIPT)
numerics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(numerics)


def test_gradient_metrics_and_precision():
    reference = np.array([1.0, -3.0, 2.0])
    result = numerics.compare_gradients(reference, reference.astype(np.float32))
    assert result["passed"] and result["informative"]
    assert result["relative_l2"] == 0 and result["relative_linf"] == 0
    assert result["cosine"] == pytest.approx(1)
    assert not result["dtype_match"]
    assert not numerics.compare_gradients(reference, 2 * reference)["passed"]
    assert not numerics.compare_gradients(reference, -reference)["passed"]
    assert not numerics.compare_gradients(reference, np.zeros_like(reference))["passed"]


@pytest.mark.parametrize("candidate", [np.zeros(2), np.array([1.0, np.nan, 2.0]),
                                        np.array([1, 2, 3]), np.ones(3, complex)])
def test_invalid_gradient_contract(candidate):
    with pytest.raises(ValueError):
        numerics.compare_gradients(np.ones(3), candidate)


def test_zero_gradient_pair_is_uninformative_and_tolerance_is_configurable():
    assert not numerics.compare_gradients(np.zeros(3), np.zeros(3))["passed"]
    assert not numerics.compare_gradients(np.full(3, 1e-13), np.full(3, 1e-13))["informative"]
    assert numerics.compare_gradients(np.full(3, 1e-13), np.full(3, 1e-13), atol=0)["passed"]
    reference = np.array(3.0)
    assert numerics.compare_gradients(reference, reference + 1e-4, atol=1e-3)["passed"]
    assert not numerics.compare_gradients(reference, reference + 1e-4, atol=0, rtol=1e-6)["passed"]
    tiny = np.array([1e-200, -2e-200])
    assert numerics.compare_gradients(tiny, tiny, atol=0)["passed"]
    assert not numerics.compare_gradients(tiny, 2 * tiny, atol=0)["passed"]


def test_structured_gradient_preserves_arbitrary_field_names():
    reference = np.array([(1.0, [2.0, 3.0])], dtype=[("elastic_modulus", "f8"), ("drag", "f8", (2,))])
    result = numerics.compare_gradients(reference, reference.copy())
    assert result["passed"] and result["relative_l2"] == 0
    wrong = reference.copy()
    wrong["drag"][0, 1] = -3
    assert not numerics.compare_gradients(reference, wrong)["passed"]
    with pytest.raises(ValueError):
        numerics.compare_gradients(reference, np.ones(1))


def test_linear_action_and_complex_hermitian_convention():
    matrix = np.array([[1 + 2j, 3 - 1j], [-2j, 0.5 + 1j], [2, -1 + 3j]])
    x, y = np.array([1 - 2j, 0.5 + 1j]), np.array([2j, -1 + 0.5j, 3 - 2j])
    correct = numerics.check_adjoint(lambda v: matrix @ v, lambda v: matrix.conj().T @ v, x, y)
    incorrect = numerics.check_adjoint(lambda v: matrix @ v, lambda v: matrix.T @ v, x, y)
    assert correct["passed"] and correct["relative_error"] < 1e-14
    assert not incorrect["passed"] and incorrect["relative_error"] > 0.1
    assert correct["lhs"]["imag"] == pytest.approx(np.vdot(matrix @ x, y).imag)
    with pytest.raises(ValueError):
        numerics.check_adjoint(lambda v: v, lambda v: v[:1], x, x)
    zero = numerics.check_adjoint(lambda v: 0 * v, lambda v: 0 * v, x, x)
    assert not zero["passed"] and not zero["informative"]


def quadratic():
    matrix = np.array([[3.0, 0.5], [0.5, 2.0]])
    model, direction = np.array([0.7, -1.2]), np.array([0.8, 0.3])
    return lambda v: 0.5 * v @ matrix @ v, model, matrix @ model, direction


def test_correct_quadratic_taylor_preserves_raw_values():
    evaluate, model, gradient, direction = quadratic()
    steps = [0.2, 0.1, 0.05, 0.025, 0.0125, 1e-15, 1e-16]
    result = numerics.taylor_test(evaluate, model, gradient, direction, steps)
    assert result["passed"] and result["second_order_observed"] and result["central_agrees"]
    assert len(result["steps"]) == len(steps)
    for row in result["steps"]:
        h = row["step"]
        expected = abs(evaluate(model + h * direction) - evaluate(model) - h * np.vdot(gradient, direction))
        assert row["first_order_residual"] == expected
    assert result["steps"][2]["first_order_order"] == pytest.approx(2, abs=1e-10)
    assert not result["steps"][-1]["perturbation_resolved"]


@pytest.mark.parametrize("factor", [0.0, -1.0, 0.5, 2.0])
def test_taylor_rejects_wrong_gradients(factor):
    evaluate, model, gradient, direction = quadratic()
    result = numerics.taylor_test(evaluate, model, factor * gradient, direction)
    assert not result["passed"]
    if factor:
        assert not result["central_agrees"]


def test_linear_correct_derivative_has_no_observable_second_order_evidence():
    gradient, model, direction = np.array([2.0, 3.0]), np.array([0.4, -1.2]), np.array([0.6, 0.7])
    result = numerics.taylor_test(lambda v: gradient @ v, model, gradient, direction)
    assert result["central_agrees"]
    assert not result["passed"] and not result["second_order_observed"]


def test_zero_directional_gradient_and_roundoff_only_steps_cannot_pass():
    evaluate, model, gradient, _ = quadratic()
    orthogonal = np.array([gradient[1], -gradient[0]])
    assert not numerics.taylor_test(evaluate, model, gradient, orthogonal)["informative"]
    result = numerics.taylor_test(evaluate, model, gradient, np.zeros(2))
    assert not result["passed"] and not result["informative"]
    result = numerics.taylor_test(evaluate, model, gradient, np.ones(2), [1e-16, 1e-17, 1e-18])
    assert not result["passed"] and not any(row["observable"] for row in result["steps"])


def test_complex_model_real_objective():
    model, direction = np.array([1 + 2j, -1 + 0.5j]), np.array([0.2 + 0.4j, 0.3 - 0.1j])
    result = numerics.taylor_test(lambda v: 0.5 * np.vdot(v, v).real, model, model, direction)
    assert result["passed"]


def test_nonlinear_central_errors_have_second_order_convergence():
    model, direction = np.array([0.5, 0.8]), np.array([0.4, 0.6])
    result = numerics.taylor_test(lambda v: np.sum(v**3) / 3, model, model**2, direction)
    assert result["passed"]
    assert result["steps"][2]["central_error_order"] == pytest.approx(2, abs=1e-9)


@pytest.mark.parametrize("steps", [[0.1, 0.1, 0.01], [0.01, 0.1, 1.0], [0.1, 0.01, 0], [0.1, np.nan, 0.01]])
def test_taylor_rejects_invalid_steps(steps):
    evaluate, model, gradient, direction = quadratic()
    with pytest.raises(ValueError):
        numerics.taylor_test(evaluate, model, gradient, direction, steps)


def test_cli_npz_default_keys_hashes_and_failure(tmp_path):
    reference, candidate, output = (tmp_path / name for name in ("ref.npz", "candidate.npz", "report.json"))
    np.savez(reference, gradient_alpha=np.array(2.0), unrelated=np.array(1.0))
    np.savez(candidate, gradient_alpha=np.array(2.0), unrelated=np.array(9.0))
    command = [sys.executable, str(SCRIPT), "compare", "--reference", str(reference),
               "--candidate", str(candidate), "--output", str(output)]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = json.loads(output.read_text())
    assert report["passed"] and report["keys"] == ["gradient_alpha"]
    assert len(report["reference"]["sha256"]) == 64
    assert report["reference"]["size_bytes"] == reference.stat().st_size
    assert json.loads(run.stdout) == report
    np.savez(candidate, gradient_alpha=np.array(4.0))
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 1 and not json.loads(run.stdout)["passed"]


def test_cli_arbitrary_parameter_keys_structures_and_invalid_files(tmp_path):
    reference, candidate = tmp_path / "ref.npz", tmp_path / "cand.npz"
    data = np.array((2.0, 3.0), dtype=[("bulk", "f8"), ("shear", "f8")])
    np.savez(reference, constitutive_state=data)
    np.savez(candidate, constitutive_state=data)
    command = [sys.executable, str(SCRIPT), "compare", "--reference", str(reference), "--candidate", str(candidate)]
    run = subprocess.run(command + ["--keys", "constitutive_state"], capture_output=True, text=True)
    assert run.returncode == 0 and json.loads(run.stdout)["passed"]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 1 and "no shared" in json.loads(run.stdout)["error"]
