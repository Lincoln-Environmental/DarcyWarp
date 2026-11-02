"""Independent NPF reference controller regressions."""
import numpy as np


def test_reference_linear_controller_rejects_finite_bad_solutions(monkeypatch):
    from scipy import sparse
    from DARCY_WARP_PACKAGE.validation.npf662_reference import solve_reference_correction, ReferenceLinearControls

    def bad_direct_solve(*, A, b):
        return np.zeros_like(b)

    monkeypatch.setattr('DARCY_WARP_PACKAGE.validation.npf662_reference.spsolve', bad_direct_solve)
    correction, info = solve_reference_correction(matrix=sparse.eye(3, format='csr'), rhs=np.ones(3),
        controls=ReferenceLinearControls())
    assert np.isfinite(correction).all() and not info['linear_accepted']
    assert info['linear_relative_residual'] == 1.



def test_reference_scaled_linear_system():
    from scipy import sparse
    from DARCY_WARP_PACKAGE.validation.npf662_reference import solve_reference_correction, ReferenceLinearControls
    matrix = sparse.csr_matrix(np.array([[2e-100, -1e-100], [-1e100, 2e100]]))
    exact = np.array([2., 3.])
    correction, info = solve_reference_correction(matrix=matrix, rhs=matrix@exact, controls=ReferenceLinearControls())
    assert info['linear_accepted']
    np.testing.assert_allclose(actual=correction, desired=exact, rtol=1e-14, atol=0.)

