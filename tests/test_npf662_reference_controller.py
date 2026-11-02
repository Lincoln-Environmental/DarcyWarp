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



def test_reference_never_line_searches_inadequate_finite_direction(monkeypatch):
    from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case, NPF662Reference

    def inaccurate_direct_solve(*, A, b):
        return .1*np.linalg.solve(a=A.toarray(), b=b)

    monkeypatch.setattr('DARCY_WARP_PACKAGE.validation.npf662_reference.spsolve', inaccurate_direct_solve)
    shape = (1, 1, 3)
    one = np.ones(shape)
    case = NPF662Case(kx=one, ky=one, kz=one, active=one.astype(int),
        fixed=np.array([1, 0, 1]).reshape(shape), values=np.array([1., 1.5, 2.]).reshape(shape),
        source=np.zeros(shape), initial=np.full(shape, 1.1), dx=1., dy=1., dz=2., top=2., cell_type=1)
    unused, info, unused_fields = NPF662Reference(case=case).solve_newton(max_outer=1)
    assert not info['history'][0]['accepted']
    assert all(row['stopping'] == 'inadequate_linear_solve' for row in info['history'][0]['attempts'])
