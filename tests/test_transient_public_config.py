"""Production controls are public and isolated between coupled callers."""
from DARCY_WARP_PACKAGE.solvers.transient_config import production_secant_sy_settings
from working_tests.transient_replay_settings import default_solve_controls


def test_coupled_callers_cannot_mutate_later_production_controls():
    first = production_secant_sy_settings()
    first['solve_controls']['max_levels'] = 99
    first['solve_controls']['hclose'] = 1.0
    second = production_secant_sy_settings()
    assert second['unconfined_storage_mode'] == 'mf6_convertible_secant_sy'
    assert second['storage_reference'] == 'current_picard'
    assert second['solve_controls']['max_levels'] == 4
    assert second['solve_controls']['hclose'] == 1.e-4
    assert default_solve_controls() == second['solve_controls']
