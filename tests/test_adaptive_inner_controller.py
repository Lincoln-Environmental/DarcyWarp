from __future__ import annotations

from dataclasses import dataclass, field

from DARCY_WARP_PACKAGE.warped_darcy import (
    AdaptiveInnerSolveConfig,
    AdaptiveInnerSolveState,
    _adaptive_practical_acceptance_allowed,
    _remaining_legacy_fallback_cycles,
    _should_continue_picard_after_refreshed_acceptance,
    _adaptive_state_requires_legacy_fallback,
    _classify_inner_contraction,
    _classify_dual_inner_contraction,
    _compute_inner_forcing_eta,
    _compute_inner_target_residual,
    _predict_next_inner_block_size,
    _run_adaptive_inner_kcycle_blocks,
    _validate_adaptive_inner_solve_config,
)


@dataclass
class ResidualSequenceRunner:
    residuals: list[float]
    relative_residuals: list[float] = field(default_factory=list)
    requested_cycles: list[int] = field(default_factory=list)

    def run(self, block_cycles: int) -> dict[str, float | int | bool]:
        self.requested_cycles.append(int(block_cycles))
        residual = self.residuals.pop(0)
        return {
            "actual_cycles": int(block_cycles),
            "residual_after_rms": float(residual),
            "relative_flow_residual_rms": float(
                self.relative_residuals.pop(0) if self.relative_residuals else 0.0
            ),
            "rollback_required": False,
            "head_nonfinite": False,
            "numerical_breakdown": False,
        }


@dataclass
class RollbackRecorder:
    count: int = 0

    def rollback(self) -> None:
        self.count += 1


def test_initial_residual_already_below_target_uses_zero_cycles():
    config = AdaptiveInnerSolveConfig()
    runner = ResidualSequenceRunner(residuals=[])

    state = _run_adaptive_inner_kcycle_blocks(
        initial_residual_rms=1.0e-5,
        target_residual_rms=1.0e-4,
        forcing_eta=0.25,
        previous_outer_residual_rms=None,
        previous_outer_dh_rms=None,
        max_cycles=200,
        config=config,
        run_block=runner.run,
    )

    assert state.total_cycles == 0
    assert state.target_achieved is True
    assert state.termination_reason == "initial_residual_already_below_target"
    assert runner.requested_cycles == []


def test_strong_contraction_increases_block_size_and_reaches_target():
    config = AdaptiveInnerSolveConfig(initial_block_cycles=4, max_block_cycles=16)
    runner = ResidualSequenceRunner(residuals=[0.20, 0.03])

    state = _run_adaptive_inner_kcycle_blocks(
        initial_residual_rms=1.0,
        target_residual_rms=0.05,
        forcing_eta=0.25,
        previous_outer_residual_rms=None,
        previous_outer_dh_rms=None,
        max_cycles=40,
        config=config,
        run_block=runner.run,
    )

    assert state.converged is True
    assert state.target_achieved is True
    assert len(runner.requested_cycles) == 2
    assert runner.requested_cycles[1] >= runner.requested_cycles[0]


def test_moderate_contraction_prediction_is_bounded():
    config = AdaptiveInnerSolveConfig(min_block_cycles=2, max_block_cycles=16)
    predicted = _predict_next_inner_block_size(
        current_block_cycles=4,
        residual_after=0.35,
        target_residual=0.10,
        contraction_ratio=0.70,
        per_cycle_factor=0.70 ** 0.25,
        classification="useful",
        remaining_cycles=20,
        config=config,
    )

    assert predicted is not None
    assert 1 <= predicted <= 16


def test_near_target_uses_small_final_block():
    config = AdaptiveInnerSolveConfig(min_block_cycles=2, max_block_cycles=16)
    predicted = _predict_next_inner_block_size(
        current_block_cycles=8,
        residual_after=0.018,
        target_residual=0.010,
        contraction_ratio=0.80,
        per_cycle_factor=0.80 ** 0.125,
        classification="useful",
        remaining_cycles=20,
        config=config,
    )

    assert predicted in (1, 2)


def test_stagnation_stops_after_patience():
    config = AdaptiveInnerSolveConfig(initial_block_cycles=2, stall_patience=2, max_block_cycles=4)
    runner = ResidualSequenceRunner(residuals=[0.99, 0.985, 0.984])

    state = _run_adaptive_inner_kcycle_blocks(
        initial_residual_rms=1.0,
        target_residual_rms=0.01,
        forcing_eta=0.25,
        previous_outer_residual_rms=None,
        previous_outer_dh_rms=None,
        max_cycles=40,
        config=config,
        run_block=runner.run,
    )

    assert state.stalled is True
    assert state.termination_reason == "residual_stagnation"
    assert state.total_cycles < 40


def test_divergence_rolls_back_and_terminates():
    config = AdaptiveInnerSolveConfig(initial_block_cycles=3)
    rollback = RollbackRecorder()
    runner = ResidualSequenceRunner(residuals=[1.10])

    state = _run_adaptive_inner_kcycle_blocks(
        initial_residual_rms=1.0,
        target_residual_rms=0.01,
        forcing_eta=0.25,
        previous_outer_residual_rms=None,
        previous_outer_dh_rms=None,
        max_cycles=40,
        config=config,
        run_block=runner.run,
        rollback_block=rollback.rollback,
    )

    assert state.diverged is True
    assert state.rollback_count == 1
    assert rollback.count == 1
    assert state.termination_reason == "block_divergence_rolled_back"


def test_nonfinite_initial_residual_falls_back_to_legacy():
    config = AdaptiveInnerSolveConfig()
    runner = ResidualSequenceRunner(residuals=[])

    state = _run_adaptive_inner_kcycle_blocks(
        initial_residual_rms=float("nan"),
        target_residual_rms=0.01,
        forcing_eta=0.25,
        previous_outer_residual_rms=None,
        previous_outer_dh_rms=None,
        max_cycles=40,
        config=config,
        run_block=runner.run,
    )

    assert state.fallback_used is True
    assert state.legacy_fallback_used is True
    assert state.fallback_reason == "nonfinite_initial_head_residual"


def test_hard_cycle_ceiling_is_enforced():
    config = AdaptiveInnerSolveConfig(initial_block_cycles=4, max_block_cycles=16)
    runner = ResidualSequenceRunner(residuals=[0.8, 0.7, 0.6])

    state = _run_adaptive_inner_kcycle_blocks(
        initial_residual_rms=1.0,
        target_residual_rms=1.0e-6,
        forcing_eta=0.25,
        previous_outer_residual_rms=None,
        previous_outer_dh_rms=None,
        max_cycles=5,
        config=config,
        run_block=runner.run,
    )

    assert state.total_cycles <= 5
    assert runner.requested_cycles[0] != 200
    assert state.termination_reason == "max_cycles_hard_ceiling"


def test_forcing_eta_uses_previous_outer_residual_ratio():
    config = AdaptiveInnerSolveConfig(eta_gamma=0.5, eta_power=1.5, eta_min=0.02, eta_max=0.30)
    eta = _compute_inner_forcing_eta(
        current_outer_residual_rms=0.1,
        previous_outer_residual_rms=0.4,
        config=config,
    )

    assert config.eta_min <= eta <= config.eta_max


def test_target_residual_respects_picard_and_absolute_bounds():
    target = _compute_inner_target_residual(
        initial_residual_rms=0.5,
        forcing_eta=0.25,
        residual_floor=1.0e-12,
        inner_head_residual_tol_min=1.0e-4,
        inner_head_residual_tol_max=1.0e-2,
        inner_picard_scale_max_fraction=0.10,
        previous_outer_dh_rms=5.0e-3,
        hclose=1.0e-4,
    )

    assert 1.0e-4 <= target <= 1.0e-2


def test_contraction_classifier_identifies_stall():
    classification = _classify_inner_contraction(
        residual_before=1.0,
        residual_after=0.99,
        block_cycles=2,
        config=AdaptiveInnerSolveConfig(),
    )

    assert classification["classification"] == "stalled"


def test_config_validation_rejects_invalid_ordering():
    config = AdaptiveInnerSolveConfig(
        min_block_cycles=4,
        initial_block_cycles=2,
    )

    try:
        _validate_adaptive_inner_solve_config(config=config, max_cycles=20)
    except ValueError as exc:
        assert "adaptive_inner_initial_block_cycles" in str(exc)
    else:
        raise AssertionError("expected invalid adaptive configuration to raise ValueError")


def test_min_total_cycles_blocks_early_target_exit():
    config = AdaptiveInnerSolveConfig(initial_block_cycles=2, min_total_cycles=4)
    runner = ResidualSequenceRunner(residuals=[0.01, 0.001, 0.001])

    state = _run_adaptive_inner_kcycle_blocks(
        initial_residual_rms=1.0,
        target_residual_rms=0.05,
        forcing_eta=0.25,
        previous_outer_residual_rms=None,
        previous_outer_dh_rms=None,
        max_cycles=8,
        config=config,
        run_block=runner.run,
    )

    assert state.target_achieved is True
    assert state.total_cycles >= 4


def test_unusable_adaptive_state_requires_legacy_fallback():
    state = AdaptiveInnerSolveState(usable_for_picard=False)
    assert _adaptive_state_requires_legacy_fallback(state) is True


def test_practical_acceptance_requires_adaptive_target():
    assert _adaptive_practical_acceptance_allowed(
        practical_acceptance_enabled=True,
        adaptive_controller_used=True,
        inner_target_achieved=False,
    ) is False


def test_forcing_uses_comparable_initial_residuals_and_handles_zero():
    config = AdaptiveInnerSolveConfig(eta_initial=0.25, eta_min=0.02, eta_max=0.30)
    assert _compute_inner_forcing_eta(
        current_outer_residual_rms=0.0,
        previous_outer_residual_rms=1.0,
        config=config,
    ) == config.eta_initial


def test_dual_residual_target_is_required():
    config = AdaptiveInnerSolveConfig(initial_block_cycles=2, min_total_cycles=2)
    runner = ResidualSequenceRunner(residuals=[0.01], relative_residuals=[0.5])
    state = _run_adaptive_inner_kcycle_blocks(
        initial_residual_rms=1.0,
        target_residual_rms=0.05,
        forcing_eta=0.25,
        previous_outer_residual_rms=None,
        previous_outer_dh_rms=None,
        max_cycles=2,
        config=config,
        run_block=runner.run,
        initial_relative_flow_residual_rms=1.0,
        target_relative_flow_residual_rms=0.1,
    )

    assert state.target_achieved is False


def test_zero_cycle_result_preserves_zero_accounting():
    config = AdaptiveInnerSolveConfig(min_total_cycles=10)
    state = _run_adaptive_inner_kcycle_blocks(
        initial_residual_rms=0.0,
        target_residual_rms=1.0e-4,
        forcing_eta=0.25,
        previous_outer_residual_rms=None,
        previous_outer_dh_rms=None,
        max_cycles=20,
        config=config,
        run_block=ResidualSequenceRunner(residuals=[]).run,
        initial_relative_flow_residual_rms=0.0,
        target_relative_flow_residual_rms=1.0e-4,
    )

    assert state.total_cycles == 0
    assert state.target_achieved is True


def test_adaptive_practical_acceptance_requires_target_achievement():
    assert _adaptive_practical_acceptance_allowed(
        practical_acceptance_enabled=True,
        adaptive_controller_used=True,
        inner_target_achieved=False,
        final_relative_flow_residual_rms=0.2,
        relative_flow_target=0.1,
    ) is False


def test_flow_only_divergence_rolls_back_block():
    config = AdaptiveInnerSolveConfig(initial_block_cycles=2)
    rollback = RollbackRecorder()
    runner = ResidualSequenceRunner(residuals=[0.5], relative_residuals=[2.0])
    state = _run_adaptive_inner_kcycle_blocks(
        initial_residual_rms=1.0,
        target_residual_rms=0.01,
        forcing_eta=0.25,
        previous_outer_residual_rms=None,
        previous_outer_dh_rms=None,
        max_cycles=4,
        config=config,
        run_block=runner.run,
        rollback_block=rollback.rollback,
        initial_relative_flow_residual_rms=1.0,
        target_relative_flow_residual_rms=0.01,
    )

    assert state.diverged is True
    assert rollback.count == 1


def test_head_progress_without_flow_progress_is_not_usable():
    config = AdaptiveInnerSolveConfig(initial_block_cycles=2, minimum_usable_reduction_ratio=0.8)
    runner = ResidualSequenceRunner(residuals=[0.2], relative_residuals=[1.0])
    state = _run_adaptive_inner_kcycle_blocks(
        initial_residual_rms=1.0,
        target_residual_rms=0.01,
        forcing_eta=0.25,
        previous_outer_residual_rms=None,
        previous_outer_dh_rms=None,
        max_cycles=2,
        config=config,
        run_block=runner.run,
        initial_relative_flow_residual_rms=1.0,
        target_relative_flow_residual_rms=0.01,
    )

    assert state.usable_for_picard is False
    assert state.head_reduction_ratios[-1] <= 0.8
    assert state.flow_reduction_ratios[-1] > 0.8


def test_dual_stagnation_uses_worse_per_cycle_factor():
    contraction = _classify_dual_inner_contraction(
        head_before=1.0,
        head_after=0.25,
        flow_before=1.0,
        flow_after=0.99,
        block_cycles=2,
        config=AdaptiveInnerSolveConfig(),
    )

    assert contraction["q_controller"] == contraction["q_flow"]
    assert contraction["classification"] == "stalled"


def test_min_total_cycles_greater_than_max_cycles_is_rejected():
    config = AdaptiveInnerSolveConfig(
        initial_block_cycles=4,
        max_block_cycles=4,
        min_total_cycles=5,
    )
    try:
        _validate_adaptive_inner_solve_config(config=config, max_cycles=4)
    except ValueError as exc:
        assert "min_total_cycles" in str(exc)
    else:
        raise AssertionError("expected impossible cycle configuration to raise ValueError")


def test_relative_flow_diagnostics_are_populated():
    config = AdaptiveInnerSolveConfig(initial_block_cycles=2, min_total_cycles=2)
    runner = ResidualSequenceRunner(residuals=[0.01], relative_residuals=[0.01])
    state = _run_adaptive_inner_kcycle_blocks(
        initial_residual_rms=1.0,
        target_residual_rms=0.05,
        forcing_eta=0.25,
        previous_outer_residual_rms=None,
        previous_outer_dh_rms=None,
        max_cycles=2,
        config=config,
        run_block=runner.run,
        initial_relative_flow_residual_rms=1.0,
        target_relative_flow_residual_rms=0.05,
    )

    assert state.initial_relative_flow_residual_rms == 1.0
    assert state.target_relative_flow_residual_rms == 0.05
    assert state.final_relative_flow_residual_rms == 0.01
    assert state.head_per_cycle_convergence_factors
    assert state.flow_per_cycle_convergence_factors
    assert state.controller_per_cycle_convergence_factors


def test_legacy_practical_acceptance_ignores_adaptive_flow_target():
    assert _adaptive_practical_acceptance_allowed(
        practical_acceptance_enabled=True,
        adaptive_controller_used=False,
        inner_target_achieved=False,
        final_relative_flow_residual_rms=1.0,
        relative_flow_target=1.0e-4,
    ) is True


def test_failed_refreshed_acceptance_continues_picard():
    assert _should_continue_picard_after_refreshed_acceptance(
        provisional_picard_acceptance=True,
        refreshed_picard_acceptance=False,
    ) is True


def test_next_block_prediction_uses_worse_target_gap():
    config = AdaptiveInnerSolveConfig(min_block_cycles=2, max_block_cycles=16)
    predicted = _predict_next_inner_block_size(
        current_block_cycles=8,
        residual_after=0.001,
        target_residual=0.01,
        contraction_ratio=0.5,
        per_cycle_factor=0.9,
        classification="useful",
        remaining_cycles=16,
        config=config,
        flow_residual_after=1.0,
        flow_target=0.01,
    )

    assert predicted is not None
    assert predicted > 2


def test_satisfied_residual_growth_below_target_is_not_divergent():
    result = _classify_dual_inner_contraction(
        head_before=1.0e-6,
        head_after=9.0e-6,
        flow_before=1.0,
        flow_after=0.1,
        block_cycles=1,
        config=AdaptiveInnerSolveConfig(),
        head_target=1.0e-5,
        flow_target=0.2,
    )

    assert result["classification"] != "divergent"
    assert result["q_controller"] == 0.0


def test_crossing_back_above_target_can_trigger_divergence():
    result = _classify_dual_inner_contraction(
        head_before=1.0e-6,
        head_after=2.0e-5,
        flow_before=0.1,
        flow_after=0.1,
        block_cycles=1,
        config=AdaptiveInnerSolveConfig(),
        head_target=1.0e-5,
        flow_target=0.2,
    )

    assert result["classification"] == "divergent"


def test_adaptive_and_legacy_fallback_cycles_cannot_exceed_ceiling():
    legacy_cycles = _remaining_legacy_fallback_cycles(
        max_cycles=20,
        adaptive_cycles_used=17,
        selected_legacy_cycles=10,
    )

    assert legacy_cycles == 3
    assert 17 + legacy_cycles <= 20


def test_fixed_block_continuation_matches_legacy_cycle_caps():
    config = AdaptiveInnerSolveConfig(
        initial_block_cycles=5,
        min_block_cycles=5,
        max_block_cycles=5,
        min_total_cycles=10,
    )
    for cycle_cap in (10, 20, 40):
        block_count = cycle_cap // 5
        residuals = [0.9 ** (index + 1) for index in range(block_count)]
        runner = ResidualSequenceRunner(
            residuals=list(residuals),
            relative_residuals=list(residuals),
        )
        state = _run_adaptive_inner_kcycle_blocks(
            initial_residual_rms=1.0,
            target_residual_rms=0.01,
            forcing_eta=0.25,
            previous_outer_residual_rms=None,
            previous_outer_dh_rms=None,
            max_cycles=cycle_cap,
            config=config,
            run_block=runner.run,
            initial_relative_flow_residual_rms=1.0,
            target_relative_flow_residual_rms=0.01,
        )

        assert state.cycles_per_block == [5] * block_count
        assert state.total_cycles == cycle_cap
        assert state.final_residual_rms == residuals[-1]
        assert state.final_relative_flow_residual_rms == residuals[-1]
