import math

from speech_jscc.diagnostics.g0_exposure import EpochSampler, steps_for_epochs, verify_resume_replay, should_continue_exposure


def test_steps_are_true_epoch_batches() -> None:
    assert steps_for_epochs(1, 16, 4) == 4
    assert steps_for_epochs(2, 17, 4) == math.ceil(2 * 17 / 4)
    assert steps_for_epochs(64, 1491, 4) == math.ceil(64 * 1491 / 4)


def test_sampler_is_deterministic_without_replacement_and_counts_presentations() -> None:
    ids = [f"u{i}" for i in range(17)]
    first = EpochSampler(ids, batch_size=4, seed=23, subset_key="17")
    second = EpochSampler(ids, batch_size=4, seed=23, subset_key="17")
    batches1 = list(first.iter_epoch(1)); batches2 = list(second.iter_epoch(1))
    assert batches1 == batches2
    assert set(x for batch in batches1 for x in batch) == set(ids)
    assert [len(x) for x in batches1] == [4, 4, 4, 4, 1]
    assert set(first.presentation_counts.values()) == {1}
    list(first.iter_epoch(2))
    assert set(first.presentation_counts.values()) == {2}
    assert first.optimizer_steps == 10


def test_resume_replay_compares_state_before_first_new_batch() -> None:
    saved = {"a": 1, "b": 1}
    current = {"a": 2, "b": 1}
    verify_resume_replay(current, ["a"], saved)


def test_after_epoch16_continues_when_improving_and_stops_only_on_joint_plateau() -> None:
    assert should_continue_exposure(16, {"train_loss": -1e-3, "unseen_loss": 0.0, "unseen_correlation": 0.0}, 1e-4)
    assert not should_continue_exposure(16, {"train_loss": 1e-6, "unseen_loss": -1e-6, "unseen_correlation": 1e-6}, 1e-4)
    assert should_continue_exposure(8, {"train_loss": 0.0, "unseen_loss": 0.0, "unseen_correlation": 0.0}, 1e-4)
