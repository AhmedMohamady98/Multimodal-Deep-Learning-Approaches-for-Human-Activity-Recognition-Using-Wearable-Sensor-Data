"""
Group cross-validation splits for HARMES — with proper train/val/test separation.

Each fold returns three disjoint participant sets:
  - train: used to fit the model
  - val:   used for early stopping and best-checkpoint selection
  - test:  evaluated ONCE at the end with the best checkpoint (no leakage)

The validation set is one held-out participant from the training group, picked
deterministically (the first participant ID in the sorted train group). This
keeps val/test disjoint at the participant level, which matches the LOPO
methodology and what the supervisor specified.

CHANGES vs previous version (May 2026 bug fix — supervisor-flagged):
  - Added `get_3fold_splits_with_val()` returning {train, val, test}
  - Added `get_lopo_split_with_val()` returning {train, val, test} for one test participant
  - Old `get_3fold_splits()` kept for backward compatibility but should NOT be
    used by new training scripts — it has no validation set and leads to test
    leakage when used naively.
"""

def get_3fold_splits(
    participant_ids: list[int] | None = None,
) -> list[dict[str, list[int]]]:
    """
    Legacy 3-fold split — train + test only, NO validation.

    ⚠️ DEPRECATED for new training scripts: using `test` for early stopping
    causes data leakage. Use `get_3fold_splits_with_val()` instead.
    Kept only for compatibility with v1/v2 analysis scripts that read
    historical results.
    """
    if participant_ids is None:
        participant_ids = list(range(1, 21))

    all_ids = list(participant_ids)
    n = len(all_ids)

    if n <= 3:
        folds = [[all_ids[i]] for i in range(n)]
        while len(folds) < 3:
            folds.append([])
    else:
        base   = n // 3
        extras = n % 3
        folds, start = [], 0
        for i in range(3):
            size = base + (1 if i < extras else 0)
            folds.append(all_ids[start : start + size])
            start += size

    return [
        {
            "train": [p for j, fold in enumerate(folds) for p in fold if j != i],
            "test":  folds[i],
        }
        for i in range(3)
    ]


def get_3fold_splits_with_val(
    participant_ids: list[int] | None = None,
) -> list[dict[str, list[int]]]:
    """
    3-fold group CV with a separate validation participant per fold.

    For each fold:
      - test = the fold's held-out group
      - val  = the first participant (smallest ID) of what would have been the train group
      - train = the remaining participants

    All three sets are disjoint at the participant level — no leakage.

    Returns a list of 3 dicts with keys {"train", "val", "test"}.

    Default 20-participant splits:
      fold 0: test=[1..7],   val=[8],  train=[9..20]
      fold 1: test=[8..14],  val=[1],  train=[2..7, 15..20]
      fold 2: test=[15..20], val=[1],  train=[2..14]
    """
    base = get_3fold_splits(participant_ids)
    result = []
    for fold in base:
        train_full = sorted(fold["train"])
        if len(train_full) < 2:
            raise ValueError(
                f"Cannot carve a validation participant from a train set of size "
                f"{len(train_full)} (need at least 2 train participants)."
            )
        val   = [train_full[0]]
        train = train_full[1:]
        result.append({"train": train, "val": val, "test": fold["test"]})
    return result


def get_lopo_split_with_val(
    test_participant: int,
    all_participants: list[int] | None = None,
) -> dict[str, list[int]]:
    """
    LOPO split with a separate validation participant.

    Args:
        test_participant: the participant held out as the test set
        all_participants: defaults to [1..20]

    Returns:
        {"train": [...], "val": [P_val], "test": [test_participant]}

    The validation participant is the smallest-ID participant in the remaining
    pool (after removing the test participant). All three sets are disjoint.

    Examples:
      test=1  → val=[2],  train=[3..20]
      test=5  → val=[1],  train=[2..4, 6..20]
      test=20 → val=[1],  train=[2..19]
    """
    if all_participants is None:
        all_participants = list(range(1, 21))

    if test_participant not in all_participants:
        raise ValueError(
            f"test_participant={test_participant} is not in all_participants "
            f"(got {sorted(all_participants)}). Pass a valid ID."
        )

    train_pool = sorted([p for p in all_participants if p != test_participant])
    if len(train_pool) < 2:
        raise ValueError(
            f"LOPO with test={test_participant}: only {len(train_pool)} other "
            f"participants — need at least 2 for train+val."
        )
    val   = [train_pool[0]]
    train = train_pool[1:]
    return {"train": train, "val": val, "test": [test_participant]}


def mini_splits(participant_ids: list[int] = None) -> list[dict[str, list[int]]]:
    """
    Convenience wrapper for the 3-participant SMOKE-TEST dry-run.
    Defaults to participants [1, 2, 3] with 1 per fold.

    ⚠️ Returns only {"train": [...], "test": [...]} per fold — NO validation
    set. This is intentional for the smoke-test path where speed matters more
    than unbiased eval. DO NOT USE for production training; use
    `get_3fold_splits_with_val()` instead.
    """
    if participant_ids is None:
        participant_ids = [1, 2, 3]
    return get_3fold_splits(participant_ids)
