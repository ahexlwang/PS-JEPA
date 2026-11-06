from psjepa.data import chain_split_indices


def test_chain_split_is_stable_grouped_and_disjoint():
    pdb_ids = [f"p{i // 2}" for i in range(40)]
    chain_ids = ["A"] * 40
    selection, final = chain_split_indices(pdb_ids, chain_ids, 0.25, seed=42)
    again = chain_split_indices(pdb_ids, chain_ids, 0.25, seed=42)
    assert (selection, final) == again
    assert set(selection).isdisjoint(final)
    assert sorted(selection + final) == list(range(40))

    side = {index: "selection" for index in selection} | {
        index: "final" for index in final
    }
    for index in range(0, 40, 2):
        assert side[index] == side[index + 1]


def test_chain_split_rejects_invalid_or_empty_partitions():
    try:
        chain_split_indices(["p"], ["A"], 1.0, seed=0)
    except ValueError as error:
        assert "strictly between" in str(error)
    else:
        raise AssertionError("invalid selection fraction was accepted")

    try:
        chain_split_indices(["p", "p"], ["A", "A"], 0.5, seed=0)
    except ValueError as error:
        assert "empty partition" in str(error)
    else:
        raise AssertionError("an indivisible one-chain dataset was accepted")
