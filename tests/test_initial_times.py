"""Checks on the tip-date initialiser.

It decides where the fit starts, and because the root date is an unconstrained
parameter in days that barely moves during a fit, where it starts is close to
where it ends. So its invariants are worth pinning down: durations positive,
parents before children, tips on their own dates, and each internal node
placed from its children rather than from a global average.
"""
import math

import jax.numpy as jnp
import numpy as np
import pytest
import treeswift

from chronumental import input_mod


def build(newick, dates, clock_rate=10.0, **kwargs):
    """Run the initialiser on a small labelled tree.

    Returns the branch-time dictionary, the root date, and the node positions
    the fit would use.
    """
    tree = treeswift.read_tree_newick(newick)
    labels = [node.label for node in tree.traverse_preorder()]
    assert all(labels), "every node in these fixtures is labelled"
    name_to_pos = {label: i for i, label in enumerate(sorted(labels))}
    distances = np.zeros(len(name_to_pos))
    for node in tree.traverse_preorder():
        distances[name_to_pos[node.label]] = node.edge_length or 0.0
    branch_times, root_date = input_mod.estimate_initial_times_local(
        tree, name_to_pos, jnp.asarray(distances), dates, clock_rate, **kwargs)
    # Absolute position of every node, which is what the fit really starts
    # at. The root sits at root_date; its own branch time is only the
    # positivity floor, and the fit zeroes the root's branch anyway.
    position = {tree.root.label: root_date}
    for node in tree.traverse_preorder():
        if node.parent is None:
            continue
        position[node.label] = (position[node.parent.label] +
                                branch_times[node.label])
    return branch_times, root_date, position


CHERRY = "((a:1,b:1)inner:2,(c:3,d:1)other:1)root;"


def test_every_branch_is_positive_and_parents_come_first():
    dates = {"a": 100.0, "b": 130.0, "c": 90.0, "d": 200.0}
    branch_times, root_date, position = build(CHERRY, dates)
    tree = treeswift.read_tree_newick(CHERRY)
    for node in tree.traverse_preorder():
        assert branch_times[node.label] > 0, node.label
        if node.parent:
            assert position[node.label] >= position[node.parent.label] - 1e-9


def test_a_dated_tip_starts_on_its_own_date_where_the_order_allows():
    """No shrinking towards a global average: a tip's own date is the target."""
    dates = {"a": 100.0, "b": 130.0, "c": 90.0, "d": 200.0}
    _, _, position = build(CHERRY, dates)
    # The latest tip in each clade is the one nothing above it has to push
    # later, so it should land exactly on its date.
    assert position["d"] == pytest.approx(200.0, abs=1e-6)
    assert position["b"] == pytest.approx(130.0, abs=1e-6)


def test_an_internal_node_averages_its_children_not_its_tips():
    """One child subtree, however many tips it holds, gets one vote.

    Here the left child is a single tip and the right child is a clade of
    four tips all at one date. Averaging over descendant tips would put the
    parent near the crowd; averaging over children puts it midway.
    """
    newick = ("(x:0,(t1:0,t2:0,t3:0,t4:0)crowd:0)root;")
    dates = {"x": 0.0, "t1": 100.0, "t2": 100.0, "t3": 100.0, "t4": 100.0}
    _, root_date, position = build(newick, dates, clock_rate=10.0)
    # Zero-length branches mean no mutation time to subtract, so the crowd
    # sits at 100 and x at 0, and their parent is the mean of the two, 50.
    assert position["crowd"] == pytest.approx(100.0, abs=1e-6)
    assert root_date == pytest.approx(50.0, abs=1e-6)


def test_mutations_are_converted_at_the_clock_rate():
    """A child's own branch buys back time in proportion to its mutations."""
    newick = "(only:20)root;"
    dates = {"only": 1000.0}
    for rate in (10.0, 20.0):
        _, root_date, _ = build(newick, dates, clock_rate=rate)
        # 20 mutations at `rate` per year is 20 * 365.25 / rate days.
        assert root_date == pytest.approx(1000.0 - 20 * 365.25 / rate,
                                          abs=1e-4)


def test_an_undated_tip_is_placed_from_its_parent_not_dropped():
    dates = {"a": 100.0, "b": 130.0, "c": 90.0}
    branch_times, _, position = build(CHERRY, dates)
    assert branch_times["d"] > 0
    assert position["d"] >= position["other"] - 1e-9


def test_the_mutation_floor_only_ever_pushes_a_branch_later():
    """The two floors differ in one direction only.

    'mutations' requires a branch to be at least as long as its own mutations
    imply, so it can only lengthen a branch relative to 'positive'. That is
    the whole of the difference measured between the two defaults.
    """
    dates = {"a": 100.0, "b": 101.0, "c": 90.0, "d": 200.0}
    _, _, loose = build(CHERRY, dates, mutation_floor=False)
    _, _, strict = build(CHERRY, dates, mutation_floor=True)
    assert set(loose) == set(strict)
    # Positions are what move monotonically. A branch time can shorten under
    # the stricter floor, because its parent moved later by more than it did.
    for label in loose:
        assert strict[label] >= loose[label] - 1e-9, label
    assert any(strict[label] > loose[label] + 1e-9 for label in loose)


def test_a_deep_tree_does_not_recurse():
    """A ladder deeper than Python's recursion limit must still initialise."""
    depth = 1500
    newick = ""
    for i in range(depth):
        newick = f"(tip{i}:1{',' + newick if newick else ''})node{i}:1"
    newick += ";"
    dates = {f"tip{i}": 1000.0 + i for i in range(depth)}
    branch_times, root_date, position = build(newick, dates)
    assert len(branch_times) == 2 * depth
    assert all(value > 0 for value in branch_times.values())
    assert math.isfinite(root_date)


def test_an_undated_subtree_starts_from_its_mutations_not_the_floor():
    """With no tip date anywhere below it, the mutation count is the only
    evidence a branch has, so it should start there rather than at the
    positivity floor."""
    newick = "((a:5,b:5)ab:10,(c:5,d:5)cd:10)root;"
    dates = {"a": 0.0, "b": 10.0}
    branch_times, _, _ = build(newick, dates, clock_rate=365.25)
    # One mutation per day at this clock rate.
    assert branch_times["cd"] == pytest.approx(10.0, abs=1e-6)
    assert branch_times["c"] == pytest.approx(5.0, abs=1e-6)
    assert branch_times["d"] == pytest.approx(5.0, abs=1e-6)


def test_duplicate_labels_are_rejected():
    """An unlabelled node is named by its preorder position, so a genuine
    label of the same form would silently share its parameters."""
    tree = treeswift.read_tree_newick("((a:1,b:1):1,(c:1,NODE_0000001:1):1);")
    with pytest.raises(ValueError, match="NODE_0000001"):
        input_mod.get_initial_branch_lengths_and_name_all_nodes(tree)
