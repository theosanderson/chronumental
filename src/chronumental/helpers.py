"""Shared constants and tree utilities."""
import numpy as np

# The Julian year, which is what "per year" means to treetime, LSD2 and to a
# decimal-year date. Chronumental previously used 365 in every conversion
# between days and years. Used consistently that constant cancels out of the
# fit -- it only renames the unit the clock rate is quoted in -- but it made
# our reported rate differ from every tool we compare against by 0.07%, and
# it made --output_unit years disagree with a decimal year by the same.
DAYS_PER_YEAR = 365.25


def get_unnnamed_node_label(i):
    name = f"NODE_{i:07d}"
    return name


def preorder_traversal(node):
    """Yield every node at or below `node`, parents before children.

    Written with an explicit stack rather than recursion. The recursive
    version overflowed Python's call stack on trees deep enough to matter:
    a simulated million-tip coalescent tree raised RecursionError before
    fitting could start, and the large SARS-CoV-2 trees this tool targets
    are deep and unbalanced too.

    The order is identical to the recursive version -- a node, then the
    whole of its first child's subtree, then its second child's, and so on
    -- which matters because unlabelled nodes are named by their position
    in this traversal.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        # Reversed, so the first child is popped first.
        stack.extend(reversed(current.children))


def make_path_sum(parent_indices, n_rounds, root_index):
    """Return a function summing branch values from the root down to each node.

    This replaces a sparse matrix holding one entry per (node, ancestor) pair.
    That structure costs memory proportional to the total root-to-tip path
    length over the whole tree, which on a 300k-tip tree measured here was
    116 million entries and 2.6 GB -- the single largest allocation in a run,
    and the reason a million-tip tree could not be dated on a 46 GB machine.

    Pointer jumping computes the same sums in O(nodes) memory. Each round
    adds a node's accumulated value to its current ancestor's and then
    doubles the ancestor pointer, so after ceil(log2(depth + 1)) rounds every
    node holds the sum over its whole path. On a 300k-tip tree that is 80 MB
    instead of 2.6 GB and 12 ms per call instead of 440 ms; on a GPU at 100k
    tips it is 2.7 MB instead of 183 MB and 0.10 ms instead of 1.53 ms.

    Both operations are a gather and an add, which suit a GPU better than the
    scatter-add the sparse version needed. The round count grows only
    logarithmically with depth, so the unrolled graph stays small.

    `parent_indices` must give each node the index of its parent, with the
    root pointing at itself, and the root's own branch value must be zero so
    that it contributes nothing to its descendants.
    """

    def path_sum(branch_values):
        accumulated = branch_values.at[root_index].set(0.0)
        ancestor = parent_indices
        for _ in range(n_rounds):
            accumulated = accumulated + accumulated[ancestor]
            ancestor = ancestor[ancestor]
        return accumulated

    return path_sum


def path_sum_numpy(branch_values, parent_indices, n_rounds, root_index):
    """The same pointer-jumping sum as make_path_sum, in numpy on the host.

    Used once, for the output, where float64 is wanted and there is no
    device to keep the values on.
    """
    accumulated = np.array(branch_values, dtype=np.float64)
    accumulated[root_index] = 0.0
    ancestor = np.asarray(parent_indices)
    for _ in range(n_rounds):
        accumulated = accumulated + accumulated[ancestor]
        ancestor = ancestor[ancestor]
    return accumulated
