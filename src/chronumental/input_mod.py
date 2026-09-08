import pandas as pd
import numpy as np
import datetime
from alive_progress import alive_it
import treeswift
import xopen
from . import helpers
from datetime import datetime as dt


def read_tabular_file(tabular_file_name, **kwargs):
    # Handle gzipped files, and csv and tsv
    tabular_file = xopen.xopen(tabular_file_name, "r")
    print(f"Reading {tabular_file_name}")

    stripped_name = tabular_file_name.replace(".gz", "").replace(
        ".bz2", "").replace(".xz", "").replace(".lzma", "")
    if stripped_name.endswith(".csv"):
        return pd.read_csv(tabular_file,
                           dtype={
                               "strain": str,
                               "name": str,
                               "taxon": str
                           },
                           **kwargs)
    if stripped_name.endswith(".tsv"):
        return pd.read_csv(tabular_file,
                           sep="\t",
                           dtype={
                               "strain": str,
                               "name": str,
                               "taxon": str
                           },
                           **kwargs)
    raise Exception(
        f"Tabular file {tabular_file_name} was expected to end in tsv or csv")


def get_correct_column(columns, possible_values):
    for column in columns:
        if str(column).strip().lower() in possible_values:
            return column
    raise Exception(
        f"""Could not find a column with one of the following names: {possible_values}. Available were:
    {columns}""")


def fromYearFraction(yearFraction):
    #check type is float
    if not isinstance(yearFraction, float):
        raise ValueError("Not a float")
    if np.isnan(yearFraction):
        raise ValueError("Is NaN")
    year = int(yearFraction)
    if year == 0:
        raise ValueError(
            "The year zero does not exist in the Gregorian calendar")
    fraction = yearFraction - year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year + 1, month=1, day=1)
    date = startOfThisYear + (fraction * ((startOfNextYear) -
                                          (startOfThisYear)))
    return date


def get_metadata(metadata_file):
    # get just the top row of the file
    metadata = read_tabular_file(metadata_file, nrows=1)
    # get the column names
    metadata_columns = metadata.columns
    name_column = get_correct_column(
        metadata_columns, possible_values=["strain", "name", "taxon"])
    print(
        f"Using {name_column} as the name column. This must be the name of the taxa in the tree."
    )

    date_column = get_correct_column(metadata_columns,
                                     possible_values=["date"])
    fields = [date_column, name_column]
    print(f"Using {fields} as the fields to parse.")

    print("Reading metadata")
    metadata = read_tabular_file(metadata_file,
                                 low_memory=False,
                                 usecols=fields).rename(columns={
                                     name_column: 'strain',
                                     date_column: 'date'
                                 })

    for field in ['date']:
        if field not in metadata:
            raise Exception(f"Metadata has no {field} column")
    return metadata


def read_tree(tree_file):
    extension = tree_file.replace(".gz", "").replace(".bz2", "").split(".")[-1]
    if extension == "nex" or extension == "nexus":
        trees = treeswift.read_tree_nexus(tree_file)
        keys = list(trees.keys())
        print(f"Using tree {keys[0]} from Nexus file")
        tree = trees[keys[0]]
    elif extension == "nwk" or extension == "newick":
        tree = treeswift.read_tree_newick(tree_file)
    else:
        print("Assuming tree file is newick (change extension for nexus)")
        tree = treeswift.read_tree_newick(tree_file)
    #raise ValueError(tree)

    # The root has no parent, so any branch length on it does not represent
    # elapsed time between two nodes and there is nothing above it to date.
    # Some tools write one anyway: treetime's divergence trees carry a root
    # branch, and on one real dataset it was 0.001 substitutions per site,
    # over a year at that dataset's clock rate. Left in place it was added to
    # every node's root-to-tip sum and, worse, pushed the reported root date
    # that far later, which is the single largest disagreement with treetime
    # on that dataset. Zeroing it moved the median disagreement across
    # internal nodes from 18.5 days to 12.6 and the root from 183 days out to
    # 75. It also removes a spurious Poisson term asking the fit to explain
    # the stem's mutations with a duration nothing else constrains.
    if tree.root.edge_length:
        print(f"Ignoring the root's branch length of {tree.root.edge_length}: "
              f"a root has no parent, so it spans no time.")
        tree.root.edge_length = 0

    for node in tree.traverse_preorder():
        if node.label:
            node.label = node.label.replace("'", "")
    return tree


def _period_midpoint(start, months=0, years=0):
    """Centre an imprecise date in its interval, and give the interval width.

    A date reported as 2020-06 is the whole of June, and 2020 is the whole of
    2020; the value the model should aim at is the middle of that period, not
    its first instant. The previous fixed offsets of 15 and 182 days were both
    early -- June's midpoint is 15.5 days in, and a year's is 182.5 or 183 --
    which biased every imprecise tip earlier than its own metadata implied.

    Computing the period's real end also gets February and leap years right,
    so the returned width is the actual number of days rather than a nominal
    30 or 365.
    """
    if months:
        end = (datetime.datetime(start.year + 1, 1, 1) if start.month == 12
               else datetime.datetime(start.year, start.month + 1, 1))
    else:
        end = datetime.datetime(start.year + years, start.month, start.day)
    span = end - start
    return [start + span / 2, span.days]


def get_datetime_and_error(x):

    try:
        return [datetime.datetime.strptime(x, '%Y-%m-%d'), 1]
    except TypeError:
        try:
            return [fromYearFraction(x), 1]
        except ValueError:
            print(
                f"Warning: could not parse date {x}, it will not feature in calculation."
            )
            return [None, None]
    except ValueError:
        try:
            return fromYearFraction(x)
        except ValueError:
            pass

        try:
            return _period_midpoint(datetime.datetime.strptime(x, '%Y-%m'),
                                    months=1)
        except ValueError:
            try:
                return _period_midpoint(datetime.datetime.strptime(x, '%Y'),
                                        years=1)
            except ValueError:
                if x != "" and x != "?":
                    print(
                        f"Warning: could not parse date {x}, it will not feature in calculation."
                    )
                return [None, None]


def process_dates(metadata):
    metadata['date_and_error'] = metadata['date'].apply(get_datetime_and_error)
    metadata['processed_date'] = metadata['date_and_error'].apply(
        lambda x: x[0])
    metadata['processed_date_error'] = metadata['date_and_error'].apply(
        lambda x: x[1])
    metadata.drop(columns=['date_and_error'], inplace=True)


def get_present_dates(metadata, only_use_full_dates):
    if only_use_full_dates:
        return metadata[(~metadata['processed_date'].isnull())
                        & (metadata['processed_date_error'] < 5)]
    else:
        return metadata[~metadata['processed_date'].isnull()]


def get_oldest(full, tree):
    leaf_to_node = tree.label_to_node(selection="leaves")
    filtered = full[full['strain'].isin(leaf_to_node.keys())]
    oldest_date = filtered['processed_date'].min()
    the_oldest = filtered[filtered['processed_date'] == oldest_date]

    try:
        reference_point = the_oldest['strain'].values[0]
    except IndexError:
        raise ValueError(
            "Could not find a reference point on the tree. This probably means that the names on your tree don't match the strain/name/taxon column of the dates file."
        )

    distance = tree.distance_between(tree.root, leaf_to_node[reference_point])
    return reference_point, distance


def get_specific(full, tree, name):
    leaf_to_node = tree.label_to_node(selection="leaves")
    reference_point = name
    distance = tree.distance_between(tree.root, leaf_to_node[reference_point])
    return reference_point, distance


def get_target_dates(tree, lookup, reference_point):
    """
    Returns a list of dictionary mapping names to integer dates being targeted.
    Dates are relative to the date of the reference point, which forms an arbitary origin.
    """
    terminal_targets = {}
    terminal_targets_error = {}
    for terminal in alive_it(tree.traverse_leaves(),
                             title="Creating target date array"):

        terminal.label = terminal.label.replace("'", "")
        if terminal.label in lookup:
            date = lookup[terminal.label][0]
            # total_seconds rather than .days: the latter floors, which
            # would throw away the half-day centring of month- and year-only
            # dates.
            diff = (date - lookup[reference_point][0]).total_seconds() / 86400
            terminal_targets[terminal.label] = diff
            terminal_targets_error[terminal.label] = lookup[terminal.label][1]
    return terminal_targets, terminal_targets_error


def get_initial_branch_lengths_and_name_all_nodes(tree):
    """Label every node, and record each one's branch length.

    Also returns the labels this invented, as opposed to the ones the input
    tree already carried. The caller needs that to write an output tree
    faithfully: unless --name_all_nodes is given, a node that arrived without
    a label should leave without one. Keeping the set means the output can
    reuse this tree instead of parsing the file a second time, which at 300k
    tips saved about three seconds and a whole second copy of the tree.
    """
    initial_branch_lengths = {}
    invented_labels = set()
    for i, node in alive_it(enumerate(helpers.preorder_traversal(tree.root)),
                            title="finding initial branch_lengths"):
        if not node.label:
            name = helpers.get_unnnamed_node_label(i)
            node.label = name
            invented_labels.add(name)
        if node.edge_length is None:
            node.edge_length = 0

        if node.label in initial_branch_lengths:
            raise ValueError(
                f"Two nodes in the tree are labelled {node.label!r}. Labels "
                "must be unique, including against the NODE_0000000 pattern "
                "used to name unlabelled nodes, because they index the "
                "fitted parameters.")
        initial_branch_lengths[node.label] = node.edge_length
    return initial_branch_lengths, invented_labels


def estimate_initial_times_local(tree,
                                 name_to_pos,
                                 branch_distances_array,
                                 target_dates,
                                 clock_rate,
                                 floor_days=0.01,
                                 mutation_floor=False):
    """Position every node from its immediate children, tips on their dates.

    Tips start exactly where their metadata says. Each internal node is then
    placed, walking up the tree, at the mean over its children of the child's
    position minus what that child's own branch's mutations represent at the
    clock rate.

    Averaging over children rather than over all descendant tips is the point.
    A flat average over tips lets a densely sampled recent clade outvote a
    sparse deep lineage, and recent tips are exactly the ones whose implied
    ancestor date is most distorted by an error in the clock rate, because
    their divergence has had longest to accumulate it. Weighting each child
    subtree equally is the same correction for shared ancestry that the
    independent-contrasts clock estimator makes.

    Returns (branch_time_init, root_date_init) in the same day-relative-to-
    reference units as target_dates.
    """
    days_per_mutation = helpers.DAYS_PER_YEAR / clock_rate
    # Indexing a jax array one element at a time is a device-to-host transfer
    # per node, which at 60k nodes took 6.6 s against 0.2 s for numpy.
    branch_distances_array = np.asarray(branch_distances_array, dtype=float)

    def own_mutation_days(label):
        pos = name_to_pos.get(label)
        if pos is None:
            return 0.0
        return float(branch_distances_array[pos]) * days_per_mutation

    estimate = {}
    for node in tree.traverse_postorder():
        label = node.label
        if node.is_leaf():
            estimate[label] = target_dates.get(label)
            continue
        implied = []
        for child in node.children:
            value = estimate.get(child.label)
            if value is None:
                continue
            implied.append(value - own_mutation_days(child.label))
        estimate[label] = (sum(implied) / len(implied)) if implied else None

    # A node with no dated descendant has nothing to say for itself; it takes
    # its parent's position plus what its own mutations represent at the
    # clock rate, top-down, since with no tip date to defer to the mutation
    # count is the only evidence there is. The same pass keeps children from
    # preceding their parents, which both guarantees positive branch times
    # and stops a noisy local estimate inverting the order.
    adjusted = {}
    branch_time_init = {}
    root_label = tree.root.label
    for node in tree.traverse_preorder():
        label = node.label
        # The floor exists to keep every branch positive. Raising it to the
        # branch's own mutations, converted at the clock rate, additionally
        # pushes a child later whenever the tip-date estimate puts it earlier
        # than the mutation count allows -- which inflates the tree wherever
        # the clock estimate is off. mutation_floor=False keeps only the
        # positivity floor and lets the tip-date estimate stand.
        floor = (max(floor_days, own_mutation_days(label))
                 if mutation_floor else floor_days)
        if node.parent is None:
            adjusted[label] = (estimate.get(label)
                               if estimate.get(label) is not None else 0.0)
            branch_time_init[label] = floor
            continue
        parent = adjusted[node.parent.label]
        candidate = estimate.get(label)
        if candidate is None:
            candidate = parent + max(floor, own_mutation_days(label))
        adjusted[label] = max(candidate, parent + floor)
        branch_time_init[label] = adjusted[label] - parent

    return branch_time_init, adjusted[root_label]


def get_parent_indices(tree, name_to_pos):
    """Parent index per node, plus the root's index and the tree's depth.

    Feeds helpers.make_path_sum. The root points at itself, which makes it a
    fixed point of the pointer-jumping iteration so its descendants stop
    accumulating there. `tree` must already have every node labelled.
    """
    n_nodes = len(name_to_pos)
    parents = np.arange(n_nodes, dtype=np.int32)
    depth_of = {}
    max_depth = 0
    root_index = 0

    for node in alive_it(helpers.preorder_traversal(tree.root),
                         title="Recording parents for path sums"):
        index = name_to_pos[node.label]
        if node.parent is None:
            root_index = index
            depth_of[index] = 0
            continue
        parent_index = name_to_pos[node.parent.label]
        parents[index] = parent_index
        depth = depth_of[parent_index] + 1
        depth_of[index] = depth
        if depth > max_depth:
            max_depth = depth

    return parents, root_index, max_depth
