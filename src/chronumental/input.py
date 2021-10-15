import pandas as pd
import gzip
import datetime
import tqdm
import treeswift

def get_metadata(metadata_file):
    fields = ['date', 'strain']

    print("Reading metadata")
    metadata = pd.read_table(metadata_file, low_memory=False, usecols=fields)

    for field in fields:
        if field not in metadata:
            raise Exception(f"Metadata has no {field} column")
    return metadata

def read_tree(tree_file):
        tree = treeswift.read_tree(tree_file, schema="newick")
        for node in tree.traverse_preorder():
            if node.label:
                node.label = node.label.replace("'", "")
        return tree
def get_datetime_and_error(x):
        try:
            return [datetime.datetime.strptime(x, '%Y-%m-%d'),1]
        except ValueError:
            try:
                return [datetime.datetime.strptime(x, '%Y-%m') + datetime.timedelta(days=30//2),30]
            except ValueError:
                try:
                    return [datetime.datetime.strptime(x, '%Y') +datetime.timedelta(days=365//2),365]
                except ValueError:
                    if x != "" and x!="?":
                        print(f"Warning: could not parse date {x}, it will not feature in calculation.")
                    return [None,None]

def process_dates(metadata):
    metadata['date_and_error'] = metadata['date'].apply(get_datetime_and_error)
    metadata['processed_date'] = metadata['date_and_error'].apply(lambda x: x[0])
    metadata['processed_date_error'] = metadata['date_and_error'].apply(lambda x: x[1])
    metadata.drop(columns=['date_and_error'], inplace=True)

def get_present_dates(metadata, only_use_full_dates):
    if only_use_full_dates:
        return metadata[(~metadata['processed_date'].isnull()) & (metadata['processed_date_error'] <5)]
    else:
        return metadata[~metadata['processed_date'].isnull()]

def get_oldest(full, tree):
    leaf_to_node = tree.label_to_node(selection="leaves")
    filtered = full[full['strain'].isin(leaf_to_node.keys())]
    oldest_date = filtered['processed_date'].min()
    the_oldest = filtered[filtered['processed_date'] ==
                           oldest_date]
    reference_point = the_oldest['strain'].values[0]

    distance = tree.distance_between(tree.root, leaf_to_node[reference_point])
    return reference_point, distance

def get_target_dates(tree, lookup, reference_point):
    """
    Returns a list of dictionary mapping names to integer dates being targeted.
    Dates are relative to the date of the reference point, which forms an arbitary origin.
    """
    terminal_targets = {}
    terminal_targets_error = {}
    for terminal in tqdm.tqdm(tree.traverse_leaves(),
                                "Creating target date array"):
        
        terminal.label = terminal.label.replace("'", "")
        if terminal.label in lookup:
            date = lookup[terminal.label][0]
            diff = (date - lookup[reference_point][0]).days
            terminal_targets[terminal.label] = diff
            terminal_targets_error[terminal.label] = lookup[terminal.label][1]
    return terminal_targets, terminal_targets_error


def get_initial_branch_lengths_and_name_all_nodes(tree):
    initial_branch_lengths = {}
    for i, node in tqdm.tqdm(enumerate(tree.traverse_preorder()),
                                "finding initial branch_lengths"):
        if not node.label:
            name = f"internal_node_{i}"
            node.label = name
        if node.edge_length is None:
            node.edge_length = 0
        initial_branch_lengths[node.label] = node.edge_length
    return initial_branch_lengths

def get_rows_and_cols_of_sparse_matrix(tree,terminal_name_to_pos, name_to_pos):
    # Here we define row col coordinates for 1s in a sparse matrix of mostly 0s
    rows = []
    cols = []
    for leaf in tqdm.tqdm(tree.traverse_leaves(), "Traversing tree for sparse matrix creation"):
        if leaf.label in terminal_name_to_pos:
            cur_node = leaf
            rows.append(terminal_name_to_pos[leaf.label])
            cols.append(name_to_pos[cur_node.label])
            while cur_node.parent is not None:
                rows.append(terminal_name_to_pos[leaf.label])
                cols.append(name_to_pos[cur_node.parent.label])
                cur_node = cur_node.parent
    return rows,cols
