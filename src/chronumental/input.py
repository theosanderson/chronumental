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
        return treeswift.read_tree(tree_file, schema="newick")
def get_datetime(x):
        try:
            return datetime.datetime.strptime(x, '%Y-%m-%d')
        except ValueError:
            return None

def process_dates(metadata):
    metadata['processed_date'] = metadata['date'].apply(get_datetime)

def get_complete_dates(metadata):
    return metadata[~metadata['processed_date'].isnull()]

def get_oldest(full):
    oldest_date = full['processed_date'].min()
    reference_point = full[full['processed_date'] ==
                           oldest_date]['strain'].values[0]
    return oldest_date, reference_point

def get_target_dates(tree, lookup, reference_point):
    """
    Returns a list of dictionary mapping names to integer dates being targeted.
    Dates are relative to the date of the reference point, which forms an arbitary origin.
    """
    terminal_targets = {}
    for terminal in tqdm.tqdm(tree.traverse_leaves(),
                                "Creating target date array"):
        
        terminal.label = terminal.label.replace("'", "")
        if terminal.label in lookup:
            date = lookup[terminal.label]
            diff = (date - lookup[reference_point]).days
            terminal_targets[terminal.label] = diff
    return terminal_targets


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
