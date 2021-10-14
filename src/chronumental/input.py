import pandas as pd
import gzip
import datetime
import tqdm
import treeswift

def get_metadata(metadata_file):
    print("Reading metadata")
    metadata = pd.read_table(metadata_file, low_memory=False)

    if "date" not in metadata:
        raise Exception("Metadata has no date column")

    if "strain" not in metadata:
        raise Exception("Metadata has no strain column")
    return metadata

def read_tree(tree_file):
        if tree_file.endswith('.gz'):
            return treeswift.read_tree(gzip.open(tree_file,"rt").read(), schema="newick")
        else:
            return treeswift.read_tree(open(tree_file,"rt").read(), schema="newick")

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
        terminal_targets = {}
        for terminal in tqdm.tqdm(tree.traverse_leaves(),
                                  "Creating target date array"):
            
            terminal.label = terminal.label.replace("'", "")
            if terminal.label in lookup:
                date = lookup[terminal.label]
                diff = (date - lookup[reference_point]).days
                terminal_targets[terminal.label] = diff
        return terminal_targets
