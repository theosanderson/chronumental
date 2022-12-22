from os import name
import pandas as pd
import numpy as np
import gzip
import datetime
from alive_progress import alive_it
import treeswift
import xopen
import lzma
from . import helpers
from datetime import datetime as dt

def read_tabular_file(tabular_file_name,**kwargs):
    # Handle gzipped files, and csv and tsv
    tabular_file = xopen.xopen(tabular_file_name, "r")
    print(f"Reading {tabular_file_name}")

    stripped_name = tabular_file_name.replace(".gz", "").replace(".bz2", "").replace(".xz", "").replace(".lzma", "")
    if stripped_name.endswith(".csv"):
        return pd.read_csv(tabular_file, **kwargs)
    if stripped_name.endswith(".tsv"):
        return pd.read_csv(tabular_file, sep="\t", **kwargs)
    raise Exception(f"Tabular file {tabular_file} was expected to end in tsv or csv")

def get_correct_column(columns, possible_values):
    for column in columns:
        if str(column).strip().lower() in possible_values:
            return column
    raise Exception(f"""Could not find a column with one of the following names: {possible_values}. Available were:
    {columns}""")

def fromYearFraction(yearFraction):
    #check type is float
    if not isinstance(yearFraction, float):
        raise ValueError("Not a float")
    if np.isnan(yearFraction):
        raise ValueError("Is NaN")
    year = int(yearFraction)
    fraction = yearFraction - year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)
    date = startOfThisYear + (fraction * ((startOfNextYear) - (startOfThisYear)))
    return date

def get_metadata(metadata_file):
    # get just the top row of the file
    metadata = read_tabular_file(metadata_file, nrows=1)
    # get the column names
    metadata_columns = metadata.columns
    name_column = get_correct_column(metadata_columns, possible_values=["strain", "name", "taxon"])
    print(f"Using {name_column} as the name column. This must be the name of the taxa in the tree.")
    
    date_column = get_correct_column(metadata_columns, possible_values=["date"])
    fields = [date_column, name_column]
    print(f"Using {fields} as the fields to parse.")

    print("Reading metadata")
    metadata = read_tabular_file(metadata_file, low_memory=False, usecols=fields).rename(columns={name_column: 'strain', date_column: 'date'})



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

        for node in tree.traverse_preorder():
            if node.label:
                node.label = node.label.replace("'", "")
        return tree
def get_datetime_and_error(x):
        
        try:
            return [datetime.datetime.strptime(x, '%Y-%m-%d'),1]
        except TypeError:
            try:
                return [fromYearFraction(x),1]
            except ValueError:
                print(f"Warning: could not parse date {x}, it will not feature in calculation.")
                return [None,None]
        except ValueError:
            try:
                return fromYearFraction(x)
            except ValueError:
                pass

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
    
    try:
        reference_point = the_oldest['strain'].values[0]
    except IndexError:
        raise ValueError("Could not find a reference point on the tree. This probably means that the names on your tree don't match the strain/name/taxon column of the dates file.")

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
                                title = "Creating target date array"):
        
        terminal.label = terminal.label.replace("'", "")
        if terminal.label in lookup:
            date = lookup[terminal.label][0]
            diff = (date - lookup[reference_point][0]).days
            terminal_targets[terminal.label] = diff
            terminal_targets_error[terminal.label] = lookup[terminal.label][1]
    return terminal_targets, terminal_targets_error


def get_initial_branch_lengths_and_name_all_nodes(tree):
    initial_branch_lengths = {}
    for i, node in alive_it(enumerate(tree.traverse_preorder()),
                                title="finding initial branch_lengths"):
        # If node label looks like a float, then it's something else, so we set to None:
        if node.label and node.label.replace(".", "").strip().isdigit():
            node.label = None
        if not node.label:
            name = helpers.get_unnnamed_node_label(i)
            node.label = name
        if node.edge_length is None:
            node.edge_length = 0

  
        initial_branch_lengths[node.label] = node.edge_length
    return initial_branch_lengths

def get_rows_and_cols_of_sparse_matrix(tree,terminal_name_to_pos, name_to_pos):
    # Here we define row col coordinates for 1s in a sparse matrix of mostly 0s
    count = 0

    for leaf in alive_it(tree.traverse_leaves(), title="Counting tree for sparse matrix creation"):
        if leaf.label in terminal_name_to_pos:
            cur_node = leaf
            count+=1
            while cur_node.parent is not None:
                count+=1
                cur_node = cur_node.parent
    
    rows = np.zeros(count, dtype=int)
    cols = np.zeros(count, dtype=int)

    location = 0
    for leaf in alive_it(tree.traverse_leaves(), title = "Populating sparse matrix rows, cols"):
        if leaf.label in terminal_name_to_pos:
            cur_node = leaf
            rows[location] = terminal_name_to_pos[leaf.label]
            cols[location] = name_to_pos[cur_node.label]
            location+=1
            while cur_node.parent is not None:
                rows[location] = terminal_name_to_pos[leaf.label]
                cols[location] = name_to_pos[cur_node.parent.label]
                location+=1
                cur_node = cur_node.parent
    return rows,cols
