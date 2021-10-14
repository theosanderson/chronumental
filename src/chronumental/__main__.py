import datetime
import functools
import pandas as pd
from Bio import Phylo
import jax.numpy as jnp
import numpy as np
from . import helpers
from . import input
import dendropy

import pandas as pd
import tqdm
import gzip
import jax
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from . import models
try:
    from . import _version
    version = _version.version
except ImportError:
    version = "dev"
import argparse

def prepend_to_file_name(full_path, to_prepend):
    if "/" in full_path:
        path, file = full_path.rsplit('/', 1)
        return f"{path}/{to_prepend}_{file}"
    else:
        return f"{to_prepend}_{full_path}"


def main():
    print(f"Chronumental {version}")
    print("")
    parser = argparse.ArgumentParser(
        description=
        'Convert a distance tree into time tree with distances in days.')
    parser.add_argument(
        '--tree',
        help=
        'an input newick tree, potentially gzipped, with distances as raw number of mutations',
        required=True)

    parser.add_argument(
        '--dates',
        help=
        'A metadata file with columns strain and date (in 2020-01-02 format)',
        required=True)

    parser.add_argument(
        '--clock',
        help='Molecular clock rate. This should be in units of something per year, where the "something" is the units on the tree.',
        default=30e3*1e-3,
        type=float)

    parser.add_argument(
        '-variance_dates',
        default=0.3,
        type=float,
        help=
        "Scale factor for date distribution. Essentially a measure of how uncertain we think the measured dates are."
    )

    parser.add_argument('-variance_branch_length',
                        default=1,
                        type=float,
                        help="Scale factor for branch length distribution. Essentially how close we want to match the expectation of the Poisson.")

    parser.add_argument('--steps',
                        default=1000,
                        type=int,
                        help="Number of steps to use for the SVI")

    parser.add_argument('--lr',
                        default=0.1,
                        type=float,
                        help="Adam learning rate")

    parser.add_argument('--dates_out',
                        default=None,
                        type=float,
                        help="Output for date tsv (otherwise will use default)")

    parser.add_argument('--tree_out',
                        default=None,
                        type=float,
                        help="Output for tree (otherwise will use default)")
    
    parser.add_argument('--name_all_nodes',
                        default=False,
                        type=bool,
                        help="Should we name all nodes in the output?")



    args = parser.parse_args()

    if args.dates_out is None:
        args.dates_out = prepend_to_file_name(args.dates, "chronumental_dates")+".tsv"

    if args.tree_out is None:
        args.tree_out = prepend_to_file_name(args.tree, "chronumental_timetree")

    metadata = input.get_metadata(args.dates)

    print("Reading tree")
    tree = input.read_tree(args.tree)

    print("Processing dates")
    input.process_dates(metadata)

    full = input.get_complete_dates(metadata)
    lookup = dict(zip(full['strain'], full['processed_date']))

    # Get oldest date in full, and corresponding strain:
    oldest_date, reference_point = input.get_oldest(full)

    print(f"Using {reference_point} as an arbitrary reference point")
    lookup[reference_point] = oldest_date



    target_dates = input.get_target_dates(tree, lookup, reference_point)
    terminal_names = sorted(target_dates.keys())
    
    terminal_target_dates_array = jnp.asarray(
        [float(target_dates[x]) for x in terminal_names])

    print(f"Found {len(terminal_names)} terminals with usable date metadata")

    terminal_name_to_pos = {x: i for i, x in enumerate(terminal_names)}


    initial_branch_lengths = input.get_initial_branch_lengths_and_name_all_nodes(tree)
    names_init = sorted(initial_branch_lengths.keys())
    branch_distances_array = jnp.array(
        [initial_branch_lengths[x] for x in names_init])
    name_to_pos = {x: i for i, x in enumerate(names_init)}

    rows, cols = input.get_rows_and_cols_of_sparse_matrix(tree,terminal_name_to_pos, name_to_pos)

    rows = jnp.asarray(rows)
    print("Rows array created")
    cols = jnp.asarray(cols)
    print("Cols array created")

    print(branch_distances_array)

    my_model = models.FixedClock(rows, cols, branch_distances_array, args.clock, args.variance_branch_length ,args.variance_dates, terminal_target_dates_array)

    print("Performing SVI:")
    svi = SVI(my_model.model, my_model.guide, optim.Adam(args.lr), Trace_ELBO())
    state = svi.init(jax.random.PRNGKey(0))

    num_steps = args.steps
    for step in range(num_steps):
        state, loss = svi.update(state)
        if step % 10 == 0:
            params = svi.get_params(state)
            times = my_model.get_branch_times(params)
            new_dates = my_model.calc_dates(times)
            date_cor = np.corrcoef(
                terminal_target_dates_array,
                new_dates)[0, 1]  # This correlation should be very high
            date_error = np.mean(
                np.abs(terminal_target_dates_array -
                       new_dates))  # Average date error should be small
            max_date_error = np.max(
                np.abs(terminal_target_dates_array - new_dates)
            )  # We know that there are some metadata errors, so there probably should be some big errors
            length_cor = np.corrcoef(
                branch_distances_array,
                times)[0, 1]  # This correlation should be relatively high
            print(f"Step:{step}\tLoss:{loss}\tDate correlation:{date_cor:10.4f}\tMean date error:{date_error:10.4f}\tMax date error:{max_date_error:10.4f}\tLength correlation:{length_cor:10.4f}\tInferred mutation rate:{svi.get_params(state)['latent_mutation_rate_auto_loc']:10.4f}")

    tree2 = input.read_tree(args.tree)

    branch_length_lookup = dict(
        zip(names_init,
            my_model.get_branch_times(svi.get_params(state)).tolist()))
    
    total_lengths_in_time = {}

    total_lengths= dict()

    for i, node in enumerate(tree2.traverse_preorder()):
        if not node.label:
            node_name = f"internal_node_{i}"
            if args.name_all_nodes:
                node.label = node_name
        else:
            node_name = node.label.replace("'", "")
        node.branch_length = branch_length_lookup[node_name]
        if not node.parent:
            total_lengths[node] = node.branch_length
        else:
            total_lengths[node] = node.branch_length + total_lengths[node.parent]

        if node.label:
            total_lengths_in_time[node.label.replace("'","")] = total_lengths[node]
    
    

    tree2.write_tree_newick(args.tree_out)
    print(f"Wrote tree to {args.tree_out}")

    origin_date = lookup[reference_point]
    output_dates = {name: origin_date + datetime.timedelta(days=x) for name,x in total_lengths_in_time.items()}

    names, values = zip(*output_dates.items())
    output_meta = pd.DataFrame({"strain": names, "predicted_date": values})

    
    output_meta.to_csv(args.dates_out, sep="\t", index=False)
    print(f"Wrote predicted dates to {args.dates_out}")

if __name__ == "__main__":
    main()
