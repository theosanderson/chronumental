import datetime
import functools
import pandas as pd
from Bio import Phylo
import jax.numpy as jnp
import numpy as np
from . import helpers

import pandas as pd
import tqdm
import gzip
import jax
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
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
        '-c',
        help='Molecular clock rate in mutations per site per year',
        default=1e-3,
        type=float)

    parser.add_argument('-g',
                        help="Genome size in bases",
                        default=30e3,
                        type=float)

    parser.add_argument(
        '-vd',
        default=0.3,
        type=float,
        help=
        "Scale factor for date distribution. Essentially a measure of how uncertain we think the measured dates are."
    )

    parser.add_argument('-vb',
                        default=1,
                        type=float,
                        help="Scale factor for branch length distribution")

    parser.add_argument('--steps',
                        default=1000,
                        type=int,
                        help="Number of steps to use for the SVI")

    parser.add_argument('--lr',
                        default=0.1,
                        type=float,
                        help="Adam learning rate")

    args = parser.parse_args()
    substitutions_per_site_per_year = args.c
    genome_size = args.g

   

    print("Reading metadata")
    metadata = pd.read_table(args.dates, low_memory=False)

    if "date" not in metadata:
        raise Exception("Metadata has no date column")

    if "strain" not in metadata:
        raise Exception("Metadata has no strain column")

    def read_tree():
        if args.tree.endswith('.gz'):
            return Phylo.read(gzip.open(args.tree, "rt"), 'newick')
        else:
            return Phylo.read(args.tree, 'newick')

    print("Reading tree")
    tree = read_tree()

    def get_datetime(x):
        try:
            return datetime.datetime.strptime(x, '%Y-%m-%d')
        except ValueError:
            return None

    print("Extracting dates from metadata")
    metadata['processed_date'] = metadata['date'].apply(get_datetime)

    full = metadata[~metadata['processed_date'].isnull()]
    lookup = dict(zip(full['strain'], full['processed_date']))

    # Get oldest date in full, and corresponding strain:
    oldest_date = full['processed_date'].min()
    reference_point = full[full['processed_date'] ==
                           oldest_date]['strain'].values[0]

    print(f"Using {reference_point} as an arbitrary reference point")
    lookup[reference_point] = oldest_date

    def get_target_dates():
        terminal_targets = {}
        for terminal in tqdm.tqdm(tree.root.get_terminals(),
                                  "Creating target date array"):
            
            terminal.name = terminal.name.replace("'", "")
            if terminal.name in lookup:
                date = lookup[terminal.name]
                diff = (date - lookup[reference_point]).days
                terminal_targets[terminal.name] = diff
        return terminal_targets

    target_dates = get_target_dates()
    terminal_names = target_dates.keys()
    
    terminal_target_dates_array = jnp.asarray(
        [float(target_dates[x]) for x in terminal_names])
    print(f"Found {len(terminal_names)} terminals")
    terminal_name_to_pos = {x: i for i, x in enumerate(terminal_names)}

    def assign_paths(tree):
        for node in tqdm.tqdm(tree.get_nonterminals(),
                              "Finding paths in the tree"):
            for clade in node.clades:
                if node == tree.root:
                    clade.path = [node]
                else:
                    clade.path = node.path + [
                        node,
                    ]

    assign_paths(tree)

    initial_branch_lengths = {}
    for i, node in tqdm.tqdm(enumerate(tree.root.find_clades()),
                             "finding initial branch_lengths"):
        if node.name == "":
            node.name = f"internal_node_{i}"
        if node.branch_length is None:
            node.branch_length = 0
        initial_branch_lengths[node.name] = node.branch_length
    names_init = initial_branch_lengths.keys()
    branch_distances_array = np.array(
        [initial_branch_lengths[x] for x in names_init])
    name_to_pos = {x: i for i, x in enumerate(names_init)}

    # Here we define row col coordinates for 1s in a sparse matrix of mostly 0s
    rows = []
    cols = []

    for i, node in tqdm.tqdm(enumerate(tree.root.get_terminals()),
                             "Creating sparse matrix of edge contributions"):
        if node.name in terminal_name_to_pos:
            for item in node.path + [
                    node,
            ]:
                rows.append(terminal_name_to_pos[node.name])
                cols.append(name_to_pos[item.name])

    rows = jnp.asarray(rows)
    print("Rows array created")
    cols = jnp.asarray(cols)
    print("Cols array created")

    final_terminal_dimension = len(terminal_name_to_pos)

    def calc_dates(branch_lengths_array):
        A = ((rows, cols), jnp.ones_like(cols))
        B = branch_lengths_array.reshape((branch_lengths_array.shape[0], 1))
        calc_dates = helpers.sp_matmul(A, B,
                                       final_terminal_dimension).squeeze()
        return calc_dates

    substitutions_per_site_per_year = 1e-3
    genome_size = 30000

    print(branch_distances_array)

    print(genome_size)

    print(substitutions_per_site_per_year)

    initial_time = 365 * (
        branch_distances_array / genome_size
    ) / substitutions_per_site_per_year + 3  # We add 3 to this prior because tranmsmission after zero days is relatively unlikely

    def model():

        branch_times = numpyro.sample(
            "latent_time_length",
            dist.TruncatedNormal(low=0,
                                 loc=initial_time,
                                 scale=args.vb,
                                 validate_args=True))

        mutation_rate = numpyro.sample(
            f"latent_mutation_rate",
            dist.TruncatedNormal(low=0,
                                 loc=substitutions_per_site_per_year,
                                 scale=substitutions_per_site_per_year,
                                 validate_args=True))

        branch_distances = numpyro.sample(
            "branch_distances",
            dist.Poisson(genome_size * mutation_rate * branch_times / 365),
            obs=branch_distances_array)

        calced_dates = calc_dates(branch_times)

        final_dates = numpyro.sample(
            f"final_dates",
            dist.Normal(calced_dates,
                        args.vd * np.ones(calced_dates.shape[0])),
            obs=terminal_target_dates_array)

    print("---------")
    print("Performing SVI:")
    guide = AutoDelta(model)
    svi = SVI(model, guide, optim.Adam(args.lr), Trace_ELBO())
    state = svi.init(jax.random.PRNGKey(0))

    num_steps = args.steps
    for step in range(num_steps):
        state, loss = svi.update(state)
        if step % 10 == 0:
            times = svi.get_params(state)['latent_time_length_auto_loc']
            new_dates = calc_dates(times)
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
            print(step, loss, date_cor, date_error, max_date_error, length_cor,
                  svi.get_params(state)['latent_mutation_rate_auto_loc'])

    tree2 = read_tree()

    branch_length_lookup = dict(
        zip(names_init,
            svi.get_params(state)['latent_time_length_auto_loc'].tolist()))
    for i, node in enumerate(tree2.root.find_clades()):
        if node.name == "":
            node_name = f"internal_node_{i}"
        else:
            node_name = node.name
        node.branch_length = branch_length_lookup[node_name]
    
    output_tree = prepend_to_file_name(args.tree,"timetree_")

    if args.tree.endswith(".gz"):
        output_handle = gzip.open(output_tree, "wt")
    else:
        output_handle = open(output_tree, "w")
    Phylo.write(tree2, output_handle, "newick")

    new_dates_absolute = [lookup[reference_point] + datetime.timedelta(days=x) for x in new_dates.tolist()]
    
    output_meta = pd.DataFrame({'strain': terminal_names,
                                'date': new_dates_absolute})

    output_meta_file = prepend_to_file_name(args.tree,"timetree_meta_")
    output_meta.to_csv(f"{output_meta_file}.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()
