import os
import sys

GPU_REQUESTED = "--use_gpu" in sys.argv
if not GPU_REQUESTED:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import datetime
import math

import pandas as pd
import jax.numpy as jnp
import numpy as np
from . import helpers
from . import input_mod
import jax
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from . import models
from scipy import stats

try:
    from . import _version
    version = _version.version
except ImportError:
    version = "dev"

print(f"Chronumental {version}")

platform = jax.default_backend()
print(f"Platform: {platform}")

if GPU_REQUESTED and platform == "cpu":
    print("GPU requested but was not available")
    print("This probably reflects your CUDA/jaxlib installation")

import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description=
        'Convert a distance tree into time tree with distances in days.')
    parser.add_argument(
        '--tree',
        help=
        'an input newick tree, potentially gzipped, with branch lengths reflecting genetic distance in integer number of mutations',
        required=True)

    parser.add_argument(
        '--dates',
        help=
        'A metadata file with columns strain and date (in "2020-01-02" format, or less precisely, "2021-01", "2021")',
        required=True)

    parser.add_argument(
        '--dates_out',
        default=None,
        type=str,
        help="Output for date tsv (otherwise will use default)")

    parser.add_argument('--tree_out',
                        default=None,
                        type=str,
                        help="Output for tree (otherwise will use default)")

    parser.add_argument(
        "--treat_mutation_units_as_normalised_to_genome_size",
        default=None,
        type=int,
        help=
        "If your branch sizes, and mutation rate, are normalised to per-site values, then enter the genome size here."
    )

    parser.add_argument(
        '--clock',
        help=
        'Molecular clock rate. This should be in units of something per year, where the "something" is the units on the tree. If not given we will attempt to estimate this by RTT. By default this value is held fixed; pass --floating_clock_rate to treat it as a starting point instead.',
        default=None,
        type=float)

    parser.add_argument(
        '--clock_filter_iqd',
        default=0.0,
        type=float,
        help=(
            "Discard the date of any tip whose root-to-tip divergence sits "
            "more than this many interquartile ranges from the regression "
            "line, before fitting. Those tips keep their place in the tree "
            "and simply become undated. 0, the default, keeps every date. "
            "4 is a reasonable value if you suspect swapped or mistyped "
            "metadata: on datasets with a tight clock it identifies "
            "genuinely wrong dates with high precision and recovers much of "
            "the damage they do, but it finds only the worst of them -- two "
            "sequences of similar divergence can swap dates and leave almost "
            "no trace in this statistic -- so it is a mitigation rather than "
            "a fix. On the Ebola test data 4 drops 20 of 362 tips and 2 "
            "drops 32, so values below 4 are aggressive."))

    parser.add_argument(
        '--variance_dates',
        default=3.0,
        type=float,
        help=
        ("How uncertain the reported tip dates are taken to be, in days: the "
         "standard deviation of the date likelihood for a tip with a full "
         "date. For a tip dated only to a month or a year this is combined "
         "in quadrature with half the width of that interval, so an "
         "imprecise date is dominated by its own window and raising this "
         "value does not inflate it. The old default of 0.3 treated a full "
         "date as known to within about seven hours, which over-constrains "
         "the fit: the tip dates then pin each root-to-tip total so tightly "
         "that the mutation counts have almost no say in how that total is "
         "divided among the branches on the path."))

    parser.add_argument(
        '--steps',
        default=20000,
        type=int,
        help=
        "Upper bound on the number of SVI steps. By default this is a ceiling, "
        "not a target: fitting stops earlier once the predicted dates stop "
        "changing by more than --convergence_tol_days, since extra steps beyond "
        "that point change the answer only in ways too small to matter. Pass "
        "--disable_early_stopping to always run exactly this many steps, e.g. "
        "for a reproducible step count.")

    parser.add_argument('--lr',
                        default=0.03,
                        type=float,
                        help="Adam learning rate")

    parser.add_argument(
        '--convergence_rate_tol',
        default=0.002,
        type=float,
        help=
        ("Relative change in the fitted clock rate below which it counts as "
         "settled, for early stopping. Checked alongside "
         "--convergence_tol_days because node dates can look stable while "
         "the rate is still moving, and the dates are more sensitive to the "
         "rate than a test on the dates alone can detect."))

    parser.add_argument(
        '--convergence_tol_days',
        default=0.1,
        type=float,
        help=
        ("Stop once the mean absolute change in predicted node dates between "
         "convergence checks falls below this many days. A looser 1 day "
         "stopped too early: the node dates had settled but the clock rate "
         "was still moving, and the rate is what the dates are most "
         "sensitive to. On one real dataset a 5%% error in the fitted rate "
         "cost 3 days of median disagreement with treetime, and tightening "
         "this from 1 to 0.1 took that disagreement from 11.3 days to 8.1. "
         "Simulated benchmarks improve too, from 14.9 to 14.1 days mean "
         "absolute error, for about 11%% more runtime."))

    parser.add_argument(
        '--convergence_check_every',
        default=50,
        type=int,
        help="How often, in steps, to evaluate the early-stopping criterion.")

    parser.add_argument(
        '--convergence_patience',
        default=3,
        type=int,
        help=
        "Number of consecutive checks (each --convergence_check_every steps apart) "
        "that must satisfy both --convergence_tol_days and "
        "--convergence_rate_tol before stopping early.")

    parser.add_argument(
        '--disable_early_stopping',
        action='store_true',
        help=
        "Always run the full --steps, ignoring the convergence criterion. Use this "
        "if you need an exact, reproducible number of SVI steps.")

    parser.add_argument('--name_all_nodes',
                        action='store_true',
                        help="Should we name all nodes in the output tree?")

    parser.add_argument(
        '--expected_min_between_transmissions',
        default=3,
        type=int,
        help="Floor, in days, under each branch's starting duration when "
        "--initialise clock is used. It has no effect under the default "
        "tip-date initialisation.")

    parser.add_argument(
        '--only_use_full_dates',
        action='store_true',
        help="Only use full dates, given to the precision of a day")

    parser.add_argument(
        '--output_unit',
        type=str,
        help="Unit for the output branch lengths on the time tree.",
        choices=["days", "years"],
        default="days")

    parser.add_argument(
        '--multiply_date_precision',
        action='store_true',
        help=
        "Restore the old date-likelihood scale, --variance_dates multiplied "
        "by each tip's precision window, rather than the two combined in "
        "quadrature. Multiplying conflates the width of the reported interval "
        "with how far a reported date can be from the truth for other "
        "reasons, so raising one inflates the other: at the default it made a "
        "year-only date uncertain to within ten years.")

    parser.add_argument(
        '--variance_on_clock_rate',
        action='store_true',
        help=("Will cause the clock rate to be "
              "drawn from a random distribution with a learnt variance. "
              "Requires --floating_clock_rate, since a fixed rate "
              "has no variance to learn."))

    parser.add_argument(
        '--root_date_prior_scale_days',
        default=36500.0,
        type=float,
        help=("Scale of the Cauchy prior on the root date, in days before "
              "the oldest tip. Heavy-tailed, so this mainly sets where the "
              "prior stops being flat rather than how far back the root may "
              "go; the default of 100 years is uninformative for almost any "
              "tree."))

    parser.add_argument(
        '--initialise',
        choices=('tip-dates', 'clock'),
        default='tip-dates',
        help=(
            "Where the fit starts. 'tip-dates' puts every tip on its own "
            "reported date and gives each internal node the mean over its "
            "children of the child's position less what that child's "
            "mutations represent at the clock rate. 'clock' is the older "
            "behaviour, which ignores the tip dates: every branch starts at "
            "its own mutations over the clock rate and the root at one "
            "reference tip's divergence over that rate. The root date barely "
            "moves during a fit, so this choice largely decides where it "
            "ends up."))

    parser.add_argument(
        '--initial_branch_floor',
        choices=('positive', 'mutations'),
        default='positive',
        help=(
            "The floor the tip-date initialiser puts under each branch. "
            "'positive' keeps only what is needed to keep durations positive "
            "and lets the tip-date estimate stand. 'mutations' additionally "
            "requires at least the time that branch's own mutations "
            "represent at the clock rate, which pushes a child later "
            "wherever the tip dates put it earlier than the mutations allow, "
            "and so inflates the tree wherever the clock estimate is off."))

    parser.add_argument(
        '--floating_clock_rate',
        action='store_true',
        help=(
            "Fit the clock rate as a free parameter instead of holding it at "
            "the estimate. This was the behaviour before the rate was fixed "
            "by default. It lets the fit recover from a poor starting "
            "estimate, at the cost of a rate biased low: fitting one free "
            "duration per branch alongside a single shared rate is the "
            "classic incidental-parameters problem, and the free fit landed "
            "below the reference rate on 18 of 24 real datasets."))

    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help=
        ("Will attempt to use the GPU. You will need a version of CUDA installed to suit Numpyro."
         ))

    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help=
        "This flag will trigger the use of Weights and Biases to log the fitting process. This must be installed with 'pip install wandb'"
    )

    parser.add_argument('--wandb_project_name',
                        default="chronumental",
                        type=str,
                        help="Wandb project name")

    parser.add_argument('--clipped_adam',
                        action='store_true',
                        help=("Will use the clipped version of Adam"))

    parser.add_argument(
        '--reference_node',
        default=None,
        type=str,
        help=
        "A reference node to use for computing dates. This should be early in the tree, and have a correct date. If not specified it will be picked as the oldest node, but often these can be metadata errors."
    )

    parser.add_argument(
        '--always_use_final_params',
        action='store_true',
        help=
        "Will force the model to always use the final parameters, rather than simply using those that gave the lowest loss"
    )

    return parser


def theil_sen_clock_rate(terminal_target_dates_array, root_to_tip):
    """Robust root-to-tip regression slope, in branch-distance units per year.

    The starting clock rate seeds the prior (Uniform(0, clock_rate * 1000))
    and every initial value in the guide, so a bad estimate here poisons the
    whole fit. Ordinary least squares is not robust: under a relaxed
    (non-strict) clock, root-to-tip divergence is noisy, and a handful of tips
    with extreme dates or divergences get high leverage and can drag the
    unweighted OLS slope far from the truth. Theil-Sen (the median of all
    pairwise slopes) has a 29% breakdown point and is far less sensitive to
    such outliers, while agreeing closely with OLS when the strict-clock
    assumption actually holds.
    """
    x = np.asarray(terminal_target_dates_array)
    y = np.asarray(root_to_tip)
    # Theil-Sen is O(n^2) in the number of tips (it forms every pairwise
    # slope), which is fine for hundreds or a few thousand tips but can
    # exhaust memory on the much larger real-world trees chronumental is
    # sometimes run on. Cap the number of points fed to it by subsampling; a
    # few thousand tips is already far more than needed to estimate one slope
    # robustly.
    max_points_for_theilsen = 5000
    if x.shape[0] > max_points_for_theilsen:
        rng = np.random.default_rng(0)
        idx = rng.choice(x.shape[0],
                         size=max_points_for_theilsen,
                         replace=False)
        x_fit, y_fit = x[idx], y[idx]
    else:
        x_fit, y_fit = x, y
    slope_per_day = stats.theilslopes(y_fit, x_fit)[0]
    return slope_per_day * helpers.DAYS_PER_YEAR


def clock_filter(days, root_to_tip, slope_per_day, iqd):
    """Mask of the tips whose root-to-tip divergence sits within `iqd`
    interquartile ranges of the clock line, or None with fewer than 20 dated
    tips.

    Uses the slope already estimated (Theil-Sen, or --clock) rather than a
    least-squares line of its own, so the outliers being hunted do not
    distort the line they are tested against. The median residual is always
    kept, so this never drops every tip.
    """
    if len(days) < 20:
        return None
    residuals = root_to_tip - slope_per_day * days
    residuals = residuals - np.median(residuals)
    q1, q3 = np.percentile(residuals, [25, 75])
    return np.abs(residuals) <= iqd * (q3 - q1)


class ConvergenceMonitor:
    """Early stopping on the fit's predicted output rather than on the loss.

    Every `check_every` steps this computes every node's predicted date on
    device, using the same path sum the model uses, and reduces the difference
    from the previous check to one scalar there. So a check costs one scalar
    host sync regardless of tree size and rides along at chunk boundaries,
    which are already the sync point.

    Two things have to settle for `patience` consecutive checks: the mean
    absolute change in node dates must fall below `tol_days`, and the relative
    change in the fitted clock rate below `rate_tol`. The dates alone are not
    enough: they can look stable while the rate is still descending, and the
    dates are more sensitive to the rate than this test is. With the rate held
    fixed, the default, the second criterion is met trivially.

    The check watches the current parameters, whereas the output uses the
    best-loss parameters. Near convergence the two agree to well within the
    tolerance.
    """

    def __init__(self, model, tol_days, rate_tol, patience, check_every):
        self.model = model
        self.tol_days = tol_days
        self.rate_tol = rate_tol
        self.patience = patience
        self.check_every = check_every
        self.last_check_step = 0
        self.previous_node_days = None
        self.previous_rate = None
        self.consecutive = 0
        self.last_change_days = None

    def update(self, step, params):
        """Run a check if one is due at `step`; True once converged."""
        if step - self.last_check_step < self.check_every:
            return False
        self.last_check_step = step
        node_days = self.model.node_dates(self.model.get_branch_times(params),
                                          params['root_date_mu'])
        converged = False
        if self.previous_node_days is not None:
            # The only place a check touches the host: one scalar.
            self.last_change_days = float(
                jnp.mean(jnp.abs(node_days - self.previous_node_days)))
            rate = float(self.model.get_mutation_rate(params))
            if self.previous_rate is None or self.previous_rate == 0:
                rate_settled = False
            else:
                rate_settled = (abs(rate - self.previous_rate) /
                                abs(self.previous_rate) < self.rate_tol)
            self.previous_rate = rate
            if self.last_change_days < self.tol_days and rate_settled:
                self.consecutive += 1
            else:
                self.consecutive = 0
            converged = self.consecutive >= self.patience
        self.previous_node_days = node_days
        return converged

    def describe(self, step, num_steps):
        return (f"Converged: mean absolute change in predicted node dates was "
                f"{self.last_change_days:.4f} days (< --convergence_tol_days "
                f"{self.tol_days}) and the clock rate moved by less than "
                f"--convergence_rate_tol {self.rate_tol}, for "
                f"{self.consecutive} consecutive checks ~{self.check_every} "
                f"steps apart. Stopping early at step {step} (of {num_steps} "
                f"requested).")


def prepend_to_file_name(full_path, to_prepend):
    if "/" in full_path:
        path, file = full_path.rsplit('/', 1)
        return f"{path}/{to_prepend}_{file}"
    else:
        return f"{to_prepend}_{full_path}"


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.convergence_patience < 1:
        parser.error("--convergence_patience must be at least 1")
    if args.convergence_check_every < 1:
        parser.error("--convergence_check_every must be at least 1")

    if args.use_wandb:
        try:
            import wandb
        except ImportError:
            raise ValueError(
                "Wandb not installed. Please install it with `pip install wandb`"
            )
        wandb.init(project=args.wandb_project_name)
        wandb.config.update(args)

    if args.dates_out is None:
        args.dates_out = prepend_to_file_name(args.dates,
                                              "chronumental_dates") + ".tsv"

    if args.tree_out is None:
        args.tree_out = prepend_to_file_name(args.tree,
                                             "chronumental_timetree")

    metadata = input_mod.get_metadata(args.dates)

    print("Reading tree")
    tree = input_mod.read_tree(args.tree)

    print("Processing dates")
    input_mod.process_dates(metadata)

    full = input_mod.get_present_dates(
        metadata, only_use_full_dates=args.only_use_full_dates)
    lookup = dict(
        zip(full['strain'],
            zip(full['processed_date'], full['processed_date_error'])))

    # Get oldest date in full, and corresponding strain:
    if args.reference_node:
        reference_point, ref_point_distance = input_mod.get_specific(
            full, tree, args.reference_node)
    else:
        reference_point, ref_point_distance = input_mod.get_oldest(full, tree)

    if args.treat_mutation_units_as_normalised_to_genome_size:
        ref_point_distance = ref_point_distance * args.treat_mutation_units_as_normalised_to_genome_size

    print(
        f"Using {reference_point}, with date: {lookup[reference_point][0]} and distance from root {ref_point_distance} as an arbitrary reference point"
    )

    target_dates, target_errors = input_mod.get_target_dates(
        tree, lookup, reference_point)
    terminal_names = sorted(target_dates.keys())

    terminal_target_dates_array = jnp.asarray(
        [float(target_dates[x]) for x in terminal_names])

    terminal_target_errors_array = jnp.asarray(
        [float(target_errors[x]) for x in terminal_names])

    print(
        f"Found {len(terminal_names)} terminals with usable date metadata{' [full date mode is on]' if args.only_use_full_dates else ''}"
    )

    initial_branch_lengths, invented_labels = (
        input_mod.get_initial_branch_lengths_and_name_all_nodes(tree))
    names_init = sorted(initial_branch_lengths.keys())
    branch_distances_array = jnp.array(
        [initial_branch_lengths[x] for x in names_init])
    if args.treat_mutation_units_as_normalised_to_genome_size:
        branch_distances_array = branch_distances_array * args.treat_mutation_units_as_normalised_to_genome_size

    name_to_pos = {x: i for i, x in enumerate(names_init)}

    # Root-to-tip sums are computed by pointer jumping over a parent-index
    # array rather than a sparse matrix of (node, ancestor) pairs. The sparse
    # form cost memory proportional to the total path length over the tree --
    # 116 million entries and 2.6 GB on a 300k-tip tree, the largest single
    # allocation in a run. See helpers.make_path_sum.
    parent_indices, root_index, max_depth = input_mod.get_parent_indices(
        tree, name_to_pos)
    n_rounds = max(1, int(math.ceil(math.log2(max_depth + 1))))
    print(f"Tree depth {max_depth}; using {n_rounds} pointer-jumping rounds")
    path_sum = helpers.make_path_sum(jnp.asarray(parent_indices), n_rounds,
                                     root_index)
    terminal_indices = jnp.asarray(
        [name_to_pos[name] for name in terminal_names], dtype=jnp.int32)

    root_to_tip = np.asarray(
        path_sum(branch_distances_array)[terminal_indices])
    genome_size = args.treat_mutation_units_as_normalised_to_genome_size

    def estimate_clock(days, root_to_tip):
        print("No clock rate specified, performing root-to-tip regression to "
              "estimate starting value")
        rate = theil_sen_clock_rate(days, root_to_tip)
        # NaN would pass a plain sign test, since every comparison with it is
        # False, and the fit would then run through to an all-NaN tree.
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(
                f"ERROR: Root-to-tip regression estimated a clock rate of "
                f"{rate}, which cannot be used. This happens when the dated "
                "tips carry no temporal signal, for instance when they all "
                "share one date. If your dataset is correct you will need to "
                "specify an initial clock rate with --clock.")
        message = f"Theil-Sen root-to-tip regression: got rate of {rate}"
        if genome_size:
            # The tree was scaled up by the genome size, so this is in
            # mutations per year; --clock takes the tree's own units.
            message += (f" mutations per year, i.e. {rate / genome_size} in "
                        "the units --clock takes")
        print(message)
        return rate

    if args.clock:
        print(f"Using clock rate {args.clock}")
        clock_rate = args.clock * genome_size if genome_size else args.clock
    else:
        clock_rate = estimate_clock(np.asarray(terminal_target_dates_array),
                                    root_to_tip)

    if args.clock_filter_iqd > 0:
        days = np.asarray(terminal_target_dates_array)
        keep = clock_filter(days, root_to_tip,
                            clock_rate / helpers.DAYS_PER_YEAR,
                            args.clock_filter_iqd)
        if keep is None:
            print("Clock filter: needs at least 20 dated tips, so keeping "
                  "every date")
        elif not keep.all():
            print(f"Clock filter: dropping the dates of {int((~keep).sum())} "
                  f"of {len(days)} tips more than {args.clock_filter_iqd} "
                  "interquartile ranges off the root-to-tip line")
            terminal_names = [n for n, k in zip(terminal_names, keep) if k]
            kept = set(terminal_names)
            target_dates = {k: v for k, v in target_dates.items() if k in kept}
            terminal_target_dates_array = terminal_target_dates_array[keep]
            terminal_target_errors_array = terminal_target_errors_array[keep]
            terminal_indices = terminal_indices[keep]
            root_to_tip = root_to_tip[keep]
            if not args.clock:
                # Estimate again without the tips just dropped.
                clock_rate = estimate_clock(
                    np.asarray(terminal_target_dates_array), root_to_tip)

    if clock_rate < 1 and not genome_size:
        raise ValueError(
            "Clock rate is less than 1 mutation per year. This probably means "
            "you need to specify a genome size with "
            "--treat_mutation_units_as_normalised_to_genome_size. If you are "
            "sure that you do not, set that parameter to 1.")

    # The clock rate is held at the estimate unless asked otherwise. Fitting
    # it jointly with one free duration per branch biases it low -- those
    # durations are incidental parameters that grow with the tree -- and on
    # real data the free fit landed below the reference rate on 18 of 24
    # datasets, costing 10% in median accuracy against published time trees.
    if args.variance_on_clock_rate and not args.floating_clock_rate:
        raise ValueError(
            "--variance_on_clock_rate needs a rate to put variance on, so it "
            "requires --floating_clock_rate.")

    # Every tip starts on its own date and each internal node takes the mean
    # over its children of the child's position less what that child's
    # mutations represent at the clock rate. Averaging over children rather
    # than over all descendant tips is the point: a tip's implied ancestor
    # date is distorted in proportion to the divergence it has accumulated,
    # so recent tips in a densely sampled clade are both numerous and badly
    # biased, and weighting each child subtree equally stops them outvoting a
    # sparse deep lineage.
    initial_branch_times_array = None
    initial_root_date = None
    if args.initialise == 'tip-dates':
        branch_time_init, initial_root_date = (
            input_mod.estimate_initial_times_local(
                tree,
                name_to_pos,
                branch_distances_array,
                target_dates,
                clock_rate,
                mutation_floor=args.initial_branch_floor == 'mutations'))
        initial_branch_times_array = jnp.asarray(
            [branch_time_init[x] for x in names_init])

    my_model = models.ChronumentalModel(
        path_sum=path_sum,
        terminal_indices=terminal_indices,
        branch_distances_array=branch_distances_array,
        terminal_target_dates_array=terminal_target_dates_array,
        terminal_target_errors_array=terminal_target_errors_array,
        clock_rate=clock_rate,
        variance_dates=args.variance_dates,
        fix_clock_rate=not args.floating_clock_rate,
        variance_on_clock_rate=args.variance_on_clock_rate,
        quadrature_date_scale=not args.multiply_date_precision,
        root_date_prior_scale=args.root_date_prior_scale_days,
        initial_branch_times_array=initial_branch_times_array,
        initial_root_date=initial_root_date,
        ref_point_distance=ref_point_distance,
        expected_min_between_transmissions=args.
        expected_min_between_transmissions)

    print("Performing SVI:")
    optimiser = optim.ClippedAdam(
        args.lr) if args.clipped_adam else optim.Adam(args.lr)
    svi = SVI(my_model.model, my_model.guide, optimiser, Trace_ELBO())
    state = svi.init(jax.random.PRNGKey(0))
    num_steps = args.steps

    # Run the fitting steps in chunks under jax.lax.scan rather than one at a
    # time from Python. The dominant cost of the old loop was not the gradient
    # computation but the per-step round trip: every step dispatched from
    # Python, and reading `loss` to test it forced a blocking device-to-host
    # transfer. Copying the whole parameter set off the device whenever the
    # loss improved, which early in fitting is most steps, cost more again.
    #
    # Chunk boundaries fall on step 0, then every tenth step, then the last,
    # so logging once per chunk gives the same lines as the old per-step loop
    # did. Best-loss parameters are tracked on device with jnp.where instead
    # of being copied to the host.
    #
    # One behaviour change: Ctrl-C is only noticed between chunks, so it now
    # stops within CHUNK_SIZE steps rather than after exactly one. A gradient
    # explosion is likewise reported once per chunk rather than once per step.
    CHUNK_SIZE = 10

    def scan_body(carry, _):
        state, lowest_loss, best_params = carry
        # svi.update returns the loss at the parameters it was given together
        # with the state after the step, so read the parameters first to pair
        # each loss with the parameters that produced it. Pairing it with the
        # updated ones let the first exploding step, whose own loss is still
        # finite, install its infinite parameters as the best.
        params = svi.get_params(state)
        state, loss = svi.update(state)
        # A NaN loss compares False here, so NaN steps never become "best".
        improved = loss < lowest_loss
        lowest_loss = jnp.where(improved, loss, lowest_loss)
        best_params = jax.tree_util.tree_map(
            lambda best, current: jnp.where(improved, current, best),
            best_params, params)
        return (state, lowest_loss, best_params), loss

    def run_chunk(state, lowest_loss, best_params, length):
        carry, losses = jax.lax.scan(scan_body,
                                     (state, lowest_loss, best_params),
                                     xs=None,
                                     length=length)
        return carry + (losses, )

    run_chunk = jax.jit(run_chunk, static_argnames=("length", ))

    # A concrete float32 rather than a Python inf: the weakly typed Python
    # value would come back as float32 after the first chunk and force any
    # later chunk of the same length to recompile.
    lowest_loss = jnp.array(jnp.inf, dtype=jnp.float32)
    # Seeded with the initial parameters, so this is never unset even if every
    # step produces a NaN loss.
    best_params = svi.get_params(state)

    monitor = None
    if not args.disable_early_stopping and args.convergence_tol_days > 0:
        monitor = ConvergenceMonitor(my_model, args.convergence_tol_days,
                                     args.convergence_rate_tol,
                                     args.convergence_patience,
                                     args.convergence_check_every)

    def log_results(step, loss, params):
        results = my_model.get_logging_results(params)
        results['step'] = step
        results['loss'] = loss
        results.move_to_end('loss', last=False)
        results.move_to_end('step', last=False)
        print("\t".join([
            f"{name}:{value:.4f}" if "." in str(value) else f"{name}:{value}"
            for name, value in results.items()
        ]))
        if args.use_wandb:
            wandb.log(results)

    step = -1
    was_interrupted = False
    try:
        while step + 1 < num_steps:
            # The first chunk is a single step so that step 0 is logged, as it
            # was before.
            length = 1 if step < 0 else min(CHUNK_SIZE, num_steps - step - 1)
            state, lowest_loss, best_params, losses = run_chunk(state,
                                                                lowest_loss,
                                                                best_params,
                                                                length=length)
            # The only host sync per chunk, against several per step before.
            losses = np.asarray(losses)
            step += length

            if np.isnan(losses).any():
                print(
                    "There may have been a 'gradient explosion'. This run may not be successful (you can stop it with ctrl-C). Suggested troubleshooting steps: specify a low learning rate e.g. '--lr 0.005'."
                )

            current_params = svi.get_params(state)
            converged = (monitor is not None
                         and monitor.update(step, current_params))
            if converged:
                print(monitor.describe(step, num_steps))
            log_results(step, losses[-1], current_params)
            if converged:
                break
    except KeyboardInterrupt:
        print(f"Interrupting model fitting after {step + 1} steps.")
        was_interrupted = True
    print("Fit completed. Extracting parameters.")

    if args.always_use_final_params:
        params = svi.get_params(state)
    else:
        params = best_params
    to_save = "y"
    if was_interrupted:
        if sys.stdin.isatty():
            to_save = ""
            while to_save.strip().lower() not in ['y', 'n']:
                to_save = input("Do you want to save the results? [y/n]")
        else:
            print("No terminal to ask on, so saving the results so far.")
    if to_save.strip().lower() == "y":
        # Every node's date comes from the same path sum the model used, so
        # the tree and the dates agree by construction. The root's own branch
        # is zeroed inside the sum, matching the zero written for it below: a
        # root has no parent for time to elapse from.
        branch_times = np.asarray(my_model.get_branch_times(params),
                                  dtype=np.float64)
        # Summed in float64 on the host, so that a date at the bottom of a
        # deep tree is not accumulated in single precision.
        node_days = helpers.path_sum_numpy(branch_times, parent_indices,
                                           n_rounds, root_index)
        node_days += float(params['root_date_mu'])
        unit = helpers.DAYS_PER_YEAR if args.output_unit == "years" else 1

        node_day_by_name = {}
        for node in helpers.preorder_traversal(tree.root):
            pos = name_to_pos[node.label]
            node.edge_length = (0.0 if node.parent is None else
                                branch_times[pos] / unit)
            # Keep setup labels for the lookup, then omit invented ones from
            # both outputs unless --name_all_nodes was requested.
            if not args.name_all_nodes and node.label in invented_labels:
                node.label = None
            else:
                node_day_by_name[node.label] = node_days[pos]

        print("Writing tree to file")
        tree.write_tree_newick(args.tree_out)
        print("")
        print(f"Wrote tree to {args.tree_out}")

        origin_date = lookup[reference_point][0]
        output_dates = {
            name: origin_date + datetime.timedelta(days=float(x))
            for name, x in node_day_by_name.items()
        }

        names, values = zip(*output_dates.items())
        output_meta = pd.DataFrame({"strain": names, "predicted_date": values})

        output_meta.to_csv(args.dates_out, sep="\t", index=False)
        print(f"Wrote predicted dates to {args.dates_out}")


if __name__ == "__main__":
    main()
