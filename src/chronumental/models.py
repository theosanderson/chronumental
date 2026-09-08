import collections

import jax.numpy as jnp
import numpy as onp
import numpyro
import numpyro.distributions as dist

from . import helpers


class ChronumentalModel:
    """The model and its delta guide: one free duration per branch, a root
    date, and a clock rate that is either held at its estimate or fitted.

    The horseshoe variant that once shared a base class with this is gone, so
    there is one class and every input is an explicit argument.
    """

    def __init__(self,
                 path_sum,
                 terminal_indices,
                 branch_distances_array,
                 terminal_target_dates_array,
                 terminal_target_errors_array,
                 clock_rate,
                 variance_dates,
                 fix_clock_rate=True,
                 variance_on_clock_rate=False,
                 quadrature_date_scale=True,
                 root_date_prior_scale=36500.0,
                 initial_branch_times_array=None,
                 initial_root_date=None,
                 ref_point_distance=None,
                 expected_min_between_transmissions=3):
        # Sums branch times from the root to every node, by pointer jumping.
        # See helpers.make_path_sum: this replaced a sparse matrix that cost
        # memory proportional to total path length over the tree.
        self.path_sum = path_sum
        self.terminal_indices = terminal_indices
        self.branch_distances_array = branch_distances_array
        self.terminal_target_dates_array = terminal_target_dates_array
        self.terminal_target_errors_array = terminal_target_errors_array
        self.clock_rate = clock_rate
        self.variance_dates = variance_dates
        self.fix_clock_rate = fix_clock_rate
        self.variance_on_clock_rate = variance_on_clock_rate
        self.quadrature_date_scale = quadrature_date_scale
        self.root_date_prior_scale = root_date_prior_scale

        if initial_branch_times_array is not None:
            # From the tip-date initialiser, which already applies its own
            # positivity floor.
            self.initial_time = initial_branch_times_array
            self.initial_root_date = initial_root_date
        else:
            # --initialise clock: each branch at its mutations over the clock
            # rate, floored at the minimum time between transmissions, and the
            # root at one reference tip's divergence over that rate.
            self.initial_time = jnp.maximum(
                helpers.DAYS_PER_YEAR * branch_distances_array / clock_rate,
                expected_min_between_transmissions)
            self.initial_root_date = (-helpers.DAYS_PER_YEAR *
                                      ref_point_distance / clock_rate)

    def _date_scale(self):
        """Standard deviation of the date likelihood, per tip, in days.

        terminal_target_errors_array holds each tip's precision window in
        days: 1 for a full date, the month's length for a month-only date,
        the year's for a year-only one. Multiplying that by --variance_dates
        conflates two different things. The window says how wide the reported
        interval is; --variance_dates says how far a reported date can be from
        the truth for other reasons. Multiplying them means raising the second
        inflates the first.

        Adding them in quadrature keeps them separate: a full date gets about
        --variance_dates, a month-only date is dominated by its own window,
        and raising --variance_dates no longer multiplies the window.
        """
        if not self.quadrature_date_scale:
            return self.variance_dates * self.terminal_target_errors_array
        window = self.terminal_target_errors_array / 2.0
        return jnp.sqrt(self.variance_dates**2 + window**2)

    def node_dates(self, branch_times, root_date):
        """Every node's predicted date, in days relative to the reference tip.

        Shared by the likelihood, which indexes the tips out of it, and by the
        early-stopping check, which watches all of it.
        """
        return self.path_sum(branch_times) + root_date

    def calc_dates(self, branch_times, root_date):
        return self.node_dates(branch_times, root_date)[self.terminal_indices]

    def model(self):
        # Cauchy, not Normal. root_date is measured in days before the
        # oldest tip, so a Normal penalises a deep root quadratically: at
        # the old scale of 1000 days this prior asserted the root lay within
        # about three years of the oldest tip, which on a tree spanning
        # centuries dominated everything else in the objective and dragged
        # the root forward. A Cauchy's penalty grows logarithmically, so a
        # tree far deeper than the scale costs a little rather than a lot,
        # and the scale stops having to be guessed from the tree's depth.
        root_date = numpyro.sample(
            "root_date", dist.Cauchy(loc=0.0,
                                     scale=self.root_date_prior_scale))

        n_branches = self.branch_distances_array.shape[0]
        branch_times = numpyro.sample(
            "latent_time_length",
            dist.Uniform(low=onp.zeros(n_branches),
                         high=onp.ones(n_branches) * 365 * 10000))

        if self.fix_clock_rate:
            mutation_rate = self.clock_rate
        else:
            mutation_rate = numpyro.sample(
                "latent_mutation_rate",
                dist.Uniform(low=0.0, high=self.clock_rate * 1000.0))

        expected_mutations = (mutation_rate * branch_times /
                              helpers.DAYS_PER_YEAR)
        numpyro.sample("branch_distances",
                       dist.Poisson(expected_mutations),
                       obs=self.branch_distances_array)

        numpyro.sample("final_dates",
                       dist.Normal(self.calc_dates(branch_times, root_date),
                                   self._date_scale()),
                       obs=self.terminal_target_dates_array)

    def guide(self):
        root_date_mu = numpyro.param("root_date_mu", self.initial_root_date)
        numpyro.sample("root_date", dist.Delta(root_date_mu))

        time_length_mu = numpyro.param("time_length_mu",
                                       self.initial_time,
                                       constraint=dist.constraints.positive)
        numpyro.sample("latent_time_length", dist.Delta(time_length_mu))

        # With the rate held fixed the model reads it straight off
        # self.clock_rate and never samples it, so the guide must not declare
        # a latent site for it either -- numpyro warns about guide sites the
        # model does not use.
        if self.fix_clock_rate:
            return

        mutation_rate_mu = numpyro.param("mutation_rate_mu",
                                         self.clock_rate,
                                         constraint=dist.constraints.positive)
        if not self.variance_on_clock_rate:
            numpyro.sample("latent_mutation_rate",
                           dist.Delta(mutation_rate_mu))
            return

        mutation_rate_sigma = numpyro.param(
            "mutation_rate_sigma",
            self.clock_rate,
            constraint=dist.constraints.positive)
        numpyro.sample(
            "latent_mutation_rate",
            dist.TruncatedNormal(mutation_rate_mu,
                                 mutation_rate_sigma,
                                 low=0.0))

    def get_branch_times(self, params):
        return params['time_length_mu']

    def get_mutation_rate(self, params):
        if self.fix_clock_rate:
            return self.clock_rate
        return params['mutation_rate_mu']

    def get_logging_results(self, params):
        results = collections.OrderedDict()
        times = self.get_branch_times(params)
        new_dates = self.calc_dates(times, params['root_date_mu'])
        errors = onp.abs(self.terminal_target_dates_array - new_dates)
        results['date_cor'] = onp.corrcoef(self.terminal_target_dates_array,
                                           new_dates)[0, 1]
        results['date_error'] = onp.mean(errors)
        results['date_error_med'] = onp.median(errors)
        # We know that there are some metadata errors, so there probably
        # should be some big errors.
        results['max_date_error'] = onp.max(errors)
        # This correlation should be relatively high.
        results['length_cor'] = onp.corrcoef(self.branch_distances_array,
                                             times)[0, 1]
        results['root_date'] = params['root_date_mu']
        results['mutation_rate'] = self.get_mutation_rate(params)
        return results
