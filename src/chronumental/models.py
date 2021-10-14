
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from numpyro.infer.autoguide import AutoDelta
from . import helpers

class FixedClock(object):
    def __init__(self, rows, cols, branch_distances_array, clock_rate, variance_branch_length ,variance_dates, terminal_target_dates_array):
        self.rows = rows
        self.cols = cols
        self.branch_distances_array = branch_distances_array
        self.clock_rate = clock_rate
        self.terminal_target_dates_array = terminal_target_dates_array
        self.variance_branch_length = variance_branch_length
        self.variance_dates = variance_dates

        self.initial_time = 365 * (
        branch_distances_array 
    ) / clock_rate + 3  # We add 3 to this prior because tranmsmission after zero days is relatively unlikely

    
        self.guide = AutoDelta(self.model)



    def calc_dates(self,branch_lengths_array):
        A = ((self.rows, self.cols), jnp.ones_like(self.cols))
        B = branch_lengths_array.reshape((branch_lengths_array.shape[0], 1))
        calc_dates = helpers.sp_matmul(A, B,
                                       self.terminal_target_dates_array.shape[0]).squeeze()
        return calc_dates
    
    def model(self):

        branch_times = numpyro.sample(
            "latent_time_length",
            dist.TruncatedNormal(low=0,
                                 loc=self.initial_time,
                                 scale=self.variance_branch_length,
                                 validate_args=True))

        mutation_rate = numpyro.sample(
            f"latent_mutation_rate",
            dist.TruncatedNormal(low=0,
                                 loc=self.clock_rate,
                                 scale=self.clock_rate,
                                 validate_args=True))

        branch_distances = numpyro.sample(
            "branch_distances",
            dist.Poisson(self.clock_rate * branch_times / 365),
            obs=self.branch_distances_array)

        calced_dates = self.calc_dates(branch_times)

        final_dates = numpyro.sample(
            f"final_dates",
            dist.Normal(calced_dates,
                        self.variance_dates * jnp.ones(calced_dates.shape[0])),
            obs=self.terminal_target_dates_array)

    def get_branch_times(self , params):
        return params['latent_time_length_auto_loc']
 