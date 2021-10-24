
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from numpyro.infer.autoguide import AutoDelta
import numpy as onp
from . import helpers

class DeltaGuideWithStrictLearntClock(object):
    def __init__(self, rows, cols, branch_distances_array, terminal_target_dates_array, terminal_target_errors_array, model_configuration, ref_point_distance):
        self.rows = rows
        self.cols = cols
        self.branch_distances_array = branch_distances_array
        self.clock_rate = model_configuration["clock_rate"]
        self.terminal_target_dates_array = terminal_target_dates_array
        self.terminal_target_errors_array = terminal_target_errors_array
        self.variance_branch_length = model_configuration["variance_branch_length"]
        self.variance_dates = model_configuration['variance_dates']
        self.ref_point_distance = ref_point_distance
        self.enforce_exact_clock = model_configuration['enforce_exact_clock']
        self.variance_on_clock_rate = model_configuration['variance_on_clock_rate']

        self.initial_time = jnp.maximum(365 * (
        branch_distances_array 
    ) / model_configuration['clock_rate'] , model_configuration['expected_min_between_transmissions'] )

    def calc_dates(self,branch_lengths_array, root_date):

        calc_dates = helpers.do_branch_matmul(self.rows, self.cols, branch_lengths_array= branch_lengths_array,
                                       final_size = self.terminal_target_dates_array.shape[0])
        return calc_dates + root_date
    
    def model(self):
        root_date = numpyro.sample(
            "root_date",
            dist.Normal( loc=0.0,  scale=1000.0))

        branch_times = numpyro.sample(
            "latent_time_length",
            dist.TruncatedNormal(low=0,
                                 loc=self.initial_time,
                                 scale=self.variance_branch_length,
                                 validate_args=True))

        if self.enforce_exact_clock:
            mutation_rate = self.clock_rate  
        else:
            mutation_rate = numpyro.sample(
                f"latent_mutation_rate",
                dist.TruncatedNormal(low=0,
                                    loc=self.clock_rate,
                                    scale=self.clock_rate,
                                    validate_args=True))

        branch_distances = numpyro.sample(
            "branch_distances",
            dist.Poisson(mutation_rate * branch_times / 365),
            obs=self.branch_distances_array)

        calced_dates = self.calc_dates(branch_times, root_date)

        final_dates = numpyro.sample(
            f"final_dates",
            dist.Normal(calced_dates,
                        self.variance_dates * self.terminal_target_errors_array),
            obs=self.terminal_target_dates_array)

    
    def guide(self):
        root_date = numpyro.param("root_date", -365*self.ref_point_distance/self.clock_rate) 

        time_length_mu = numpyro.param("time_length_mu", self.initial_time,
                                constraint=dist.constraints.positive)

        mutation_rate_mu = numpyro.param("mutation_rate_mu", self.clock_rate,
                                constraint=dist.constraints.positive)
        mutation_rate_sigma = numpyro.param("mutation_rate_sigma", self.clock_rate,
                                constraint=dist.constraints.positive)
        
        branch_times = numpyro.sample("latent_time_length",dist.Delta(time_length_mu))

        if not self.variance_on_clock_rate:
            mutation_rate = numpyro.sample("latent_mutation_rate",dist.Delta(mutation_rate_mu))
        else:
            mutation_rate = numpyro.sample(f"latent_mutation_rate", dist.TruncatedNormal(0,mutation_rate_mu,mutation_rate_sigma ))

    def get_branch_times(self , params):
        return params['time_length_mu']

    def get_mutation_rate(self , params):
        if self.enforce_exact_clock:
            return self.clock_rate
        return params['mutation_rate_mu']
 


models = {"DeltaGuideWithStrictLearntClock": DeltaGuideWithStrictLearntClock}
