
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import numpy as onp
from . import helpers
import collections

class ChronumentalModelBase(object):
    def __init__(self,**kwargs):
        self.rows = kwargs['rows']
        self.cols = kwargs['cols']
        self.branch_distances_array = kwargs['branch_distances_array']
        self.terminal_target_dates_array = kwargs['terminal_target_dates_array']
        self.terminal_target_errors_array = kwargs['terminal_target_errors_array']
        self.ref_point_distance = kwargs['ref_point_distance']

        self.set_initial_time()

    def get_logging_results(self,params):
        results = collections.OrderedDict()
        times = self.get_branch_times(params)
        new_dates = self.calc_dates(times, params['root_date_mu'])
        results['date_cor'] = onp.corrcoef(
            self.terminal_target_dates_array,
            new_dates)[0, 1]
        results['date_error']  = onp.mean(
            onp.abs(self.terminal_target_dates_array -
                    new_dates))  # Average date error should be small
        results['date_error_med']  = onp.median(
            onp.abs(self.terminal_target_dates_array -
                    new_dates))  # Average date error should be small
        
        results['max_date_error'] = onp.max(
            onp.abs(self.terminal_target_dates_array - new_dates)
        )  # We know that there are some metadata errors, so there probably should be some big errors
        results['length_cor'] = onp.corrcoef(
            self.branch_distances_array,
            times)[0, 1]  # This correlation should be relatively high
        
        results['root_date'] = params['root_date_mu']
        return results

        

class DeltaGuideWithStrictLearntClock(ChronumentalModelBase):
    def __init__(self, **kwargs):

        self.clock_rate = kwargs['model_configuration']["clock_rate"]
        self.variance_branch_length = kwargs['model_configuration']["variance_branch_length"]
        self.variance_dates = kwargs['model_configuration']['variance_dates']
        self.enforce_exact_clock = kwargs['model_configuration']['enforce_exact_clock']
        self.variance_on_clock_rate = kwargs['model_configuration']['variance_on_clock_rate']
        self.expected_min_between_transmissions = kwargs['model_configuration']['expected_min_between_transmissions']

        super().__init__(**kwargs)
       
    def get_logging_results(self, params):
        results =  super().get_logging_results(params)
        results['mutation_rate'] = self.get_mutation_rate(params)
        return results

    def set_initial_time(self):
        self.initial_time = jnp.maximum(365 * (
        self.branch_distances_array 
    ) / self.clock_rate ,self.expected_min_between_transmissions )

    def calc_dates(self,branch_lengths_array, root_date):

        calc_dates = helpers.do_branch_matmul(self.rows,        self.cols, branch_lengths_array= branch_lengths_array,
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
        root_date_mu = numpyro.param("root_date_mu", -365*self.ref_point_distance/self.clock_rate) 

        root_date = numpyro.sample("root_date",dist.Delta(root_date_mu))

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
