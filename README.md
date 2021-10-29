## Chronumental
***Chron**&#8203;ologies from mon&#8203;**umental** phylogenetic trees*

<hr>

Chronumental is a tool for creating a "time-tree" (where distance on the tree represents time) from a phylogenetic distance-tree (where distance on the tree reflects a number of genetic substitutions).

What sets Chronumental apart from most other tools is that it scales to extremely large trees, which can contain millions of nodes. Chronumental uses JAX to represent the task of computing a time tree in a differentiable graph for efficient calculation on a CPU or GPU.

### Installation

#### Method 1: Using pipx (recommended for basic use - installs in its own isolated environment)
```
pip install --local pipx
pipx install  chronumental
```

#### Method 2: In your python environment
```
pip install chronumental
```

### Usage
This demo uses trees and metadata collated by the [UShER](https://github.com/yatisht/usher) [team](https://hgwdev.gi.ucsc.edu/~angie/UShER_SARS-CoV-2/).
```
wget https://hgwdev.gi.ucsc.edu/~angie/UShER_SARS-CoV-2/2021/10/06/public-2021-10-06.all.nwk.gz
wget https://hgwdev.gi.ucsc.edu/~angie/UShER_SARS-CoV-2/2021/10/06/public-2021-10-06.metadata.tsv.gz
chronumental --tree public-2021-10-06.all.nwk.gz --dates public-2021-10-06.metadata.tsv.gz --steps 100
```

### Parameters
```
usage: chronumental [-h] --tree TREE --dates DATES [--dates_out DATES_OUT] [--tree_out TREE_OUT] [--clock CLOCK] [--variance_dates VARIANCE_DATES] [--variance_branch_length VARIANCE_BRANCH_LENGTH]
                    [--steps STEPS] [--lr LR] [--name_all_nodes] [--expected_min_between_transmissions EXPECTED_MIN_BETWEEN_TRANSMISSIONS] [--only_use_full_dates] [--model MODEL] [--output_unit {days,years}]
                    [--variance_on_clock_rate] [--enforce_exact_clock] [--use_gpu] [--use_wandb] [--wandb_project_name WANDB_PROJECT_NAME] [--clipped_adam]

Convert a distance tree into time tree with distances in days.

optional arguments:
  -h, --help            show this help message and exit
  --tree TREE           an input newick tree, potentially gzipped, with branch lengths reflecting genetic distance
  --dates DATES         A metadata file with columns strain and date (in "2020-01-02" format, or less precisely, "2021-01", "2021")
  --dates_out DATES_OUT
                        Output for date tsv (otherwise will use default)
  --tree_out TREE_OUT   Output for tree (otherwise will use default)
  --clock CLOCK         Molecular clock rate. This should be in units of something per year, where the "something" is the units on the tree. If not given we will attempt to estimate this by RTT. This is only
                        used as a starting point, unless you supply --enforce_exact_clock.
  --variance_dates VARIANCE_DATES
                        Scale factor for date distribution. Essentially a measure of how uncertain we think the measured dates are.
  --variance_branch_length VARIANCE_BRANCH_LENGTH
                        Scale factor for branch length distribution. Essentially how close we want to match the expectation of the Poisson.
  --steps STEPS         Number of steps to use for the SVI
  --lr LR               Adam learning rate
  --name_all_nodes      Should we name all nodes in the output tree?
  --expected_min_between_transmissions EXPECTED_MIN_BETWEEN_TRANSMISSIONS
                        For forming the prior, an expected minimum time between transmissions in days
  --only_use_full_dates
                        Should we only use full dates?
  --model MODEL         Model type to use
  --output_unit {days,years}
                        Unit for the output branch lengths on the time tree.
  --variance_on_clock_rate
                        Will cause the clock rate to be drawn from a random distribution with a learnt variance.
  --enforce_exact_clock
                        Will cause the clock rate to be exactly fixed at the value specified in clock, rather than learnt
  --use_gpu             Will attempt to use the GPU. You will need a version of CUDA installed to suit Numpyro.
  --use_wandb           This flag will trigger the use of Weights and Biases to log the fitting process. This must be installed with 'pip install wandb'
  --wandb_project_name WANDB_PROJECT_NAME
                        Wandb project name
  --clipped_adam        Will use the clipped version of Adam
```

### Similar tools
[TreeTime](https://github.com/neherlab/treetime) is a more advanced tool for inferring time trees. If you have a dataset of e.g. <10,000 rather than millions of nodes you are definitely best off trying it. The TreeTime README also links to other similar tools.
