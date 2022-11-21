## Chronumental

***Chron**&#8203;ologies from mon&#8203;**umental** phylogenetic trees*

<hr>

Chronumental is a tool for creating a "time-tree" (where distance on the tree represents time) from a phylogenetic divergence-tree (where distance on the tree reflects a number of genetic substitutions).

What sets Chronumental apart from most other tools is that it scales to extremely large trees, which can contain millions of nodes. Chronumental uses JAX to represent the task of computing a time tree in a differentiable graph for efficient calculation on a CPU or GPU.

#### [📝 Read the preprint](https://www.biorxiv.org/content/10.1101/2021.10.27.465994v1)

#### [📚 View the documentation](https://chronumental.readthedocs.io/en/latest/)


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

#### Method 3: Bioconda

Chronumental is [now](https://github.com/bioconda/bioconda-recipes/pull/34410) available on bioconda

```
conda config --add channels bioconda
conda install chronumental
```


### Usage

This demo uses trees and metadata collated by the [UShER](https://github.com/yatisht/usher) [team](https://hgwdev.gi.ucsc.edu/~angie/UShER_SARS-CoV-2/).

```
wget https://hgwdev.gi.ucsc.edu/~angie/UShER_SARS-CoV-2/2021/10/06/public-2021-10-06.all.nwk.gz
wget https://hgwdev.gi.ucsc.edu/~angie/UShER_SARS-CoV-2/2021/10/06/public-2021-10-06.metadata.tsv.gz
chronumental --tree public-2021-10-06.all.nwk.gz --dates public-2021-10-06.metadata.tsv.gz --steps 100
```

📚 Please [visit our documentation page](https://chronumental.readthedocs.io/en/latest/) to learn more about the parameters you can use to control Chronumental.

### Integrations

[Taxonium](https://github.com/theosanderson/taxonium) can automatically call Chronumental, and generate a combined visualisation that allows switching between distance and time phylogenies


### Similar tools

[TreeTime](https://github.com/neherlab/treetime) is a more advanced tool for inferring time trees. If you have a dataset of e.g. <10,000 rather than millions of nodes you are definitely best off trying it. The TreeTime readme also links to other similar tools.

### Troubleshooting

- Chronumental uses the earliest date in your dataset as an anchor to calibrate everything else. If this earliest date is wrong due to a metadata error things won't work well, you can set the reference node manually with `--reference_node`
