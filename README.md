## Chronumental
[**Chron**&#8203;ologies from mon&#8203;**umental** phylogenetic trees]

Chronumental is a tool for creating a "time-tree", where each internal node is annotated with position according to its date from a phylogenetic distance-tree, where each node is positioned according to genetic distance.

What sets chronumental apart from most other tools is that it scales to extremely large trees, which can contain millions of nodes. Chronumental uses JAX, which means it can optimise the tree rapidly if you have a GPU available, but also performs reasonably with a CPU.

Chronumental is in an early stage of development and has not been benchmarked or subject to peer-review.

### Installation

```
pip3 install chronumental
```

### Usage

```
wget https://hgwdev.gi.ucsc.edu/~angie/UShER_SARS-CoV-2/2021/10/06/public-2021-10-06.all.nwk.gz
wget https://hgwdev.gi.ucsc.edu/~angie/UShER_SARS-CoV-2/2021/10/06/public-2021-10-06.metadata.tsv.gz
chronumental --tree public-2021-10-06.all.nwk.gz --dates public-2021-10-06.metadata.tsv.gz --estimated_substitutions_per_year 30
```

### Similar tools
[TreeTime](https://github.com/neherlab/treetime) is a much more advanced tool for inferring time trees. If you have a dataset of thousands rather than millions of nodes you are best off trying it.
