## Chronumental
[**Chron**&#8203;ologies from mon&#8203;**umental** phylogenetic trees]

Chronumental is a tool for creating a "time-tree" (where distance on the tree represents time) from a phylogenetic distance-tree (where distance on the tree reflects a number of genetic substitutions).

What sets chronumental apart from most other tools is that it scales to extremely large trees, which can contain millions of nodes. Chronumental uses JAX, which means it can optimise the tree rapidly if you have a GPU available, and also performs well on a CPU.

Chronumental is in an early stage of development and has not been benchmarked or subject to peer-review.

### Installation

```
pip3 install chronumental
```

### Usage
This demo uses trees and metadata collated by the [UShER](https://github.com/yatisht/usher) [team](https://hgwdev.gi.ucsc.edu/~angie/UShER_SARS-CoV-2/).
```
wget https://hgwdev.gi.ucsc.edu/~angie/UShER_SARS-CoV-2/2021/10/06/public-2021-10-06.all.nwk.gz
wget https://hgwdev.gi.ucsc.edu/~angie/UShER_SARS-CoV-2/2021/10/06/public-2021-10-06.metadata.tsv.gz
chronumental --tree public-2021-10-06.all.nwk.gz --dates public-2021-10-06.metadata.tsv.gz
```

### Similar tools
[TreeTime](https://github.com/neherlab/treetime) is a much more advanced tool for inferring time trees. If you have a dataset of <10,000 rather than millions of nodes you are best off trying it.
