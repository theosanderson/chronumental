import pandas as pd
import sys
# Get first argument from command line:

argument=float(sys.argv[1])

metadata = pd.read_table('public-2021-09-15.metadata.tsv.gz').drop_duplicates().sample(frac=argument)
filename = f"metadata_subset_{argument}.tsv"
metadata.to_csv(filename, sep='\t', index=False)
import os

command = f"chronumental --tree  public-2021-09-15.all.nwk.gz --dates {filename} --steps 2000 --dates_out {filename}.out.tsv --tree_out {filename}.out.nwk"
os.system(command)
