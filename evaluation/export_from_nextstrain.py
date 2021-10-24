import json
nextstrain = json.load(open("global.json"))
from datetime import datetime as dt
from datetime import timedelta
import treeswift
import pandas as pd

def fromYearFraction(yearFraction):
    year = int(yearFraction)
    fraction = yearFraction - year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)
    date = startOfThisYear + (fraction * ((startOfNextYear) - (startOfThisYear)))
    #date = date.isoformat()
    date = date.strftime("%Y-%m-%d")
    return date

def decimal_year_to_iso_date(decimal_year):
        """Convert a decimal year to an ISO 8601 date."""

tree = treeswift.Tree()
parent_lookup = {}
name_to_node= {}
nodes = [nextstrain['tree']   ]

names = []
lower_dates = []
upper_dates = []
dates = []

for node in nodes:
    if len(parent_lookup.keys()) == 0:
        tree.root.label = node['name']
        name_to_node[node['name']] = tree.root
        new_node = tree.root
    else:
        new_node =  treeswift.Node(label=node['name'])
        name_to_node[node['name']] = new_node
        parent = name_to_node[parent_lookup[node['name']]]
        parent.add_child(new_node)
    

    if("children" in node):
        nodes.extend(node['children'])
        for child in node['children']:
            child_name = child['name']
            parent_lookup[child_name] = node['name']

    if 'branch_attrs' in node and "mutations" in node['branch_attrs'] and "nuc" in node['branch_attrs']['mutations']:
        num_mutations = len(node['branch_attrs']['mutations']['nuc'])
    else:
        num_mutations = 0
    new_node.edge_length = num_mutations
        


    if('node_attrs' in node):
        if "num_date" in node['node_attrs']:
            time_values = [node['node_attrs']['num_date']['value'],
            node['node_attrs']['num_date']['confidence'][0],
            node['node_attrs']['num_date']['confidence'][1]]
            date, lower, upper = [fromYearFraction(x) for x in time_values]
            names.append(node['name'])
            lower_dates.append(lower)
            upper_dates.append(upper)
            dates.append(date)

metadata = pd.DataFrame({'strain': names, 'lower_date': lower_dates, 'upper_date': upper_dates, 'date': dates})
metadata['precise_date'] = metadata['date']
metadata.loc[metadata['upper_date'] > metadata['lower_date'], 'date'] = "?"
metadata.to_csv('nextstrain_metadata.tsv', sep='\t', index=False)
tree.write_tree_newick("nextstrain_tree.nwk")

