import json
nextstrain = json.load(open("global.json"))
from datetime import datetime as dt
from datetime import timedelta


def fromYearFraction(yearFraction):
    year = int(yearFraction)
    fraction = yearFraction - year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)
    date = startOfThisYear + (fraction * ((startOfNextYear) - (startOfThisYear)))
    date = date.isoformat()
    return date

def decimal_year_to_iso_date(decimal_year):
        """Convert a decimal year to an ISO 8601 date."""

nodes = [nextstrain['tree']   ]
for node in nodes:
    if("children" in node):
        nodes.extend(node['children'])
    if('node_attrs' in node and "num_date" in node['node_attrs']):
        time_values = [node['node_attrs']['num_date']['value'],
        node['node_attrs']['num_date']['confidence'][0],
        node['node_attrs']['num_date']['confidence'][1]]
        print(node['name']+"\t"+"\t".join([fromYearFraction(x) for x in time_values]))
