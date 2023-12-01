from rdflib import Graph, Namespace, RDF, RDFS, XSD, Literal, URIRef
import pandas

saref = Namespace("https://saref.etsi.org/core/")
ic = Namespace("https://interconnectproject.eu/example/")


def get_dev_uri(prop_type):
	get_dev_query = """
	PREFIX saref: <https://saref.etsi.org/core/>
	PREFIX ic: <https://interconnectproject.eu/example/>

	SELECT DISTINCT ?dev 
	WHERE {{
		?dev saref:makesMeasurement ?meas .
	    ?meas saref:relatesToProperty ?prop .
	    ?prop a {} .
	}} LIMIT 1
	"""
	qres = g.query(get_dev_query.format(prop_type))
	for row in qres:
		return row.dev

def add_target_to_entities(entities_df, g, prop_type, dev_uri=None, column_name="target_values"):
	values_query = """
	PREFIX saref: <https://saref.etsi.org/core/>
	PREFIX ic: <https://interconnectproject.eu/example/>

	SELECT DISTINCT ?val
	WHERE {{
		<{}> saref:makesMeasurement ?meas .
	    ?meas ic:measuredAtTime <{}> .
	    ?meas saref:hasValue ?val .
	    ?meas saref:relatesToProperty ?prop .
	    ?prop a {} .
	}}"""

	# prop_type = "<https://interconnectproject.eu/example/BatteryLevel>"
	# prop_type = "ic:Contact"
	# prop_type = "saref:Temperature"

	if dev_uri==None:
		dev_uri = get_dev_uri(prop_type)
	# dev_uri = "https://interconnectproject.eu/example/SmartSense_Multi_Sensor_47"
	print(dev_uri)

	measurements = []
	# qres = g.query(values_query.format(entities_df["timestamp_uri"][0]))
	# for time_uri in ["https://interconnectproject.eu/example/2022-07-01_090000"]:
	for time_uri in entities_df["timestamp_uri"]:
		qres = g.query(values_query.format(dev_uri, time_uri, prop_type))
		temps = []
		for row in qres:
			# print(row.val)
			temps.append(float(row.val))
		if len(temps) > 0:
			# print("average temperature =", sum(temps)/len(temps))
			measurements.append(sum(temps)/len(temps))
		else:
			measurements.append(None)

		    # measurements.append(Measurent(row.meas, row.dev))
		    # measurements[-1].set_timestamp(row.timestamp)
		    # measurements[-1].set_value(row.val)

	# print(measurements)

	entities_df[column_name] = measurements

	print(len(entities_df))

	entities_df = entities_df.dropna()

	print(len(entities_df))

	return entities_df


if __name__ == '__main__':
	# entities_fn = "entityfile_small.csv"
	# graph_fn = "samsung_try.ttl"
	# graph_fn = "small_graph.ttl"

	graph_fn = "big_graph_001.ttl"
	# entities_fn = "entities_2months_001_w_emb.csv"
	# entities_fn = "entities_full_001_w_emb.csv"
	entities_fn = "entities_full_111_w_emb.csv"

	dev_uri_base = "https://interconnectproject.eu/example/R5_"
	# r5_devs = ["180", "211", "2", "95", "154"]
	r5_devs = ["211", "154"]

	prop_type = "saref:Temperature"

	g = Graph()
	g.parse(graph_fn)
	g.bind("saref", "https://saref.etsi.org/core/")
	g.bind("ic", "https://interconnectproject.eu/example/")
	print("loaded the Graph")

	entities_df = pandas.read_csv(entities_fn)

	for r5_dev in r5_devs:
		dev_uri = dev_uri_base+r5_dev
		column_name = "target_values_" + r5_dev
		entities_df = add_target_to_entities(entities_df, g, prop_type, dev_uri, column_name)

	# entities_df.to_csv(entities_fn[:-4]+"_with_value.csv", index=False)
	entities_df.to_csv(entities_fn[:-4]+"_with_value.csv", index=False)

	# g = Graph()
	# g.bind("saref", "https://saref.etsi.org/core/")
	# g.bind("ic", "https://interconnectproject.eu/example/")



