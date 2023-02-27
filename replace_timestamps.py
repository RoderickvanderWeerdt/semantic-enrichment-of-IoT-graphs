import random
import re
import pandas as pd

def read_file(filename):
    txt = ""
    with open(filename, 'r') as to_read:
        txt = to_read.read()
    return txt

#returns a list of all the dates in the ttl file, and the txt file
def find_dates_in_text(txt):
    dates = re.findall(r"(\d\d\d\d-\d\d-\d\dT\d\d:\d\d:\d\d)", txt)
    dates = list(set(dates))
    print("found", len(dates), "dates!")
    return dates

def create_date_random_dict(dates):
    date_rand_dict = {}
    used_randoms = []
    new_random = str(random.random())[2:10]
    for date in dates:
        while new_random in used_randoms:
            new_random = str(random.random())[2:10]
        date_rand_dict[date] = new_random
        used_randoms.append(new_random)
    return date_rand_dict

def create_date_id_dict(dates):
    date_id_dict = {}
    for date in dates:
        new_date = date.replace('-', '').replace(':', '').replace('T', '')[:-4]
        date_id_dict[date] = new_date
    return date_id_dict

def write_new_files(new_ttl_file, txt):
    with open(new_ttl_file, 'w') as new_ttl_file:
        new_ttl_file.write(txt)


if __name__ == '__main__':
    # graph_folder = "graphs_1000_Ares_Adev"
    # graph_folder = "graphs_1000_Ares_Adev"
    # graph_folder = "graphs_1000_Ares_AdevWithoutheatpumps"
    # graph_folder = "graphs_opsd_1res_1dev"
    graph_folder = "graphs_2weeks_Ares_Adev_small"
    # graph_name = "000000111"
    graph_name = "111111111"
    graph_name = "enriched"

    ttl_file = "data/"+graph_folder+"/"+graph_name+".ttl"
    # entity_file = "data/entity_files/1000_light.csv"
    entity_file = "data/entity_files/large_german_entities.csv"
    entity_file = "data/entity_files/2WEEKS.csv"

    new_ttl_file = "data/"+graph_folder+"/"+graph_name+"_uriTimeStamps.ttl"
    new_entity_file = entity_file[:-4]+"_uriTimeStamps.csv"

    change_entity_file = False

    txt = read_file(ttl_file)
    dates = find_dates_in_text(txt)

    # date_replace_dict = create_date_random_dict(dates)
    date_replace_dict = create_date_id_dict(dates)


    timestamp_prefix = "https://interconnectproject.eu/example/time_"

    for date in date_replace_dict.keys():
        # date_with_type = "\"" + date + "\"^^xsd:dateTime"
        new_date = '<'+timestamp_prefix+date_replace_dict[date]+'>'
        # txt = txt.replace(date_with_type, new_date)
        txt = txt.replace('<'+date+'>', new_date) #only works with graphs containing the timestamp URI

    if change_entity_file:
        df = pd.read_csv(entity_file, sep=",")
        new_timestamps = []
        new_uris = []
        # for timestamp, measurement_uri in zip(df["timestamp"], df["https://interconnectproject.eu/example/DEKNres4_HP_URI"]):
        for timestamp in df["timestamp"]:
            new_timestamps.append(timestamp_prefix+date_replace_dict[timestamp])
            # new_uris.append(measurement_uri.replace(timestamp, date_replace_dict[timestamp]))
        df["timestamp"] = new_timestamps
        # df["https://interconnectproject.eu/example/DEKNres4_HP_URI"] = new_uris
        df.to_csv(new_entity_file, index=False)


    write_new_files(new_ttl_file, txt)