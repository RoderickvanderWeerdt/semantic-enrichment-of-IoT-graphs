import pandas as pd

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
from cachetools import MRUCache

def shuffle_dataset(dataset_fn):
    df = pd.read_csv(dataset_fn, sep=",")
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("data/shuffled_dataset.csv", index=False)

def make_embeddings(entities_fn="test_files/res1_entities_SMALL_CHANGED.tsv", kg_fn="test_files/res1_hp_temp_kg_SMALL_CHANGED.ttl", new_entities_fn=None, entities_column_name="power_usage", reverse=False, show_all=True):
    if new_entities_fn == None:
        new_entities_fn = entities_fn[:-4]+'_embeddings.csv'

    shuffle_dataset(entities_fn)

    data = pd.read_csv("data/shuffled_dataset.csv", sep=",")
    
    #print(data.head())
    #exit()
    if show_all:
        verb = 1
    else:
        verb = 0
    
    entities = [entity for entity in data[entities_column_name]]
    transformer = RDF2VecTransformer(
        Word2Vec(epochs=10),
        ####     RandomWalker(walkLength, numberOfWalks, ...)
        walkers=[RandomWalker(6, 6, with_reverse=reverse, n_jobs=12, md5_bytes=None)],
        verbose=verb
    )
    # kg = KG(location="http://130.37.71.132:3020", skip_predicates=["www.w3.org/1999/02/22-rdf-syntax-ns#type"])
    kg = KG(location="http://localhost:3020", skip_predicates=["www.w3.org/1999/02/22-rdf-syntax-ns#type"])
    # kg = KG(location="http://130.37.53.36:6789", skip_predicates=["www.w3.org/1999/02/22-rdf-syntax-ns#type"]) #cliopatria server on the interconnect-vu server
    # kg = KG(location="http://130.37.53.36:6789", cache=MRUCache(maxsize=2048), skip_predicates=["www.w3.org/1999/02/22-rdf-syntax-ns#type"]) #cliopatria server on the interconnect-vu server

    # print(kg.is_exist(entities))
    embeddings, literals = transformer.fit_transform(kg, entities)

    new_emb = []
    for embedding in embeddings:
        new_emb.append(embedding.tolist())
    
    # print("literals", literals)

    data['emb'] = new_emb
    data.to_csv(new_entities_fn, sep=",", index=False)
    
    # print("saving all walks...")
    with open("data/all_walks.txt", 'w') as walks_file:
        for walks_list in transformer._walks:
            for walks in walks_list:
                walk_strings = ""
                for walk in walks:
#                     for node in walk:
#                         walk_strings += node
                    walk_strings += walk
                    walk_strings += "\n"
                walks_file.write(walk_strings+'\n')
                walk_string = ""