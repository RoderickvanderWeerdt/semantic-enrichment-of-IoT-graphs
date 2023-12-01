import pandas as pd

from pyrdf2vec import RDF2VecTransformer
# from pyrdf2vec.embedders import Word2Vec, Word2Vec_Epochs
from word2vec_from_rdf2vec import Word2Vec, Word2Vec_Epochs
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

def shuffle_dataset(dataset_fn):
    df = pd.read_csv(dataset_fn, sep=",")
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("shuffled_dataset.csv", index=False)

def save_walks(walks_from_transformer):
    # print("saving all walks...")
    with open("data/all_walks.txt", 'w') as walks_file:
        for walks_list in walks_from_transformer:
            for walks in walks_list:
                walk_strings = ""
                for walk in walks:
#                     for node in walk:
#                         walk_strings += node
                    walk_strings += walk
                    walk_strings += "\n"
                walks_file.write(walk_strings+'\n')
                walk_string = ""

def make_embeddings(entities_fn, kg_fn, new_entities_fn=None, entities_column_name="power_usage", reverse=False, show_all=True):
    if new_entities_fn == None:
        new_entities_fn = entities_fn[:-4]+'_embeddings.csv'

    shuffle_dataset(entities_fn)

    data = pd.read_csv("shuffled_dataset.csv", sep=",")
    
    #print(data.head())
    #exit()
    if show_all:
        verb = 1
    else:
        verb = 0
    
    entities = [entity for entity in data[entities_column_name]]
    transformer = RDF2VecTransformer(
        Word2Vec(epochs=20),
        ####     RandomWalker(walkLength, numberOfWalks, ...)
        walkers=[RandomWalker(2, 25, with_reverse=reverse, n_jobs=8, md5_bytes=None)],
        verbose=verb
    )
    kg = KG(location=kg_fn, skip_predicates={"http://www.w3.org/1999/02/22-rdf-syntax-ns#type"},)
    
    print(kg.is_exist(entities))
    embeddings, literals = transformer.fit_transform(kg, entities)

    new_emb = []
    for embedding in embeddings:
        new_emb.append(embedding.tolist())
    
    # print("literals", literals)

    data['emb'] = new_emb
    data.to_csv(new_entities_fn, sep=",", index=False)
    
    # save_walks(transformer._walks)


def epochs_make_embeddings(epochs, entities_fn, kg_fn, new_entities_fn=None, entities_column_name="power_usage", reverse=False, show_all=True, walkLength=6, nWalks=6):
    if new_entities_fn == None:
        new_entities_fn = entities_fn[:-4]+'_embeddings.csv'

    shuffle_dataset(entities_fn)

    data = pd.read_csv("shuffled_dataset.csv", sep=",")
    
    #print(data.head())
    #exit()
    if show_all:
        verb = 1
    else:
        verb = 0
    
    entities = [entity for entity in data[entities_column_name]]
    transformer = RDF2VecTransformer(
        Word2Vec_Epochs(epochs=epochs),
        ####     RandomWalker(walkLength, numberOfWalks, ...)
        walkers=[RandomWalker(walkLength, nWalks, with_reverse=reverse, n_jobs=8, md5_bytes=None)],
        verbose=verb
    )
    kg = KG(location=kg_fn, skip_predicates={"http://www.w3.org/1999/02/22-rdf-syntax-ns#type"},)
    
    print(kg.is_exist(entities))
    epochs_embeddings, literals = transformer.fit_transform(kg, entities)

    epoch_embs = []
    for embeddings in epochs_embeddings:
        new_emb = []
        for embedding in embeddings:
            new_emb.append(embedding.tolist())
        epoch_embs.append(new_emb)
    
    # print("literals", literals)

    for embs, i in zip(epoch_embs,range(0,len(epoch_embs))):
        column_name = 'emb'
        if i > 0:
            column_name = column_name + str(i)
        data[column_name] = embs
    data.to_csv(new_entities_fn, sep=",", index=False)
    
    # save_walks(transformer._walks)