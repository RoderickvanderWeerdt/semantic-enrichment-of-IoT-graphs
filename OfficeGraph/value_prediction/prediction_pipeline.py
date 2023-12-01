from make_embeddings import make_embeddings, epochs_make_embeddings
from epochs_prediction_mlp import epochs_perform_predictions

def calc_distribution(tss):
    distribution = [0, 0]
    for ts in tss:
        for t in ts:
            if t == 0:
                distribution[0] += 1
            else:
                distribution[1] += 1
    return (distribution, distribution[1]/(distribution[0]+distribution[1]))


def walks_pipeline(epochs, graph_c, entities_c, create_emb=True, show_all=True, walkLength=6, nWalks=6, reverse=True):
    graph_c_name = graph_c.split('/')[-1]
    entities_emb = "data/" + graph_c_name[:-4] + "_w_emb.csv"
    entities_emb = entities_file[:-4] + "_w_emb_with_value.csv"


    if create_emb:
        make_embeddings(entities_fn=entities_c, 
                        kg_fn=graph_c, 
                        new_entities_fn=entities_emb,
                        entities_column_name="timestamp_uri",reverse=reverse,
                        show_all=show_all)#, #need to show_all, only way to see time it takes to train representations

    final_results = epochs_perform_predictions(epochs, dataset_fn=entities_emb, show_all=show_all)
    return final_results


if __name__ == '__main__':

    # list_of_graphs = [ "samsung_try.ttl" ]
    list_of_graphs = [ "big_graph.ttl" ]
    # list_of_graphs = [ "big_graph_001.ttl" ]

    entities_file = "entities_full_111.csv"

    epochs = 20
    walkLength = 2
    nWalks = 25
    reverse = True

    counter = 0
    for graph in list_of_graphs:
        print(graph, '-', entities_file, "epochs=", epochs, "walkLength=", walkLength, "- nWalks=", nWalks)
        for i in range(0,1):
            results = walks_pipeline(epochs, graph, entities_file, create_emb=False, show_all=True, walkLength=walkLength, nWalks=nWalks, reverse=reverse)
            for k in range(0,9):
                results = walks_pipeline(epochs, graph, entities_file, create_emb=False, show_all=False, walkLength=walkLength, nWalks=nWalks, reverse=reverse)
            print("train_mae, test_mae, test_mape, base_mae, base_mape")

            print("--- finished: ", counter, "---")
            counter = counter+1