# from make_embeddings_online import make_embeddings
from make_embeddings import make_embeddings, epochs_make_embeddings
# from hotcold_mlp import perform_prediction
from epochs_hotcold_mlp import epochs_perform_predictions

def calc_distribution(tss):
    distribution = [0, 0]
    for ts in tss:
        for t in ts:
            if t == 0:
                distribution[0] += 1
            else:
                distribution[1] += 1
    return (distribution, distribution[1]/(distribution[0]+distribution[1]))

def pipeline4(graph_c, entities_c, create_emb=True, show_all=True):
    graph_c_name = graph_c.split('/')[-1]
    entities_emb = "data/" + graph_c_name[:-4] + "_w_emb.csv"
    # entities_emb = "data/1000_w_random_emb.csv"

    if create_emb:
        make_embeddings(entities_fn=entities_c, 
                        kg_fn=graph_c, 
                        new_entities_fn=entities_emb,
                        entities_column_name="timestamp",reverse=True,
                        # entities_column_name="https://interconnectproject.eu/example/DEKNres4_HP_URI",reverse=True,
                        show_all=True) #need to show_all, only way to see time it takes to train representations
                        # show_all=show_all)

    final_results = perform_prediction(dataset_fn=entities_emb, show_all=show_all)
    # print(final_results["train_result"], final_results["test_result"])
    return final_results

def epochs_pipeline(epochs, graph_c, entities_c, create_emb=True, show_all=True):
    graph_c_name = graph_c.split('/')[-1]
    entities_emb = "data/" + graph_c_name[:-4] + "_w_emb.csv"
    # entities_emb = "data/1000_w_random_emb.csv"

    if create_emb:
        epochs_make_embeddings(epochs, entities_fn=entities_c, 
                        kg_fn=graph_c, 
                        new_entities_fn=entities_emb,
                        entities_column_name="timestamp",reverse=True,
                        # entities_column_name="https://interconnectproject.eu/example/DEKNres4_HP_URI",reverse=True,
                        show_all=True) #need to show_all, only way to see time it takes to train representations
                        # show_all=show_all)

    final_results = epochs_perform_predictions(epochs, dataset_fn=entities_emb, show_all=show_all)
    # print(final_results["train_result"], final_results["test_result"])
    return final_results

def walks_pipeline(epochs, graph_c, entities_c, create_emb=True, show_all=True, walkLength=6, nWalks=6, reverse=True):
    graph_c_name = graph_c.split('/')[-1]
    entities_emb = "data/" + graph_c_name[:-4] + "_w_emb.csv"
    # entities_emb = "data/1000_w_random_emb.csv"

    if create_emb:
        epochs_make_embeddings(epochs, entities_fn=entities_c, 
                        kg_fn=graph_c, 
                        new_entities_fn=entities_emb,
                        entities_column_name="timestamp",reverse=reverse,
                        # entities_column_name="https://interconnectproject.eu/example/DEKNres4_HP_URI",reverse=True,
                        show_all=True, #need to show_all, only way to see time it takes to train representations
                        # show_all=show_all)
                        walkLength=walkLength, nWalks=nWalks)

    final_results = epochs_perform_predictions(epochs, dataset_fn=entities_emb, show_all=show_all)
    # print(final_results["train_result"], final_results["test_result"])
    return final_results


if __name__ == '__main__':
    list_of_graphs = ["data/graphs_2weeks_Ares_Adev_small/enriched.ttl", "data/graphs_2weeks_Ares_Adev_small/basic.ttl"]

    entities_file = "data/entity_files/2WEEKS.csv"

    epochs = 40
    walkLength = 2
    nWalks = 25
    reverse = True

    counter = 0
    for graph in list_of_graphs:
        print(graph, '-', entities_file, "epochs=", epochs, "walkLength=", walkLength, "- nWalks=", nWalks)
        for i in range(0,1):
            results = walks_pipeline(epochs, graph, entities_file, create_emb=True, show_all=False, walkLength=walkLength, nWalks=nWalks, reverse=reverse)
            for k in range(0,2):
                results = walks_pipeline(epochs, graph, entities_file, create_emb=False, show_all=False, walkLength=walkLength, nWalks=nWalks, reverse=reverse)
            print("--- finished: ", counter, "---")
            counter = counter+1


########PRE_EPOCHS
    # counter = 1
    # for graph in list_of_graphs:
    #     for i in range(0,1):
    #         # print(graph, '-', entities_file   )
    #         results = pipeline4(graph, entities_file, create_emb=True, show_all=False)
    #         print(results["train_result"], results["test_result"])
    #         # print("targets x:", calc_distribution(results["targets_x"]))
    #         # print("targets y:", calc_distribution(results["targets_y"]))
    #         res = str(results["train_result"]).replace('.', ',') +"\t"+ str(results["test_result"]).replace('.', ',')
    #         for k in range(0,2):
    #             results = pipeline4(graph, entities_file, create_emb=False, show_all=False)
    #             print(results["train_result"], results["test_result"])
    #             # print("targets x:", calc_distribution(results["targets_x"]))
    #             # print("targets y:", calc_distribution(results["targets_y"]))
    #             res += "\t" + str(results["train_result"]).replace('.', ',') +"\t"+ str(results["test_result"]).replace('.', ',')        
    #         print(res)
    #         # print("--- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    #         # print("--- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    #         print("--- finished: ", counter, "---")
    #         # print("--- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    #         # print("--- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    #         counter = counter+1
    # #         with open("all_results_alldev_timestamp.csv", 'a') as results_file:
    # #             results_file.write(graph+';'+str(results["train_result"])+';'+str(results["test_result"])+'\n')