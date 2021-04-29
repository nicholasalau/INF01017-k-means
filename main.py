import pandas as pd
import argparse
import numpy as np
from algorithm.kmeans import Kmeans
from data.dataframe import DataFrame

if __name__ == '__main__':
    datasets = ['PhobiasVars']
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_clusters", type=int, default=2,
                        help="The number of clusters used in the execution. The default is 2.")
    parser.add_argument("--dataset", type=str, default='PhobiasVars',
                        help="The dataset to test. List of available datasets: " + str(datasets))
    parser.add_argument("--max_iter", type=int, default=20,
                        help="The maximum number of iterations till the centroids converge.")
    args = parser.parse_args()

    # Data set apenas para testar o algoritmo. Phobias_Vars.txt
    test_data = pd.DataFrame({
        '1': [1, 1, 1, 1, 1, 5, 3, 1, 3, 2],
        '2': [2, 1,	1, 3, 5, 5, 5, 4, 5, 5]
    })

    if args.dataset in datasets:
        filename = ""
        delimiter = ""

        if args.dataset.strip() == "PhobiasVars":
            filename = "dataset/Phobias_Vars.txt"
            delimiter = "\t"
        #TODO: Add datasets.
            data_frame = DataFrame(np.array(pd.read_csv(filename, sep=delimiter)))
            print(data_frame.get_dataframe())

            kmeans = Kmeans(args.k_clusters, data_frame.get_dataframe(), args.max_iter)

        else:
            print("The chosen dataset is not supported")
    else:
        print("Unknown error")
