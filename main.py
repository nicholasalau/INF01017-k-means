import pandas as pd
import argparse
import numpy as np
from algorithm.kmeans import Kmeans
from data.dataframe import DataFrame
import matplotlib.pyplot as plt

if __name__ == '__main__':
    datasets = ['PhobiasVars']
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='PhobiasVars',
                        help="The dataset to test. List of available datasets: " + str(datasets))
    args = parser.parse_args()

    if args.dataset in datasets:
        filename = ""
        delimiter = ""

        if args.dataset.strip() == "PhobiasVars":
            filename = "dataset/Phobias_Vars.txt"
            delimiter = "\t"
        #TODO: Add datasets.
            data_frame = DataFrame(pd.read_csv(filename, sep=delimiter))

            WSS = []
            k_clusters = []
            for k in range(1,12):
                kmeans = Kmeans(k, data_frame, 10)
                WSS.append(kmeans.total_intracluster_distance)
                k_clusters.append(k)

            plt.figure(figsize=(16,8))
            plt.plot(k_clusters, WSS, 'bx-')
            plt.xlabel('k')
            plt.ylabel('WSS')
            plt.title('The Elbow Method showing the optimal k')
            plt.show()
        else:
            print("The chosen dataset is not supported")
    else:
        print("Unknown error")
