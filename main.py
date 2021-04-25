import pandas as pd
import argparse
from algorithm.kmeans import Kmeans

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_clusters", type=int, default=2,
                        help="The number of clusters used in the execution. The default is 2.")
    args = parser.parse_args()

    # Data set apenas para testar o algoritmo. Phobias_Vars.txt
    test_data = pd.DataFrame({
        '1': [1, 1, 1, 1, 1, 5, 3, 1, 3, 2],
        '2': [2, 1,	1, 3, 5, 5, 5, 4, 5, 5]
    })

    kmeans = Kmeans(args.k_clusters, test_data)

