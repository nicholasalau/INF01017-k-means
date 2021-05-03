import pandas as pd
import argparse
import numpy as np
from algorithm.kmeans import Kmeans
from data.dataframe import DataFrame
import matplotlib.pyplot as plt

if __name__ == '__main__':
    hypothesis = [1, 2, 3, 4]

    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis", type=int, default=1,
                        help="Tem que ser uma dessas: " + str(hypothesis))
    parser.add_argument("--elbow_plot", type=bool, default=False,
                        help="Either plot the Elbow Plot or not. Default is false.")
    parser.add_argument("--k_clusters", type=int, default=2,
                        help="Number of clusters. Default is 2.")
    parser.add_argument("--centroids_iterations", type=int, default=10,
                        help="Number of iteratrions to find the best centroids initialization. Default is 10.")
    args = parser.parse_args()

    if args.hypothesis in hypothesis:
        filename = ""
        delimiter = ""

        if args.hypothesis == 1:
            filename1 = "dataset/Personality_Vars.txt"
            columns1 = ['Happiness.in.life']

            filename2 = "dataset/HobbiesAndInterests_Vars.txt"
            columns2 = ['Religion']

            delimiter = "\t"
            
            data_frame1 = DataFrame(pd.read_csv(filename1, sep=delimiter))
            data_frame2 = DataFrame(pd.read_csv(filename2, sep=delimiter))

            data_frame = DataFrame(pd.concat([
                pd.DataFrame(data_frame1.get_column('Happiness.in.life'), columns=columns1), 
                pd.DataFrame(data_frame2.get_column('Religion'), columns=columns2), 
            ], axis=1))

            if args.elbow_plot == True:

                WSS = []
                k_clusters = []
                for k in range(1,12):
                    kmeans = Kmeans(k, data_frame, args.centroids_iterations)
                    WSS.append(kmeans.total_intracluster_distance)
                    k_clusters.append(k)

                plt.figure(figsize=(16,8))
                plt.plot(k_clusters, WSS, 'bx-')
                plt.xlabel('k')
                plt.ylabel('WSS')
                plt.title('The Elbow Method showing the optimal k')
                plt.show()

            else:
                kmeans = Kmeans(args.k_clusters, data_frame, args.centroids_iterations)

                centroids = kmeans.centroids

                X = np.array(data_frame.get_dataframe())

                colors = kmeans.get_cluster_color()

                print(data_frame.get_dataframe())


                plt.figure(figsize=(16,8))
                plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='black')
                plt.scatter(X[:, 0], X[:, 1], alpha=0.1, color=colors)
                plt.title("Pessoas felizes tem interesse em pets")
                plt.xlabel("Happiness in life")
                plt.ylabel("Pets")

                plt.show()

        else:
            print("The chosen hypothesis is not supported")
    else:
        print("Unknown error")
