import pandas as pd
import argparse
import numpy as np
from algorithm.kmeans import Kmeans
from data.dataframe import DataFrame
import matplotlib.pyplot as plt

# Se der pra usar
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plotElbow(k_clusters, WSS):
    plt.figure(figsize=(16,8))
    plt.plot(k_clusters, WSS, 'bx-')
    plt.xlabel('k')
    plt.ylabel('WSS')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def plotScatter(centroids, x1, x2, alpha, colors, title, xlabel="", ylabel=""):
    plt.figure(figsize=(16,8))
    plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='black')
    plt.scatter(x1, x2, alpha=alpha, color=colors)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()

if __name__ == '__main__':
    hypothesis = [1, 2, 3, 4]

    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, default=1,
                        help="Available hypothesis: " + str(hypothesis))
    parser.add_argument("--elbow_plot", type=bool, default=False,
                        help="Either plot the Elbow Plot or not. Default is false.")
    parser.add_argument("--k_clusters", type=int, default=2,
                        help="Number of clusters. Default is 2.")
    parser.add_argument("--centroids_iterations", type=int, default=50,
                        help="Number of iteratrions to find the best centroids initialization. Default is 50.")
    args = parser.parse_args()

    if args.h in hypothesis:
        filename = ""
        delimiter = ""

        if args.h == 1:
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

                plotElbow(k_clusters, WSS)
            else:
                kmeans = Kmeans(args.k_clusters, data_frame, args.centroids_iterations)
                centroids = kmeans.centroids

                X = np.array(data_frame.get_dataframe())
                colors = kmeans.get_cluster_color()

                plotScatter(centroids, X[:, 0], X[:, 1], 0.1, colors, "H1: Happy people have interest in religion", "Happiness in life", "Religion")

        elif args.h == 2:
            filename1 = "dataset/Personality_Vars.txt"
            columns1 = ['Happiness.in.life']

            filename2 = "dataset/HobbiesAndInterests_Vars.txt"
            columns2 = ['Pets']

            delimiter = "\t"
            
            data_frame1 = DataFrame(pd.read_csv(filename1, sep=delimiter))
            data_frame2 = DataFrame(pd.read_csv(filename2, sep=delimiter))

            data_frame = DataFrame(pd.concat([
                pd.DataFrame(data_frame1.get_column('Happiness.in.life'), columns=columns1), 
                pd.DataFrame(data_frame2.get_column('Pets'), columns=columns2), 
            ], axis=1))

            if args.elbow_plot == True:

                WSS = []
                k_clusters = []
                for k in range(1,12):
                    kmeans = Kmeans(k, data_frame, args.centroids_iterations)
                    WSS.append(kmeans.total_intracluster_distance)
                    k_clusters.append(k)

                plotElbow(k_clusters, WSS)
            else:
                kmeans = Kmeans(args.k_clusters, data_frame, args.centroids_iterations)
                centroids = kmeans.centroids

                X = np.array(data_frame.get_dataframe())
                colors = kmeans.get_cluster_color()

                plotScatter(centroids, X[:, 0], X[:, 1], 0.1, colors, "H2: Happy people have interest in pets", "Happiness in life", "Pets")

        elif args.h == 3:
            filename1 = "dataset/SocioDemographic_Vars.txt"
            columns1 = ['Education']

            filename2 = "dataset/HobbiesAndInterests_Vars.txt"
            columns2 = ['Theatre','Art.exhibitions', 'Dancing', 'Reading', 'Musical.instruments']

            delimiter = "\t"
            
            data_frame1 = DataFrame(pd.read_csv(filename1, sep=delimiter))
            data_frame1.pre_process('Education', True, {'currently a primary school pupil': 1, 
                                                        'primary school': 2, 
                                                        'secondary school': 3, 
                                                        'college/bachelor degree': 4,
                                                        'masters degree': 5, 
                                                        'doctorate degree': 6 })

            data_frame2 = DataFrame(pd.read_csv(filename2, sep=delimiter))
            
            data_frame = DataFrame(pd.concat([
                pd.DataFrame(data_frame1.get_column('Education'), columns=columns1), 
                pd.DataFrame(data_frame2.get_dataframe()[columns2], columns=columns2), 
            ], axis=1))

            if args.elbow_plot == True:
                WSS = []
                k_clusters = []
                for k in range(1,13):
                    kmeans = Kmeans(k, data_frame, args.centroids_iterations)
                    WSS.append(kmeans.total_intracluster_distance)
                    k_clusters.append(k)

                plotElbow(k_clusters, WSS)


            else:
                x = data_frame.get_dataframe().loc[:, columns2].values
                data_frame = DataFrame(pd.DataFrame(StandardScaler().fit_transform(x), columns=columns2))

                x = data_frame.get_dataframe().loc[:, columns2].values
                data_frame = DataFrame(pd.DataFrame(x, columns=columns2))

                pca = PCA(n_components=2)
                PCs = pca.fit_transform(x)

                data_frame = DataFrame(pd.DataFrame(data = PCs, columns = ['PC1', 'PC2']))

                kmeans = Kmeans(args.k_clusters, data_frame, args.centroids_iterations)

                centroids = kmeans.centroids

                X = np.array(data_frame.get_dataframe())
                y = kmeans.centroids_instances_set.loc[:, ['cluster' ]].values

                colors = kmeans.get_cluster_color()

                X = np.array(data_frame.get_dataframe())
                
                plotScatter(centroids, X[:, 0], X[:, 1], 1, colors, "H3", "PC1", "PC2")

        elif args.h == 4:
            filename1 = "dataset/SocioDemographic_Vars.txt"
            columns1 = ['Education']

            filename2 = "dataset/HobbiesAndInterests_Vars.txt"
            columns2 = ['Theatre','Art.exhibitions', 'Dancing', 'Internet', 'Gardening']

            delimiter = "\t"
            
            data_frame1 = DataFrame(pd.read_csv(filename1, sep=delimiter))
            data_frame1.pre_process('Education', True, {'currently a primary school pupil': 1, 
                                                        'primary school': 2, 
                                                        'secondary school': 3, 
                                                        'college/bachelor degree': 4,
                                                        'masters degree': 5, 
                                                        'doctorate degree': 6 })

            data_frame2 = DataFrame(pd.read_csv(filename2, sep=delimiter))
            
            data_frame = DataFrame(pd.concat([
                pd.DataFrame(data_frame1.get_column('Education'), columns=columns1), 
                pd.DataFrame(data_frame2.get_dataframe()[columns2], columns=columns2), 
            ], axis=1))

            if args.elbow_plot == True:
                WSS = []
                k_clusters = []
                for k in range(1,13):
                    kmeans = Kmeans(k, data_frame, args.centroids_iterations)
                    WSS.append(kmeans.total_intracluster_distance)
                    k_clusters.append(k)

                plotElbow(k_clusters, WSS)


            else:
                x = data_frame.get_dataframe().loc[:, columns2].values
                data_frame = DataFrame(pd.DataFrame(StandardScaler().fit_transform(x), columns=columns2))

                x = data_frame.get_dataframe().loc[:, columns2].values
                data_frame = DataFrame(pd.DataFrame(x, columns=columns2))

                pca = PCA(n_components=2)
                PCs = pca.fit_transform(x)

                data_frame = DataFrame(pd.DataFrame(data = PCs, columns = ['PC1', 'PC2']))

                kmeans = Kmeans(args.k_clusters, data_frame, args.centroids_iterations)

                centroids = kmeans.centroids

                X = np.array(data_frame.get_dataframe())
                y = kmeans.centroids_instances_set.loc[:, ['cluster' ]].values

                colors = kmeans.get_cluster_color()

                X = np.array(data_frame.get_dataframe())
                
                plotScatter(centroids, X[:, 0], X[:, 1], 1, colors, "H4", "PC1", "PC2")

        else:
            print("The chosen hypothesis is not supported")
    else:
        print("Unknown error")
