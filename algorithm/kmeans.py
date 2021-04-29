import pandas as pd
import numpy as np
import random

class Kmeans(object):
    def __init__(self, k_clusters, dataset, max_iter):
        print("Running algorithm...")

        self.dataset = dataset
        self.initialize(k_clusters, max_iter)

    def initialize(self, k, max_iter):
        print("Max iterations: ", max_iter)
        print("Dataset: ", self.dataset)
        print("Initializing the random centroids of the", k, "clusters...")
        random.seed(9)

        # 1. Posicionar os k centróides c1,c2, ... ck, em pontos aleatórios no espaço dos dados originais (observando intervalo de valores para cada atributo descritivo das instâncias)
        centroids = []
        # Utilizando k instância(s) aleatoria(s) do dataset como centroide(s).
        centroids_pos = random.sample(range(0, len(self.dataset)), k)
        print("Random centroids indexes: ", centroids_pos)

        # Atribuindo os valores de cada centroide de acordo com o index obtido anteriormente.
        for pos in centroids_pos:
            centroids.append(self.dataset[pos])
        centroids = np.array(centroids)
        #print("Centroids: ", centroids)

        # 2. Enquanto houverem alterações nas associações de instâncias aos k clusters, faça:
        # a. Para cada instância xi:
            # i. Encontre o centróide cj mais próximo, com base na medida D: argminj D(xi,cj).
            # ii. Associe a instância xi ao cluster j (cujo centróide é o mais próximo).
        for i in range(max_iter):
            # Instâncias associadas ao centroide mais próximo.
            centroids_instances_set = self.closest_centroid(centroids)

            # b. Para cada cluster, j = 1...k :
                # i. Atualize o seu centróide cj como a média de todos os pontos associados a ele.
            centroids = self.update_centroids(centroids_instances_set)
            print("Centroids: ", centroids)
            # TODO: Add convergence point.

    def euclidian_distances(self, centroids, instance):
        # Distancia para cada centroide.
        distances = []
        for centroid in centroids:
            distances.append(np.sqrt(np.sum((centroid - instance)**2)))
        #print("Distancia para cada centroide: ", distances)
        return distances

    def update_centroids(self, new_set):
        new_centroids = []
        for c in set(new_set['cluster']):
            print("New set instance: ", c)
            current_cluster = new_set[new_set['cluster'] == c][new_set.columns[:-1]]
            print("Current cluster: ", current_cluster)
            cluster_mean = current_cluster.mean(axis=0)
            new_centroids.append(cluster_mean)
        return np.array(new_centroids)

    def closest_centroid(self, centroids):
        closest_centroids = []
        #print("Centroids: ", centroids)
        # i. Encontre o centróide cj mais próximo, com base na medida D: argminj D(xi,cj).
        for instance in self.dataset:
            D = []
            # print("Instancia do dataset: ", instance)
            D = self.euclidian_distances(centroids, instance)
            closest_centroids.append(np.argmin(D))
        print("Closest centroids: ", closest_centroids)

        # ii. Associe a instância xi ao cluster j (cujo centróide é o mais próximo).
        new_set = pd.concat([pd.DataFrame(self.dataset), pd.DataFrame(closest_centroids, columns=['cluster'])], axis=1)
        print("New set: ", new_set)
        return new_set
