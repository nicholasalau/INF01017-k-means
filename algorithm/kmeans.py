import pandas as pd
import numpy as np
import random
random.seed(27)

# TODO: Inicializar as centróides N vezes, 
# selecionando instancias aleatórias, 
# executar o algoritmo k-means e calcular a dissimilaridade intracluster V e verificar qual configuração gerou uma distâncias intracluster mínima, 
# e daí se usa esse configuração 

class Kmeans(object):
    def __init__(self, k_clusters, dataset, max_attempts):
        print("Running algorithm... for", k_clusters, "clusters")

        self.max_attempts = max_attempts

        self.dataset = dataset
        self.values = dataset.get_values()
        self.k_clusters = k_clusters

        self.intracluster_distance = []
        self.total_intracluster_distance = 0

        for attempt in range(1, max_attempts+1):
            self.current_attempt = attempt
            self.initialize(k_clusters)

    def initialize(self, k):
        # print("Max iterations: ", max_iter)
        # print("Dataset: ", self.values)
        print("Initializing the random centroids of the", k, "clusters. Attempt", self.current_attempt ,"of", self.max_attempts)

        # 1. Posicionar os k centróides c1,c2, ... ck, em pontos aleatórios no espaço dos dados originais (observando intervalo de valores para cada atributo descritivo das instâncias)
        centroids = []

        # Utilizando k instância(s) aleatoria(s) do dataset como centroide(s).
        centroids_pos = self.dataset.get_random_instances(k)
        # print("Random centroids indexes: ", centroids_pos)

        # Atribuindo os valores de cada centroide de acordo com o index obtido anteriormente.
        for pos in centroids_pos:
            centroids.append(self.values[pos])
        centroids = np.array(centroids)
        # print("Centroids: ", centroids)

        # 2. Enquanto houverem alterações nas associações de instâncias aos k clusters, faça:
        # a. Para cada instância xi:
            # i. Encontre o centróide cj mais próximo, com base na medida D: argminj D(xi,cj).
            # ii. Associe a instância xi ao cluster j (cujo centróide é o mais próximo).
        iterations = 1
        while True:
            # Instâncias associadas ao centroide mais próximo.
            centroids_instances_set = self.closest_centroid(centroids)

            # b. Para cada cluster, j = 1...k :
                # i. Atualize o seu centróide cj como a média de todos os pontos associados a ele.
            new_centroids = self.update_centroids(centroids_instances_set)

            if self.check_convergence(centroids, new_centroids):
                # print("Total iterations: ", iterations)
                self.get_wss(new_centroids, centroids_instances_set)
                break
            else:
                centroids = new_centroids
                iterations += 1
            # print("Centroids: ", centroids)
            # TODO: Add convergence point.

    def euclidian_distances(self, centroids, instance):
        # Distancia para cada centroide.
        distances = []
        for centroid in centroids:
            distances.append(np.sqrt(np.sum((centroid - instance)**2)))
        # print("Distancia para cada centroide: ", distances)
        
        return distances

    def update_centroids(self, new_set):
        new_centroids = []
        for c in set(new_set['cluster']):
            # print("New set instance: ", c)
            current_cluster = new_set[new_set['cluster'] == c][new_set.columns[:-1]]
            # print("Current cluster: ", current_cluster)
            cluster_mean = current_cluster.mean(axis=0)
            new_centroids.append(cluster_mean)
        return np.array(new_centroids)

    def closest_centroid(self, centroids):
        closest_centroids = []
        #print("Centroids: ", centroids)
        # i. Encontre o centróide cj mais próximo, com base na medida D: argminj D(xi,cj).

        for instance in self.values:
            D = []
            # print("Instancia do dataset: ", instance)
            D = self.euclidian_distances(centroids, instance)
            closest_centroids.append(np.argmin(D))
        # print("Closest centroids: ", closest_centroids)

        # ii. Associe a instância xi ao cluster j (cujo centróide é o mais próximo).
        new_set = pd.concat([pd.DataFrame(self.values, columns=self.dataset.get_attributes()), pd.DataFrame(closest_centroids, columns=['cluster'])], axis=1)
        # print("New set: ", new_set)
        return new_set

    def check_convergence(self, old_centroids, new_centroids):
        # Compara a lista de centróides antes e depois da iteração para verificar se há atribuições de novas instâncias entre os clusters
        return ((old_centroids == new_centroids).all())

    def get_wss(self, centroids, centroids_instances_set):
        intracluster_distance = np.zeros(self.k_clusters)

        cluster = centroids_instances_set.cluster
        centroids_instances_set = centroids_instances_set.drop('cluster', axis=1)

        for i, v in enumerate(centroids_instances_set.values):
            instance_cluster = cluster[i]
            instance_centroid = centroids[instance_cluster]

            distance = np.sqrt(np.sum((instance_centroid - v)**2))
            
            intracluster_distance[instance_cluster] += distance
            total_intracluster_distance = np.sum(intracluster_distance)

        if self.total_intracluster_distance == 0 or self.total_intracluster_distance > total_intracluster_distance:
            print("Found a better centroid inicialization at attempt ", self.current_attempt)
            self.intracluster_distance = intracluster_distance
            self.total_intracluster_distance = total_intracluster_distance


            