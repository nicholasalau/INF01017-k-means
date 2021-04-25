import pandas as pd
import numpy as np

class Kmeans(object):
    def __init__(self, k_clusters, dataset):
        print("Running algorithm...")

        self.dataset = dataset
        self.initialize(k_clusters)

    def initialize(self, k):
        print("Initializing the random centroids of the", k, "clusters...")
        np.random.seed(9)
        # 1. Posicionar os k centróides c1,c2, ... ck, em pontos aleatórios no espaço dos dados originais (observando intervalo de valores para cada atributo descritivo das instâncias)
        centroids = {}
        values = []

        for i in range(k):
            values.clear()
            #TODO: Pegar range (1,5) direto do dataset.
            for j in range(len(self.dataset)):
                values.append(np.random.randint(1, 5))
            print(i)
            print("values: ", values)
            centroids[i] = values
            print("centroid[",i,"]", centroids[i])

        print(centroids)

        # 2. Enquanto houverem alterações nas associações de instâncias aos k clusters, faça:
            # a. Para cada instância xi:
                # i. Encontre o centróide cj mais próximo, com base na medida D: argminj D(xi,cj).
                # ii. Associe a instância xi ao cluster j (cujo centróide é o mais próximo).
        #D = []
        for instance in self.dataset:
            self.euclidian_distance(centroids, self.dataset[instance])

    def euclidian_distance(self, centroids, instance):
        # Distancia para cada centroide
        distances = []
        #for i in centroids.keys():
        #    distances.append(np.sqrt(np.sum((instance - B)**2)))
        #return distances

