"""
This file will be used to explore the following datasets:
1. 'allUsers.lcl.csv'
2. 'HTRU_2.csv'
---------------------------------------------------------
For the first dataset assign '1', and for the second assign '2'
The tags given with the datasets
"""
import Functions
key = input("Please Pick The Wanted Dataset:\n"
            "\t1: allUsers.lcl.csv\n"
            "\t2: HTRU_2.csv\n")
if key == '1':
    file_name = 'allUsers.lcl.csv'
    tag_list = ['Class']
    number_of_clusters = 3
elif key == '2':
    file_name = 'HTRU_2.csv'
    tag_list = ['Class']
    number_of_clusters = 2
else:
    print("Oops! Invalid Dataset")
    file_name = None  # To disable warnings
    tag_list = None  # To disable warnings
    number_of_clusters = 0  # To disable warnings
    exit(1)
# --------------------------- Handle The Data --------------------------- #
producer = Functions.DataClustering(file_name, tag_list)

# --------------------------------- Plot --------------------------------- #
# producer.plot_data()

# ------------------------------ Algorithms ------------------------------ #
# producer.elbow_method()
# print("Silhouette Score Result: " + str(producer.silhouette_score_test())) # We got number_of_clusters
# k=number_of_clusters as we infer from the Elbow Method and the Silhouette Score
producer.set_k(number_of_clusters)

# ------------------------------- K-Means ------------------------------- #
mi_kmeans = producer.k_means_clustering()
# Compute how well the clustering method fits the external classification
print("MI of K-Means: " + str(mi_kmeans))

# ---------------------------- Fuzzy C Means ---------------------------- #
mi_fcm = producer.fuzzy_cmeans_clustering()
# Compute how well the clustering method fits the external classification
print("MI of Fuzzy C Means: " + str(mi_fcm))

# --------------------------------- GMM --------------------------------- #
mi_gmm = producer.gmm_clustering()
# Compute how well the clustering method fits the external classification
print("MI of GMM: " + str(mi_gmm))

# ----------------------- Hierarchical Clustering ----------------------- #
# producer.hierarchical_clustering()
mi_hc = producer.agglomerative_clustering()
# Compute how well the clustering method fits the external classification
print("MI of Hierarchical Clustering: " + str(mi_hc))

# ------------------------- Spectral Clustering ------------------------- #
mi_sc = producer.spectral_clustering()
# Compute how well the clustering method fits the external classification
print("MI of Spectral Clustering: " + str(mi_sc))

# ------------------------- DBSCAN Clustering ------------------------- #
mi_dbscan = producer.dbscan_clustering()
# Compute how well the clustering method fits the external classification
print("MI of DBSCAN Clustering: " + str(mi_dbscan))

# ------------------------------- T - Test ------------------------------- #
# We will take a few samples (13 Samples) of the MI from each of the clustering
# methods and than use their mean for the T-test
arg = producer.get_mi_matrix(13)
best = Functions.find_best_algorithm(arg[0], arg[1], int(key))
print(best)
