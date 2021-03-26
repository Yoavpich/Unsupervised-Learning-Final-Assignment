import pandas
import matplotlib.pyplot as plt
from fcmeans import FCM
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler


# In this class we get the data, normalize it and reduce dimensions
class DataHandling:
    # file_name = the name of the CSV file that contain the dataset
    # tag_list = a list of the names of the tags (the column name)
    def __init__(self, file_name, tag_list):
        self.data = None
        self.tags = None
        self.get_data(file_name, tag_list)

    # This function made to duplicate the Dataset into an dataframe
    def get_data(self, file_name, tag_list):
        # To use only the first N rows, use 'nrows=N' in read_csv
        df = pandas.read_csv(file_name, sep=",", nrows=40000, low_memory=False)
        # Convert the above columns to numbers and fill the empty cells
        self.fill_missing_data(df)
        # Save the data and tags dataframes
        self.data = df.drop(tag_list, axis=1)
        self.tags = df[tag_list]
        # Normalize and reduce dimensions of both data and tags
        self.normalize_and_reduce()

    def normalize_and_reduce(self):
        # Data
        self.data = StandardScaler().fit_transform(self.data)
        pca = PCA(n_components=2)
        self.data = pca.fit_transform(self.data)
        # Tags
        self.tags = StandardScaler().fit_transform(self.tags)
        pca = PCA(n_components=1)
        self.tags = pca.fit_transform(self.tags)
        # Turn it into a list
        self.tags = [item for sublist in self.tags for item in sublist]

    # Get the mean of column 'col' with missing data ('?')
    # 'col" should be a list
    @staticmethod
    def get_mean_of_column(col: list):
        # If summ and count equals to 0 in the end, it will throw exception instead of return 0
        # So we will set count to be -1 so if all the variables equals to '?' it will return 0 / (-1) = 0
        count = -1
        summ = 0
        for m in col:
            if not m == '?':
                summ += float(m)
                count += 1
        return summ / (count + 1)

    # Get dataframe and fill the missing data with the mean of the specific column
    def fill_missing_data(self, df):
        list_column = []
        column_num = -1
        for column in df:
            column_num += 1
            list_column = df[column].tolist()
            mean = self.get_mean_of_column(list_column)
            for i in range(len(list_column)):
                if list_column[i] == '?':
                    list_column[i] = mean
            df.update(pandas.Series(list_column, name=df.columns[column_num]))


# In this class you can find the clustering methods that will be used for this assignment
class DataClustering:
    # file_name = the name of the CSV file that contain the dataset
    # tag_list = a list of the names of the tags (the column name)
    # k = number of clusters
    def __init__(self, file_name, tag_list):
        self.input = DataHandling(file_name, tag_list)
        self.data = self.input.data
        self.tags = self.input.tags
        self.k = 1

    # Set a new k (number of clusters)
    def set_k(self, new_k):
        self.k = new_k

    # Dataset should be 2D
    def plot_data(self):
        principalDf = pandas.DataFrame(data=self.data, columns=['dem1', 'dem2'])
        plt.scatter(principalDf['dem1'], principalDf['dem2'], s=7)
        plt.show()

    # Search for the correct number of clusters by using the 'Elbow Method'
    # Dataset should be 2D
    def elbow_method(self):
        distortions = []
        K = range(1, 18)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(self.data)
            distortions.append(kmeanModel.inertia_)
        # Print plot
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()

    # Run the algorithm with the chosen k
    # Dataset should be 2D
    # Return the MI
    def k_means_clustering(self, show_plt=True):
        kmeans = KMeans(n_clusters=self.k)
        labels = kmeans.fit_predict(self.data)
        # Make plot of the result
        if show_plt:
            plt.title('K - Means')
            plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, s=7, cmap='rainbow')
            centers = kmeans.cluster_centers_
            plt.scatter(centers[:, 0], centers[:, 1], c='black', s=15, alpha=0.3)
            plt.show()
        return metrics.adjusted_mutual_info_score(self.tags, labels)

    # Run the algorithm with the chosen k
    # Dataset should be 2D
    # Return the MI
    def fuzzy_cmeans_clustering(self, show_plt=True):
        fcm = FCM(n_clusters=self.k)
        fcm.fit(self.data)
        labels = fcm.predict(self.data)
        if show_plt:
            plt.title('Fuzzy C Means')
            plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, s=7, cmap='rainbow')
            plt.show()
        return metrics.adjusted_mutual_info_score(self.tags, labels)

    # Search for the correct number of clusters by using the Silhouette Score
    # Dataset should be 2D
    def silhouette_score_test(self, show_plt=True):
        max_score = -1
        best_match = 1
        scores = []
        r = range(2, 15)
        for k in r:
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(self.data)
            score = silhouette_score(self.data, labels)
            scores.append(score)
            if score > max_score:
                max_score = score
                best_match = k
        if show_plt:
            plt.plot(r, scores)
            plt.title("Silhouette")
            plt.xlabel("Number Of Clusters")
            plt.ylabel("Silhouette Score")
            plt.show()
        return best_match

    # Run the algorithm with the chosen k
    # Dataset should be 2D
    # Return the MI
    def gmm_clustering(self, show_plt=True):
        gmm = GaussianMixture(n_components=self.k)
        labels = gmm.fit_predict(self.data)
        if show_plt:
            plt.title('Gaussian Mixture Model')
            plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, s=7, cmap='rainbow')
            plt.show()
        return metrics.adjusted_mutual_info_score(self.tags, labels)

    def hierarchical_clustering(self):
        dendrogram(linkage(self.data, method="ward"))
        plt.title('Dendrogram')
        plt.show()

    # Run the algorithm with the chosen k
    # Dataset should be 2D
    # Return the MI
    def agglomerative_clustering(self, show_plt=True):
        agg = AgglomerativeClustering(n_clusters=self.k)
        labels = agg.fit_predict(self.data)
        if show_plt:
            plt.title('Agglomerative Clustering')
            plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='rainbow', s=7)
            plt.show()
        return metrics.adjusted_mutual_info_score(self.tags, labels)

    # Run the algorithm with the chosen k
    # Dataset should be 2D
    # Return the MI
    def spectral_clustering(self, show_plt=True):
        sc = SpectralClustering(n_clusters=self.k, affinity='nearest_neighbors', random_state=0)
        labels = sc.fit_predict(self.data)
        if show_plt:
            plt.title('Spectral Clustering')
            plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='rainbow', s=7)
            plt.show()
        return metrics.adjusted_mutual_info_score(self.tags, labels)

    # Dataset should be 2D
    # Return the MI
    def dbscan_clustering(self, show_plt=True):
        dbscan = DBSCAN(eps=0.3, min_samples=5).fit(self.data)
        labels = dbscan.fit_predict(self.data)
        if show_plt:
            plt.title('DBSCAN Clustering')
            plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='rainbow', s=7)
            plt.show()
        return metrics.adjusted_mutual_info_score(self.tags, labels)

    # Get the following information (dataset, tags, k) from self and the asked number of samples
    # Return a list contain 5 lists of num_of_samples MI results of clustering_method
    # Also return a dictionary that's maps the index with the clustering method
    def get_mi_matrix(self, num_of_samples):
        matrix_keys = {0: 'K-Means', 1: 'Fuzzy C Means', 2: 'Gaussian Mixture Model',
                       3: 'Agglomerative Clustering', 4: 'Spectral Clustering',
                       5: 'DBSCAN'}
        mi_matrix = [[], [], [], [], [], []]
        for i in range(num_of_samples):
            mi_matrix[0].append(self.k_means_clustering(show_plt=False))
            mi_matrix[1].append(self.fuzzy_cmeans_clustering(show_plt=False))
            mi_matrix[2].append(self.gmm_clustering(show_plt=False))
            mi_matrix[3].append(self.agglomerative_clustering(show_plt=False))
            mi_matrix[4].append(self.spectral_clustering(show_plt=False))
            mi_matrix[5].append(self.dbscan_clustering(show_plt=False))
        return mi_matrix, matrix_keys


# Check if the mean of the first algorithm's mi scores is greater than the second one
# Return the P-value
def t_test(mi_1, mi_2):
    _, p_value = ttest_ind(mi_1, mi_2, equal_var=False)
    mean1 = sum(mi_1) / len(mi_1)
    mean2 = sum(mi_2) / len(mi_2)
    if mean2 > mean1:
        return p_value / 2
    return 1 - p_value / 2


# Turn a list into a string
def list_to_str(li):
    s = "["
    for k in li:
        s += str(k) + ", "
    s += "]"
    return s


# Compare between the different clustering method using t_test (above)
# mi_matrix contain the mi scores of the 5 algorithms
# matrix_keys is a dic that's fit between row in the mi_matrix to clustering method's name
# Return the most accurate clustering method name and a .txt file with the information from the tests
def find_best_algorithm(mi_matrix, matrix_keys, current_dataset):
    # Create a text for documentation of the process
    txt = "-+-+-+-+-+-+-+-+-+-+-+-\n" \
          "Find Best Algorithm - Dataset {} \n" \
          "Significance level = 5% (Recommended)\n" \
          "-+-+-+-+-+-+-+-+-+-+-+-\n\n".format(current_dataset)
    i = 1
    best_algorithm = 0
    while i < len(mi_matrix):
        # Define the significance level as 5%
        txt += "Test #" + str(i) + ": " + matrix_keys[best_algorithm] + " & " + matrix_keys[i] + "\n"
        p_value = t_test(mi_matrix[best_algorithm], mi_matrix[i])
        txt += "\tP-value: " + str(p_value) + "\n"
        if p_value <= 0.05:
            txt += "\tFrom the P-value above we can infer that " + matrix_keys[i] + \
                   " is better than " + matrix_keys[best_algorithm] + "\n\n"
            best_algorithm = i
        else:
            # Else its stay the same
            txt += "\tFrom the P-value above we can infer that " + matrix_keys[best_algorithm] + \
                   " is better than " + matrix_keys[i] + "\n\n"
        i += 1
    txt += "For conclusion, the best algorithm for the current data (according to the T-test) is " \
           + matrix_keys[best_algorithm]
    file = open("Statistical Test For Dataset {}.txt".format(current_dataset), 'w')
    file.write(txt)
    file.close()
    return matrix_keys[best_algorithm]
