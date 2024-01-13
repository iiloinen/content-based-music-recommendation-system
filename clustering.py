from minisom import MiniSom
from sklearn.model_selection import KFold
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler,  MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score, homogeneity_score, adjusted_mutual_info_score
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


tags_df = pd.read_csv('artists_and_their_five_genres.csv')
features_df = pd.read_csv('artists_with_mean_and_features.csv')

concat_data = pd.merge(tags_df, features_df, on="artist")

new_concat_data = concat_data.drop(['id','uri','track_href','analysis_url'], axis=1)
print(new_concat_data.head())
print(new_concat_data.columns)


correlation_matrix = new_concat_data.corr()


def k_means(dataset):
    x_tr = dataset.drop(["artist", "name", "type"], axis=1)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_tr)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(x_train)

    plt.scatter(x_train[:, 0], x_train[:, 1], c=kmeans.labels_, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red')
    plt.xlabel('genre_1st')
    plt.ylabel('weighted_mean')
    plt.title('K-means Clustering')
    plt.show()

    silhouette_avg = silhouette_score(x_train, kmeans.labels_)
    ch_score = calinski_harabasz_score(x_train, kmeans.labels_)
    ad_rand = adjusted_rand_score(dataset.T.columns.tolist(), kmeans.labels_)  
    print("Silhouette Score:", silhouette_avg)
    print("Calinski-Harabasz Score:", ch_score)
    print("Adjusted Rand Score: ", ad_rand)
    hscore = homogeneity_score(dataset.T.columns.tolist(), kmeans.labels_)  
    print("Homogeneity Score: ", hscore)
    mutual_info = adjusted_mutual_info_score(dataset.T.columns.tolist(), kmeans.labels_)  
    print("Adjusted Mutual Information: ", mutual_info)


def k_means_pca(dataset):
    x_tr = dataset.drop(["artist", "name", "type"], axis=1)
    pca = PCA(n_components=2)  # Specify the number of components you want to keep
    principal_components = pca.fit_transform(x_tr)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    print(principal_df)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(x_tr)

    plt.scatter(principal_df['PC1'], principal_df['PC2'], c=kmeans.labels_, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('K-means Clustering with PCA')
    plt.show()

    silhouette_avg = silhouette_score(x_tr, kmeans.labels_)
    ch_score = calinski_harabasz_score(x_tr, kmeans.labels_)
    ad_rand = adjusted_rand_score(dataset.T.columns.tolist(), kmeans.labels_)
    print("Silhouette Score:", silhouette_avg)
    print("Calinski-Harabasz Score:", ch_score)
    print("ad_rand", ad_rand)
    hscore = homogeneity_score(dataset.T.columns.tolist(), kmeans.labels_)
    print("H score; ", hscore)
    mutual_info = adjusted_mutual_info_score(dataset.T.columns.tolist(), kmeans.labels_)
    print("M info; ", mutual_info)


def gmm_clustering(dataset):
    x_tr = dataset.drop(["artist", "name", "type"], axis=1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_tr)
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(x_train)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=gmm.predict(x_train), cmap=sns.cubehelix_palette(as_cmap=True))
    plt.xlabel('genre_1st')
    plt.ylabel('weighted_mean')
    plt.title('Gaussian Mixture Models Clustering')
    plt.show()
    silhouette_avg = silhouette_score(x_train, gmm.predict(x_train))
    ch_score = calinski_harabasz_score(x_train, gmm.predict(x_train))
    print("Silhouette Score:", silhouette_avg)
    print("Calinski-Harabasz Score:", ch_score)
    ad_rand = adjusted_rand_score(dataset.T.columns.tolist(), gmm.predict(x_train))
    print("ad_rand", ad_rand)
    hscore = homogeneity_score(dataset.T.columns.tolist(), gmm.predict(x_train))
    print("H score; ", hscore)
    mutual_info = adjusted_mutual_info_score(dataset.T.columns.tolist(), gmm.predict(x_train))
    print("M info; ", mutual_info)


def gmm_clustering(dataset):
    x_tr = dataset.drop(["artist", "name", "type"], axis=1)


    pca = PCA(n_components=2)
    x_train_pca = pca.fit_transform(x_tr)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_pca)

    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(x_train)

    plt.scatter(x_train[:, 0], x_train[:, 1], c=gmm.predict(x_train), cmap=sns.cubehelix_palette(as_cmap=True))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Gaussian Mixture Models Clustering')
    plt.show()

    silhouette_avg = silhouette_score(x_train, gmm.predict(x_train))
    ch_score = calinski_harabasz_score(x_train, gmm.predict(x_train))
    ad_rand = adjusted_rand_score(dataset.T.columns.tolist(), gmm.predict(x_train))
    hscore = homogeneity_score(dataset.T.columns.tolist(), gmm.predict(x_train))
    mutual_info = adjusted_mutual_info_score(dataset.T.columns.tolist(), gmm.predict(x_train))

    print("Silhouette Score:", silhouette_avg)
    print("Calinski-Harabasz Score:", ch_score)
    print("Adjusted Rand Score:", ad_rand)
    print("Homogeneity Score:", hscore)
    print("Adjusted Mutual Information:", mutual_info)


def hierarchical_clustering_scaling(dataset):
    x_tr = dataset.drop(["artist", "name", "type"], axis=1)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_tr)
    hierarchical = AgglomerativeClustering(n_clusters=16) 
    hierarchical.fit(x_train)

    silhouette_avg = silhouette_score(x_train, hierarchical.labels_)
    ch_score = calinski_harabasz_score(x_train, hierarchical.labels_)
    print("Silhouette Score:", silhouette_avg)
    print("Calinski-Harabasz Score:", ch_score)
    ad_rand = adjusted_rand_score(dataset.T.columns.tolist(), hierarchical.labels_)
    print("ad_rand", ad_rand)
    hscore = homogeneity_score(dataset.T.columns.tolist(), hierarchical.labels_)
    print("H score; ", hscore)
    mutual_info = adjusted_mutual_info_score(dataset.T.columns.tolist(), hierarchical.labels_)
    print("M info; ", mutual_info)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=hierarchical.labels_, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.xlabel('genre_1st')
    plt.ylabel('weighted_mean')
    plt.title('Hierarchical Clustering with Scaling')
    plt.show()
    return hierarchical



def hierarchical_clustering_pca(dataset):
    x_tr = dataset.drop(["artist", "name", "type"], axis=1)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x_tr)
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(principal_components)
    hierarchical = AgglomerativeClustering(n_clusters=16)  # 15
    hierarchical.fit(x_train)
    silhouette_avg = silhouette_score(x_train, hierarchical.labels_)
    ch_score = calinski_harabasz_score(x_train, hierarchical.labels_)
    print("Silhouette Score:", silhouette_avg)
    print("Calinski-Harabasz Score:", ch_score)
    ad_rand = adjusted_rand_score(dataset.T.columns.tolist(), hierarchical.labels_)
    print("ad_rand", ad_rand)
    hscore = homogeneity_score(dataset.T.columns.tolist(), hierarchical.labels_)
    print("H score; ", hscore)
    mutual_info = adjusted_mutual_info_score(dataset.T.columns.tolist(), hierarchical.labels_)
    print("M info; ", mutual_info)
    
    plt.scatter(x_train[:, 0], x_train[:, 1], c=hierarchical.labels_, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Hierarchical Clustering with PCA')
    plt.show()

    return hierarchical



def dbscan_clustering(dataset):
    x_tr = dataset.drop(["artist", "name", "type"], axis=1)

    pca = PCA(n_components=2)
    x_train_pca = pca.fit_transform(x_tr)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_pca)

    dbscan = DBSCAN(eps=0.3, min_samples=5)  # Adjust the parameters as needed
    dbscan.fit(x_train)

    if len(set(dbscan.labels_)) > 1:
        silhouette_avg = silhouette_score(x_train, dbscan.labels_)
        ch_score = calinski_harabasz_score(x_train, dbscan.labels_)
        ad_rand = adjusted_rand_score(dataset.T.columns.tolist(), dbscan.labels_)
        hscore = homogeneity_score(dataset.T.columns.tolist(), dbscan.labels_)
        mutual_info = adjusted_mutual_info_score(dataset.T.columns.tolist(), dbscan.labels_)

        print("Silhouette Score:", silhouette_avg)
        print("Calinski-Harabasz Score:", ch_score)
        print("Adjusted Rand Score:", ad_rand)
        print("Homogeneity Score:", hscore)
        print("Adjusted Mutual Information:", mutual_info)

        plt.scatter(x_train[:, 0], x_train[:, 1], c=dbscan.labels_, cmap=sns.cubehelix_palette(as_cmap=True))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('DBSCAN Clustering')
        plt.show()
    else:
        print("Only one cluster found. Evaluation metrics not applicable.")

    return dbscan


def hierarchical_clustering(dataset):
    x_train = dataset.drop(["artist", "name", "type"], axis=1)

    hierarchical = AgglomerativeClustering(n_clusters=15)
    hierarchical.fit(x_train)
    silhouette_avg = silhouette_score(x_train, hierarchical.labels_)
    ch_score = calinski_harabasz_score(x_train, hierarchical.labels_)
    print("Silhouette Score:", silhouette_avg)
    print("Calinski-Harabasz Score:", ch_score)
    ad_rand = adjusted_rand_score(dataset.T.columns.tolist(), hierarchical.labels_)
    print("ad_rand", ad_rand)
    hscore = homogeneity_score(dataset.T.columns.tolist(), hierarchical.labels_)
    print("H score; ", hscore)
    mutual_info = adjusted_mutual_info_score(dataset.T.columns.tolist(), hierarchical.labels_)
    print("M info; ", mutual_info)

    plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=hierarchical.labels_, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.xlabel('genre')
    plt.ylabel('weighted_mean')
    plt.title('Hierarchical Clustering')
    plt.show()

    return hierarchical







def neural_network(dataset, num_clusters):
    x_train = dataset.drop(["artist", "name", "type"], axis=1)
    
    x_train_target = x_train[["weighted_mean"]]
    x_train.drop(["weighted_mean"],axis = 1 , inplace = True)

    x_train_features, x_test_features, x_train_target, x_test_target = train_test_split(x_train, x_train_target,test_size=.2)
    
    x_train_fetures_normalized = (x_train_features - x_train_features.mean())/x_train_features.std()
    x_test_features_normalized = (x_test_features - x_test_features.mean())/x_test_features.std()

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='softmax'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(num_clusters, activation='softmax')
    ])


    model.compile(optimizer='adam', loss='kld')  # kld Kullback-Leibler Divergence loss is often used for clustering
    model.fit(x_train_fetures_normalized, x_train_target, epochs=1000)  # Use input data as target
    predicted = np.argmax(model.predict(x_test_features_normalized), axis = 1)

    print(x_test_target)
    print(len(predicted))
    
    
    x_test_features["label"] = list(predicted)
    x_test_features["weighted_mean"] = x_test_target["weighted_mean"]
    print(x_test_features.head())
    sns.pairplot(x_test_features[["genre_1st","weighted_mean","label","tempo"]], hue = "genre_1st")
    plt.show()
    sns.pairplot(x_test_features[["genre_1st","weighted_mean","label","tempo"]], hue = "weighted_mean")
    plt.show()

    # sns.pairplot(x_test_features[["genre_1st","weighted_mean","label","tempo"]], hue = "label")
    # plt.show()


    
    print(x_test_features)
    print(predicted)
    print(x_test_target)
    silhouette_avg = silhouette_score(x_test_target, predicted)
    ch_score = calinski_harabasz_score(x_test_target, predicted)
    print("Silhouette Score:", silhouette_avg)
    print("Calinski-Harabasz Score:", ch_score)


# print(new_concat_data.shape)
# nn = neural_network(new_concat_data, 6)

# def song_recommendation(cluster_labels, dataset, genre):

#     genre_data = dataset[dataset['genre_1st'] == genre]
#     genre_cluster = cluster_labels[dataset['genre_1st'] == genre].mode()[0]
#     cluster_songs = genre_data[cluster_labels == genre_cluster]
#     sorted_songs = cluster_songs.sort_values(by='popularity', ascending=False)
#     recommended_song = sorted_songs.iloc[0]

#     return recommended_song

# sr = song_recommendation(hierarchical_result, neural_network, "rock")

# print(new_concat_data.head())


def calculate_similarity(mean1, mean2):
    return abs(mean1 - mean2)

def find_closest_songs(song_name, num_songs, dataset):
    given_song = dataset[dataset['name'] == song_name]
    given_artist = given_song['artist'].iloc[0]
    given_mean = given_song['weighted_mean'].iloc[0]
    given_liveliness = given_song['energy']
    similarities = []

    for index, row in dataset.iterrows():
        if row['artist'] != given_artist:
            song_mean = row['weighted_mean']
            song_liveliness = row['energy']
            similarity = calculate_similarity(given_mean, song_mean) + calculate_similarity(given_liveliness, song_liveliness)
            similarities.append((index, similarity))

    # print([ _[1] for _ in similarities])
    #similarities.sort(key=lambda x: x[1][4])
    closest_songs = []
    recommended_artists = set()

    for index, similarity in similarities:
        if len(closest_songs) >= num_songs:
            break
        if index != given_song.index[0] and dataset.loc[index, 'artist'] not in recommended_artists:
            song_name = dataset.loc[index, 'name']
            artist_name = dataset.loc[index, 'artist']
            song_mean = dataset.loc[index, 'weighted_mean']
            percentage_difference = 100 - abs((given_mean - song_mean ) / given_mean) * 100
            closest_songs.append((song_name, artist_name, percentage_difference))
            recommended_artists.add(artist_name)

    return closest_songs


given_song_name = "Gravity"
closest_songs = find_closest_songs(given_song_name, 3, new_concat_data)

if closest_songs:
    print(f"The closest songs to '{given_song_name}' are:")
    for i, (song, artist, percentage_diff) in enumerate(closest_songs, start=1):
        print(f"{i}. Song: {song}, Artist: {artist}, Percentage simillarity: {percentage_diff:.2f}%")
else:
    print(f"No closest songs found for '{given_song_name}'.")


