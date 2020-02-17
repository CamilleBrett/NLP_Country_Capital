# -*- coding: utf-8 -*-

import os
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy

# Clean dictionary


def cleanDictionary(dictionary, embedded_space):
    dictionary_copy = deepcopy(dictionary)
    for word in dictionary:
        if word not in embedded_space or dictionary[word] not in embedded_space:
            del dictionary_copy[word]
    print(len(dictionary), len(dictionary_copy))
    return dictionary_copy


# Clean list of words
def cleanWords(list_of_words, embedded_space):
    clean_list = [word for word in list_of_words if word in embedded_space]
    return clean_list



# Create embedded space dictionary from vectors .txt file
def toEmbeddedSpace(path, file_name): 
    embedded_space = dict()
    tokens = dict()
    with open(os.path.join(path, file_name), 'r', encoding='utf8') as file:
        for index, line in enumerate(file):
            items = line.split()
            word = items[0]
            vector = np.asarray(items[1:], dtype='float32')
            embedded_space[word] = vector
            tokens[word] = index
    print('Done loading vectors !')
    return embedded_space, tokens


def computeWordSimilarity(word1, word2, embedded_space):
    """
    Computes cosine similarity between 2 words (between 0 and 1). 
    The closer the result isto 1, the more similar the vectors.
    https://en.wikipedia.org/wiki/Cosine_similarity
    """
    vect1 = embedded_space[word1]
    vect2 = embedded_space[word2]
    return (np.dot(vect1, vect2)/(np.linalg.norm(vect1)*np.linalg.norm(vect2)))


def computeVectorSimilarity(vector1, vector2):
    """
    Computes cosine similarity between 2 vectors (between 0 and 1). 
    The closer the result is to 1, the more similar the vectors.
    https://en.wikipedia.org/wiki/Cosine_similarity
    """
    return (np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)))


def printSimilarity(word1, word2, embedded_space):
    print("The cosine similarity between " + word1 + " and " + word2 +
          " is: ", computeWordSimilarity(word1, word2, embedded_space))


country_capital_vocab = "France Paris Belgium Brussels Canada Toronto Russia Moscow China Beijing Japan Tokyo Spain Madrid Germany Berlin Netherlands Amsterdam Italy Roma Nicaragua Managua"



def createEmbeddedMatrix(words, embedded_space):
    mat = []
    for word in words:
        mat.append(embedded_space[word])
    return mat


def showElbowCriteria(words, embedded_space, n_dimensions=None):
    '''
    Plots the explained variance ratio for the n_dimensions first dimensions of the PCA.
    Elbow criteria permits us to deduce how many dimensions are relevant for PCA.
    '''

    if n_dimensions == None:
        n_dimensions = min(200, len(words))
    # Create matrix : we assume the list of words is clean
    mat = createEmbeddedMatrix(words, embedded_space)
    transformer = PCA(n_components=n_dimensions)
    transformer.fit(mat)
    explained_variance = transformer.explained_variance_ratio_
    print(explained_variance)
    # Plot explained variance
    x = np.arange(1, n_dimensions + 1)
    plt.figure()
    plt.plot(x, explained_variance, '*')
    plt.title("Explained variance for each dimension of the ACP")
    plt.grid()
    plt.show()


def toPCA(words, embedded_space, dimensions):
    # Create matrix
    mat = createEmbeddedMatrix(words, embedded_space)
    transformer = PCA(n_components=dimensions)
    coordinates = transformer.fit(mat)
    explained_variance = transformer.explained_variance_ratio_
    print(explained_variance)

    coordinates = transformer.transform(mat)
    words_coordinates = dict(zip(words, coordinates))
    return words_coordinates

# TODO: IPCA function


def plotWords(words_to_plot, words_coordinates):
    words_to_plot = [word for word in words_to_plot if (
        word in words_coordinates)]
    xs = []
    ys = []

    for word in words_to_plot:
        x, y = words_coordinates[word][:2]
        xs.append(x)
        ys.append(y)

    plt.figure(figsize=(12, 8))
    plt.scatter(xs, ys, marker='o')

    for i in range(len(words_to_plot)):
        plt.text(xs[i], ys[i], words_to_plot[i])

    plt.show()


if __name__ == "__main__":

    glove_path = './data/glove' # Where your vectors are stored
    glove_file_name = 'glove.6B.200d.txt'

    embedded_space, tokens = toEmbeddedSpace(glove_path, glove_file_name)

    country_capital_data = pd.read_csv("./data/country-list.csv") # a .csv file with all capitals associated with their country
    countries = [word.lower()
                 for word in country_capital_data.country.values.tolist()]
    capitals = [word.lower()
                for word in country_capital_data.capital.values.tolist()]

    country_capital_mapping = cleanDictionary(
        dict(zip(countries, capitals)), embedded_space)

    country_capital_list = []
    for country in country_capital_mapping:
        country_capital_list.append(country)
        country_capital_list.append(country_capital_mapping[country])


    geo_coordinates = toPCA(country_capital_list, embedded_space, 7)

    geo_to_plot = ["france", "paris", "belgium", "brussels", "canada", "toronto", "russia", "moscow", "china",
                   "beijing", "japan", "tokyo", "spain", "madrid", "germany", "berlin", "netherlands", "amsterdam", "italy", "roma"]
    
    plotWords(geo_to_plot, geo_coordinates) # tries to show a spatial relation

    if os.path.exists('./results/country_capital_vector_similarities.csv'):
        print("CSV file already exists")
    else:
        number_of_countries = len(country_capital_mapping)
        similarity_matrix = np.zeros(
            (number_of_countries, number_of_countries), dtype='float32')
        for i, country_i in enumerate(country_capital_mapping):
            for j, country_j in enumerate(country_capital_mapping):
                if j > i:
                    continue
                elif j == i:
                    similarity_matrix[i][j] = 0.5
                else:
                    vector_i = geo_coordinates[country_i] - \
                        geo_coordinates[country_capital_mapping[country_i]]
                    vector_j = geo_coordinates[country_j] - \
                        geo_coordinates[country_capital_mapping[country_j]]
                    similarity_matrix[i][j] = computeVectorSimilarity(
                        vector_i, vector_j)
        similarity_matrix = similarity_matrix + similarity_matrix.transpose()
        index = []
        for country in country_capital_mapping:
            index.append(country + '-' + country_capital_mapping[country])

        # print(similarity_matrix)
        similarities_df = pd.DataFrame(
            similarity_matrix, index=index, columns=index)
        print(similarities_df.head())
        similarities_df.to_csv(
            './results/country_capital_vector_similarities.csv')

    similarities_df = pd.read_csv(
        './results/country_capital_vector_similarities.csv')
    france_similarities = similarities_df.iloc[48].to_numpy
    print(france_similarities)
    plt.figure()
    similarities_df.plot()
    plt.show()

    # Box plot
    plt.figure()
    color = {'boxes': 'DarkGreen', 'whiskers': 'DarkOrange',
            'medians': 'DarkBlue', 'caps': 'Gray'}

    similarities_df.plot.box(color=color, sym='r+', rot=90, figsize=(20,30))
    plt.show()
