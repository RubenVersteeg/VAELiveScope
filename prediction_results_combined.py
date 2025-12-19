import joblib.numpy_pickle as joblib
import numpy as np
import cupy as cp
import torch
import umap
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os
from sklearn.preprocessing import StandardScaler
import random
import seperate_spec_and_features
import features_single
from scipy.stats import spearmanr
import create_padded_spectograms
import normalize_padded_spectograms
from sklearn.preprocessing import StandardScaler
import autoencoder_functions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import matplotlib.ticker as ticker
# Set the global font to Arial
plt.rcParams['font.family'] = 'Arial'
def developmental_stage_prediction_model_weighted_and_hardcoded(m_train, m_test, emitters_train, emitters_test, features, emitters, n_neighbours = 30, n_permutations =100):

    #Weights of pups and adults
    weight_pups = 1/np.sum(emitters_train==0)
    weight_adults = 1/np.sum(emitters_train==1)
    print('Weight WT =',weight_pups)
    print('Weight KO =', weight_adults)
    weights = {0: weight_pups, 1: weight_adults}

    #check if there are as many labels as latent spaces and feature
    assert len(m_train) == len(emitters_train) , "Number of train instances do not match"
    assert len(m_test) == len(emitters_test), "Number of test instances do not match"

    train_latent = m_train.numpy()
    test_latent = m_test.numpy()

    # print('train latent shape:',train_latent.shape)
    # print('test latent shape:',test_latent.shape) 
    # X_train = train_latent
    # X_test = test_latent
    # # #perform normalization on the train data and transform it to the test data
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # train_latent = X_train
    # test_latent = X_test


    #perform umap on train_latent and fit to test latent
    # reducer = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 2)
    # reducer.fit(train_latent)
    # train_latent = reducer.transform(train_latent)
    # test_latent = reducer.transform(test_latent)

    # print('Train latent shape after umap =',np.shape(train_latent))
    # print('Test latent shape after umap =',np.shape(test_latent))
    #get the nearest neighbours of the test set in the train set
    neigh = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
    neigh.fit(train_latent)
    distances, indices = neigh.kneighbors(test_latent)
    print("Indices shape: ", indices.shape)
    print("Distances shape: ", distances.shape)

    #get the labels of the nearest neighbours
    labels_nearest_neighbours = emitters_train[indices]
    print("Labels nearest neighbours shape: ", labels_nearest_neighbours.shape)

    #determine the predicted label based on the nearest neighbours
    predicted_labels = np.zeros(test_latent.shape[0])
    for i in range(test_latent.shape[0]):
        #get the labels of the nearest neighbours
        labels_nearest_neighbours = emitters_train[indices[i]]
        #get the unique labels and their counts
        unique_labels, counts = np.unique(labels_nearest_neighbours, return_counts=True)
        #calculate wieghted counts
        weighted_counts = np.zeros(len(unique_labels))
        for j in range(len(unique_labels)):
            label = unique_labels[j]
            weighted_counts[j] = counts[j] * weights[label]
        #get the index of the label with the most counts
        predicted_labels[i] = unique_labels[np.argmax(weighted_counts)]       

    #determine the accuracy of the total prediction and the accuracy of the prediction for each class
    accuracy = np.sum(predicted_labels == emitters_test) / len(emitters_test)
    accuracy_pups = np.sum(predicted_labels[emitters_test == 0] == emitters_test[emitters_test == 0]) / len(emitters_test[emitters_test == 0])
    accuracy_adults = np.sum(predicted_labels[emitters_test == 1] == emitters_test[emitters_test == 1]) / len(emitters_test[emitters_test == 1])
    print("Overall accuracy: ", accuracy)
    #now permute the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy
    permuted_accuracies = np.zeros(n_permutations)
    permuted_accuracy_pups = np.zeros(n_permutations)
    permuted_accuracy_adults = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled_labels = emitters_train.copy()
        np.random.shuffle(shuffled_labels)
         #determine the predicted label based on the nearest neighbours
        predicted_labels = np.zeros(test_latent.shape[0])
        for j in range(test_latent.shape[0]):
            #get the labels of the nearest neighbours
            labels_nearest_neighbours = shuffled_labels[indices[j]]
            #get the unique labels and their counts
            unique_labels, counts = np.unique(labels_nearest_neighbours, return_counts=True)
            #calculate weighted counts
            weighted_counts = np.zeros(len(unique_labels))
            for k in range(len(unique_labels)):
                label = unique_labels[k]
                weighted_counts[k] = counts[k] * weights[label]
            #get the index of the label with the most counts
            predicted_labels[j] = unique_labels[np.argmax(weighted_counts)]   
        permuted_accuracies[i] = np.sum(predicted_labels == emitters_test) / len(emitters_test)
        permuted_accuracy_pups[i] = np.sum(predicted_labels[emitters_test == 0] == emitters_test[emitters_test == 0]) / len(emitters_test[emitters_test == 0])
        permuted_accuracy_adults[i] = np.sum(predicted_labels[emitters_test == 1] == emitters_test[emitters_test == 1]) / len(emitters_test[emitters_test == 1])
    print(permuted_accuracies)
    mean_permuted_accuracy = np.mean(permuted_accuracies)
    mean_permuted_accuracy_pups = np.mean(permuted_accuracy_pups)
    mean_permuted_accuracy_adults = np.mean(permuted_accuracy_adults)
    std_permuted_accuracy = np.std(permuted_accuracies)
    std_permuted_accuracy_pups = np.std(permuted_accuracy_pups)
    std_permuted_accuracy_adults = np.std(permuted_accuracy_adults)

    #calculate the p-values for every individual emitter and the p-value for the total accuracy
    count_higher = np.sum(permuted_accuracies >= accuracy)
    count_higher_pups = np.sum(permuted_accuracy_pups >= accuracy_pups)
    count_higher_adults = np.sum(permuted_accuracy_adults >= accuracy_adults)
    p_value = count_higher / n_permutations
    p_value_pups = count_higher_pups / n_permutations
    p_value_adults = count_higher_adults / n_permutations

    
    print('Amount of vocalisations',len(emitters))



    print('Number of instances in each class')
    print('pups:' ,np.sum(emitters==0))
    print('adults:',np.sum(emitters==1))

    #seperate the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, emitters, test_size=0.2)
    print('Number of instances in each class in training set')
    print('pups:',np.sum(y_train==0))
    print('adults:',np.sum(y_train==1))
    print('Total:',len(y_train))
    print('Number of features:',len(X_train[0]))
    print('Number of instances in each class in testing set')
    print('pups:',np.sum(y_test==0))
    print('aduls:',np.sum(y_test==1))
    print('Total:',len(y_test))
    print('Number of features:',len(X_test[0]))
    weight_pups = 1/np.sum(y_train==0)
    weight_adults = 1/np.sum(y_train==1)
    print('Weight pups =',weight_pups)
    print('Weight adults =', weight_adults)
    weights2 = {0: weight_pups, 1: weight_adults}

    #perform normalization on the train data and transform it to the test data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_latent2 = X_train
    test_latent2 = X_test
    labels_train2 = y_train
    labels_test2 = y_test

    #perform umap on train_latent and fit to test latent
    reducer = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 2)
    reducer.fit(train_latent2)
    train_latent2 = reducer.transform(train_latent2)
    test_latent2 = reducer.transform(test_latent2)

    print('Train latent shape after umap =',np.shape(train_latent2))
    print('Test latent shape after umap =',np.shape(test_latent2))

       #get the nearest neighbours of the test set in the train set
    neigh = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
    neigh.fit(train_latent2)
    distances2, indices2 = neigh.kneighbors(test_latent2)
    print("Indices shape: ", indices2.shape)
    print("Distances shape: ", distances2.shape)

    #get the labels of the nearest neighbours
    labels_nearest_neighbours2 = labels_train2[indices2]
    print("Labels nearest neighbours shape: ", labels_nearest_neighbours2.shape)

    #determine the predicted label based on the nearest neighbours
    predicted_labels2 = np.zeros(test_latent2.shape[0])
    for i in range(test_latent2.shape[0]):
        #get the labels of the nearest neighbours
        labels_nearest_neighbours2 = labels_train2[indices2[i]]
        #get the unique labels and their counts
        unique_labels2, counts2 = np.unique(labels_nearest_neighbours2, return_counts=True)
        #calculate wieghted counts
        weighted_counts2 = np.zeros(len(unique_labels2))
        for j in range(len(unique_labels2)):
            label2 = unique_labels2[j]
            weighted_counts2[j] = counts2[j] * weights2[label2]
        #get the index of the label with the most counts
        predicted_labels2[i] = unique_labels2[np.argmax(weighted_counts2)]       

    #determine the accuracy of the total prediction and the accuracy of the prediction for each class
    accuracy2 = np.sum(predicted_labels2 == labels_test2) / len(labels_test2)
    accuracy_pups2 = np.sum(predicted_labels2[labels_test2 == 0] == labels_test2[labels_test2 == 0]) / len(labels_test2[labels_test2 == 0])
    accuracy_adults2 = np.sum(predicted_labels2[labels_test2 == 1] == labels_test2[labels_test2 == 1]) / len(labels_test2[labels_test2 == 1])
    print("Overall accuracy: ", accuracy2)
    #now permute the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy
    permuted_accuracies2 = np.zeros(n_permutations)
    permuted_accuracy_pups2 = np.zeros(n_permutations)
    permuted_accuracy_adults2 = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled_labels2 = labels_train2.copy()
        np.random.shuffle(shuffled_labels2)
         #determine the predicted label based on the nearest neighbours
        predicted_labels2 = np.zeros(test_latent2.shape[0])
        for j in range(test_latent2.shape[0]):
            #get the labels of the nearest neighbours
            labels_nearest_neighbours2 = shuffled_labels2[indices2[j]]
            #get the unique labels and their counts
            unique_labels2, counts2 = np.unique(labels_nearest_neighbours2, return_counts=True)
            #calculate weighted counts
            weighted_counts2 = np.zeros(len(unique_labels2))
            for k in range(len(unique_labels2)):
                label2 = unique_labels2[k]
                weighted_counts2[k] = counts2[k] * weights2[label2]
            #get the index of the label with the most counts
            predicted_labels2[j] = unique_labels2[np.argmax(weighted_counts2)]   
        permuted_accuracies2[i] = np.sum(predicted_labels2 == labels_test2) / len(labels_test2)
        permuted_accuracy_pups2[i] = np.sum(predicted_labels2[labels_test2 == 0] == labels_test2[labels_test2 == 0]) / len(labels_test2[labels_test2 == 0])
        permuted_accuracy_adults2[i] = np.sum(predicted_labels2[labels_test2 == 1] == labels_test2[labels_test2 == 1]) / len(labels_test2[labels_test2 == 1])
    print(permuted_accuracies2)
    mean_permuted_accuracy2 = np.mean(permuted_accuracies2)
    mean_permuted_accuracy_pups2 = np.mean(permuted_accuracy_pups2)
    mean_permuted_accuracy_adults2 = np.mean(permuted_accuracy_adults2)
    std_permuted_accuracy2 = np.std(permuted_accuracies2)
    std_permuted_accuracy_pups2 = np.std(permuted_accuracy_pups2)
    std_permuted_accuracy_adults2 = np.std(permuted_accuracy_adults2)

    #count how many times the permuted accuracy is higher than the normal accuracy for p-value calculation
    count_higher2 = np.sum(permuted_accuracies2 >= accuracy2)
    count_higher_pups2 = np.sum(permuted_accuracy_pups2 >= accuracy_pups2)
    count_higher_adults2 = np.sum(permuted_accuracy_adults2 >= accuracy_adults2)
    p_value2 = count_higher2 / n_permutations
    p_value_pups2 = count_higher_pups2 / n_permutations
    p_value_adults2 = count_higher_adults2 / n_permutations


    #now plot the overall accuracy in the first subplot and the accuracy for each class in the second subplot
    plt.figure()
    plt.suptitle('Developmental stage prediction for dataset', fontweight='bold')
    plt.subplot(2, 1, 1)
    labels = ['pups', 'adults', 'Total']
    normal_accuracies = [accuracy_pups, accuracy_adults, accuracy]
    permuted_accuracies = [mean_permuted_accuracy_pups, mean_permuted_accuracy_adults, mean_permuted_accuracy]
    permuted_std_errors = [std_permuted_accuracy_pups, std_permuted_accuracy_adults, std_permuted_accuracy]
    x_labels = ['Accuracy pups', 'Accuracy adults', 'Accuracy Total']
    plt.bar(x_labels, normal_accuracies, yerr=[0, 0,0], label='Accuracy')
    plt.bar(x_labels, permuted_accuracies, yerr=permuted_std_errors, alpha=0.75, label='Permuted Accuracy') ##
    if p_value_pups < 0.01:
        plt.text(0, normal_accuracies[0] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(0, normal_accuracies[0] + 0.05, 'p = {:.2f}'.format(p_value_pups), ha='center')
    if p_value_adults < 0.01:
        plt.text(1, normal_accuracies[1] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(1, normal_accuracies[1] + 0.05, 'p = {:.2f}'.format(p_value_adults), ha='center')
    if p_value < 0.01:
        plt.text(2, normal_accuracies[2] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(2, normal_accuracies[2] + 0.05, 'p = {:.2f}'.format(p_value), ha='center')

    plt.title('Prediction Accuracy based on latent features')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.text(-0.13, 1.0, 'A', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    plt.tight_layout() #adjust plot to fit legend.
    plt.ylim(0, 1)
    plt.xticks([0, 1,2], ['Accuracy pups', 'Accuracy adults', 'Accuracy Total'])
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    plt.grid()
    plt.subplot(2, 1, 2)
    labels = ['WT', 'KO', 'Total']
    normal_accuracies2 = [accuracy_pups2, accuracy_adults2, accuracy2]
    permuted_accuracies2 = [mean_permuted_accuracy_pups2, mean_permuted_accuracy_adults2, mean_permuted_accuracy2]
    permuted_std_errors2 = [std_permuted_accuracy_pups2, std_permuted_accuracy_adults2, std_permuted_accuracy2]
    x_labels2 = ['Accuracy WT', 'Accuracy KO', 'Accuracy Total']
    plt.bar(x_labels2, normal_accuracies2, yerr=[0, 0,0], label='Accuracy')
    plt.bar(x_labels2, permuted_accuracies2, yerr=permuted_std_errors2, alpha=0.75, label='Permuted Accuracy')
    if p_value_pups2 < 0.01:
        plt.text(0, normal_accuracies2[0] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(0, normal_accuracies2[0] + 0.05, 'p = {:.2f}'.format(p_value_pups2), ha='center')
    if p_value_adults2 < 0.01:
        plt.text(1, normal_accuracies2[1] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(1, normal_accuracies2[1] + 0.05, 'p = {:.2f}'.format(p_value_adults2), ha='center')
    if p_value2 < 0.01:
        plt.text(2, normal_accuracies2[2] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(2, normal_accuracies2[2] + 0.05, 'p = {:.2f}'.format(p_value2), ha='center')

    plt.title('Prediction Accuracy based on traditional features')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.ylim(0, 1)
    plt.xticks([0, 1,2], ['Accuracy pups', 'Accuracy adults', 'Accuracy Total'])
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    plt.grid()
    plt.tight_layout()
    plt.text(-0.13, 1.0, 'B', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    #print model used in the bottom left_corner of the figure
    plt.figtext(0.01, 0.01, 'Model_name: shank3_pups_combined_8_v0',  fontsize=8, verticalalignment='bottom')
    plt.show()

def developmental_stage_prediction_model_weighted_and_hardcoded_set(m_train, m_test, emitters_train, emitters_test, features_train, features_test, emitters, n_neighbours = 30, n_permutations =100):

    #Weights of pups and adults
    weight_pups = 1/np.sum(emitters_train==0)
    weight_adults = 1/np.sum(emitters_train==1)
    print('Weight WT =',weight_pups)
    print('Weight KO =', weight_adults)
    weights = {0: weight_pups, 1: weight_adults}

    #check if there are as many labels as latent spaces and feature
    assert len(m_train) == len(emitters_train) , "Number of train instances do not match"
    assert len(m_test) == len(emitters_test), "Number of test instances do not match"

    train_latent = m_train.numpy()
    test_latent = m_test.numpy()

    print('train latent shape:',train_latent.shape)
    print('test latent shape:',test_latent.shape) 
    # X_train = train_latent
    # X_test = test_latent
    # # #perform normalization on the train data and transform it to the test data
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # train_latent = X_train
    # test_latent = X_test


    # #perform umap on train_latent and fit to test latent
    # reducer = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 2)
    # reducer.fit(train_latent)
    # train_latent = reducer.transform(train_latent)
    # test_latent = reducer.transform(test_latent)

    print('Train latent shape after umap =',np.shape(train_latent))
    print('Test latent shape after umap =',np.shape(test_latent))
    #get the nearest neighbours of the test set in the train set
    neigh = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
    neigh.fit(train_latent)
    distances, indices = neigh.kneighbors(test_latent)
    print("Indices shape: ", indices.shape)
    print("Distances shape: ", distances.shape)

    #get the labels of the nearest neighbours
    labels_nearest_neighbours = emitters_train[indices]
    print("Labels nearest neighbours shape: ", labels_nearest_neighbours.shape)

    #determine the predicted label based on the nearest neighbours
    predicted_labels = np.zeros(test_latent.shape[0])
    for i in range(test_latent.shape[0]):
        #get the labels of the nearest neighbours
        labels_nearest_neighbours = emitters_train[indices[i]]
        #get the unique labels and their counts
        unique_labels, counts = np.unique(labels_nearest_neighbours, return_counts=True)
        #calculate wieghted counts
        weighted_counts = np.zeros(len(unique_labels))
        for j in range(len(unique_labels)):
            label = unique_labels[j]
            weighted_counts[j] = counts[j] * weights[label]
        #get the index of the label with the most counts
        predicted_labels[i] = unique_labels[np.argmax(weighted_counts)]       

    #determine the accuracy of the total prediction and the accuracy of the prediction for each class
    accuracy = np.sum(predicted_labels == emitters_test) / len(emitters_test)
    accuracy_pups = np.sum(predicted_labels[emitters_test == 0] == emitters_test[emitters_test == 0]) / len(emitters_test[emitters_test == 0])
    accuracy_adults = np.sum(predicted_labels[emitters_test == 1] == emitters_test[emitters_test == 1]) / len(emitters_test[emitters_test == 1])
    print("Overall accuracy: ", accuracy)
    #now permute the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy
    permuted_accuracies = np.zeros(n_permutations)
    permuted_accuracy_pups = np.zeros(n_permutations)
    permuted_accuracy_adults = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled_labels = emitters_train.copy()
        np.random.shuffle(shuffled_labels)
         #determine the predicted label based on the nearest neighbours
        predicted_labels = np.zeros(test_latent.shape[0])
        for j in range(test_latent.shape[0]):
            #get the labels of the nearest neighbours
            labels_nearest_neighbours = shuffled_labels[indices[j]]
            #get the unique labels and their counts
            unique_labels, counts = np.unique(labels_nearest_neighbours, return_counts=True)
            #calculate weighted counts
            weighted_counts = np.zeros(len(unique_labels))
            for k in range(len(unique_labels)):
                label = unique_labels[k]
                weighted_counts[k] = counts[k] * weights[label]
            #get the index of the label with the most counts
            predicted_labels[j] = unique_labels[np.argmax(weighted_counts)]   
        permuted_accuracies[i] = np.sum(predicted_labels == emitters_test) / len(emitters_test)
        permuted_accuracy_pups[i] = np.sum(predicted_labels[emitters_test == 0] == emitters_test[emitters_test == 0]) / len(emitters_test[emitters_test == 0])
        permuted_accuracy_adults[i] = np.sum(predicted_labels[emitters_test == 1] == emitters_test[emitters_test == 1]) / len(emitters_test[emitters_test == 1])
    print(permuted_accuracies)
    mean_permuted_accuracy = np.mean(permuted_accuracies)
    mean_permuted_accuracy_pups = np.mean(permuted_accuracy_pups)
    mean_permuted_accuracy_adults = np.mean(permuted_accuracy_adults)
    std_permuted_accuracy = np.std(permuted_accuracies)
    std_permuted_accuracy_pups = np.std(permuted_accuracy_pups)
    std_permuted_accuracy_adults = np.std(permuted_accuracy_adults)

    #calculate the p-values for every individual emitter and the p-value for the total accuracy
    count_higher = np.sum(permuted_accuracies >= accuracy)
    count_higher_pups = np.sum(permuted_accuracy_pups >= accuracy_pups)
    count_higher_adults = np.sum(permuted_accuracy_adults >= accuracy_adults)
    p_value = count_higher / n_permutations
    p_value_pups = count_higher_pups / n_permutations
    p_value_adults = count_higher_adults / n_permutations

    
    print('Amount of vocalisations',len(emitters))



    print('Number of instances in each class')
    print('pups:' ,np.sum(emitters==0))
    print('adults:',np.sum(emitters==1))

    #seperate the data into training and testing sets
    X_train = features_train
    X_test = features_test
    y_train = emitters_train
    y_test = emitters_test

    print('Number of instances in each class in training set')
    print('pups:',np.sum(y_train==0))
    print('adults:',np.sum(y_train==1))
    print('Total:',len(y_train))
    print('Number of features:',len(X_train[0]))
    print('Number of instances in each class in testing set')
    print('pups:',np.sum(y_test==0))
    print('aduls:',np.sum(y_test==1))
    print('Total:',len(y_test))
    print('Number of features:',len(X_test[0]))
    weight_pups = 1/np.sum(y_train==0)
    weight_adults = 1/np.sum(y_train==1)
    print('Weight pups =',weight_pups)
    print('Weight adults =', weight_adults)
    weights2 = {0: weight_pups, 1: weight_adults}

    #perform normalization on the train data and transform it to the test data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_latent2 = X_train
    test_latent2 = X_test
    labels_train2 = y_train
    labels_test2 = y_test

    #perform umap on train_latent and fit to test latent
    reducer = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 2)
    reducer.fit(train_latent2)
    train_latent2 = reducer.transform(train_latent2)
    test_latent2 = reducer.transform(test_latent2)

    print('Train latent shape after umap =',np.shape(train_latent2))
    print('Test latent shape after umap =',np.shape(test_latent2))

       #get the nearest neighbours of the test set in the train set
    neigh = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
    neigh.fit(train_latent2)
    distances2, indices2 = neigh.kneighbors(test_latent2)
    print("Indices shape: ", indices2.shape)
    print("Distances shape: ", distances2.shape)

    #get the labels of the nearest neighbours
    labels_nearest_neighbours2 = labels_train2[indices2]
    print("Labels nearest neighbours shape: ", labels_nearest_neighbours2.shape)

    #determine the predicted label based on the nearest neighbours
    predicted_labels2 = np.zeros(test_latent2.shape[0])
    for i in range(test_latent2.shape[0]):
        #get the labels of the nearest neighbours
        labels_nearest_neighbours2 = labels_train2[indices2[i]]
        #get the unique labels and their counts
        unique_labels2, counts2 = np.unique(labels_nearest_neighbours2, return_counts=True)
        #calculate wieghted counts
        weighted_counts2 = np.zeros(len(unique_labels2))
        for j in range(len(unique_labels2)):
            label2 = unique_labels2[j]
            weighted_counts2[j] = counts2[j] * weights2[label2]
        #get the index of the label with the most counts
        predicted_labels2[i] = unique_labels2[np.argmax(weighted_counts2)]       

    #determine the accuracy of the total prediction and the accuracy of the prediction for each class
    accuracy2 = np.sum(predicted_labels2 == labels_test2) / len(labels_test2)
    accuracy_pups2 = np.sum(predicted_labels2[labels_test2 == 0] == labels_test2[labels_test2 == 0]) / len(labels_test2[labels_test2 == 0])
    accuracy_adults2 = np.sum(predicted_labels2[labels_test2 == 1] == labels_test2[labels_test2 == 1]) / len(labels_test2[labels_test2 == 1])
    print("Overall accuracy: ", accuracy2)
    #now permute the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy
    permuted_accuracies2 = np.zeros(n_permutations)
    permuted_accuracy_pups2 = np.zeros(n_permutations)
    permuted_accuracy_adults2 = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled_labels2 = labels_train2.copy()
        np.random.shuffle(shuffled_labels2)
         #determine the predicted label based on the nearest neighbours
        predicted_labels2 = np.zeros(test_latent2.shape[0])
        for j in range(test_latent2.shape[0]):
            #get the labels of the nearest neighbours
            labels_nearest_neighbours2 = shuffled_labels2[indices2[j]]
            #get the unique labels and their counts
            unique_labels2, counts2 = np.unique(labels_nearest_neighbours2, return_counts=True)
            #calculate weighted counts
            weighted_counts2 = np.zeros(len(unique_labels2))
            for k in range(len(unique_labels2)):
                label2 = unique_labels2[k]
                weighted_counts2[k] = counts2[k] * weights2[label2]
            #get the index of the label with the most counts
            predicted_labels2[j] = unique_labels2[np.argmax(weighted_counts2)]   
        permuted_accuracies2[i] = np.sum(predicted_labels2 == labels_test2) / len(labels_test2)
        permuted_accuracy_pups2[i] = np.sum(predicted_labels2[labels_test2 == 0] == labels_test2[labels_test2 == 0]) / len(labels_test2[labels_test2 == 0])
        permuted_accuracy_adults2[i] = np.sum(predicted_labels2[labels_test2 == 1] == labels_test2[labels_test2 == 1]) / len(labels_test2[labels_test2 == 1])
    print(permuted_accuracies2)
    mean_permuted_accuracy2 = np.mean(permuted_accuracies2)
    mean_permuted_accuracy_pups2 = np.mean(permuted_accuracy_pups2)
    mean_permuted_accuracy_adults2 = np.mean(permuted_accuracy_adults2)
    std_permuted_accuracy2 = np.std(permuted_accuracies2)
    std_permuted_accuracy_pups2 = np.std(permuted_accuracy_pups2)
    std_permuted_accuracy_adults2 = np.std(permuted_accuracy_adults2)

    #count how many times the permuted accuracy is higher than the normal accuracy for p-value calculation
    count_higher2 = np.sum(permuted_accuracies2 >= accuracy2)
    count_higher_pups2 = np.sum(permuted_accuracy_pups2 >= accuracy_pups2)
    count_higher_adults2 = np.sum(permuted_accuracy_adults2 >= accuracy_adults2)
    p_value2 = count_higher2 / n_permutations
    p_value_pups2 = count_higher_pups2 / n_permutations
    p_value_adults2 = count_higher_adults2 / n_permutations


    #now plot the overall accuracy in the first subplot and the accuracy for each class in the second subplot
    plt.figure()
    plt.suptitle('Developmental stage prediction for dataset', fontweight='bold')
    plt.subplot(2, 1, 1)
    labels = ['pups', 'adults', 'Total']
    normal_accuracies = [accuracy_pups, accuracy_adults, accuracy]
    permuted_accuracies = [mean_permuted_accuracy_pups, mean_permuted_accuracy_adults, mean_permuted_accuracy]
    permuted_std_errors = [std_permuted_accuracy_pups, std_permuted_accuracy_adults, std_permuted_accuracy]
    x_labels = ['Accuracy pups', 'Accuracy adults', 'Accuracy Total']
    plt.bar(x_labels, normal_accuracies, yerr=[0, 0,0], label='Accuracy')
    plt.bar(x_labels, permuted_accuracies, yerr=permuted_std_errors, alpha=0.75, label='Permuted Accuracy') ##
    if p_value_pups < 0.01:
        plt.text(0, normal_accuracies[0] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(0, normal_accuracies[0] + 0.05, 'p = {:.2f}'.format(p_value_pups), ha='center')
    if p_value_adults < 0.01:
        plt.text(1, normal_accuracies[1] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(1, normal_accuracies[1] + 0.05, 'p = {:.2f}'.format(p_value_adults), ha='center')
    if p_value < 0.01:
        plt.text(2, normal_accuracies[2] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(2, normal_accuracies[2] + 0.05, 'p = {:.2f}'.format(p_value), ha='center')

    plt.title('Prediction Accuracy based on latent features')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.text(-0.1, 1.0, 'A', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    plt.tight_layout() #adjust plot to fit legend.
    plt.ylim(0, 1)
    plt.xticks([0, 1,2], ['Accuracy pups', 'Accuracy adults', 'Accuracy Total'])
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    plt.grid()
    plt.subplot(2, 1, 2)
    labels = ['WT', 'KO', 'Total']
    normal_accuracies2 = [accuracy_pups2, accuracy_adults2, accuracy2]
    permuted_accuracies2 = [mean_permuted_accuracy_pups2, mean_permuted_accuracy_adults2, mean_permuted_accuracy2]
    permuted_std_errors2 = [std_permuted_accuracy_pups2, std_permuted_accuracy_adults2, std_permuted_accuracy2]
    x_labels2 = ['Accuracy WT', 'Accuracy KO', 'Accuracy Total']
    plt.bar(x_labels2, normal_accuracies2, yerr=[0, 0,0], label='Accuracy')
    plt.bar(x_labels2, permuted_accuracies2, yerr=permuted_std_errors2, alpha=0.75, label='Permuted Accuracy')
    if p_value_pups2 < 0.01:
        plt.text(0, normal_accuracies2[0] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(0, normal_accuracies2[0] + 0.05, 'p = {:.2f}'.format(p_value_pups2), ha='center')
    if p_value_adults2 < 0.01:
        plt.text(1, normal_accuracies2[1] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(1, normal_accuracies2[1] + 0.05, 'p = {:.2f}'.format(p_value_adults2), ha='center')
    if p_value2 < 0.01:
        plt.text(2, normal_accuracies2[2] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(2, normal_accuracies2[2] + 0.05, 'p = {:.2f}'.format(p_value2), ha='center')

    plt.title('Prediction Accuracy based on traditional features')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.ylim(0, 1)
    plt.xticks([0, 1,2], ['Accuracy pups', 'Accuracy adults', 'Accuracy Total'])
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    plt.grid()
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.text(-0.1, 1.0, 'B', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    #print model used in the bottom left_corner of the figure
    plt.figtext(0.01, 0.01, 'Model_name: shank3_pups_combined_8_v0',  fontsize=8, verticalalignment='bottom')
    plt.show()



def get_explained_variance_features_across_2_models(m_train, m_test, emitters_train, emitters_test, features_train, features_test, n_neighbours = 30, n_permutations =100, remove_double_specs = False):

    hardcoded_feature1 = 'duration'
    hardcoded_feature2 = 'mean_freq'
    hardcoded_feature3 = 'min_freq'
    hardcoded_feature4 = 'max_freq'
    hardcoded_feature5 = 'bandwidth'
    hardcoded_feature6 = 'starting_freq'
    hardcoded_feature7 = 'stopping_freq'
    hardcoded_feature8 = 'directionality'
    hardcoded_feature9 = 'coefficient_of_variation'
    hardcoded_feature10 = 'normalized_irregularity'
    hardcoded_feature11 = 'local_variability'
    hardcoded_feature12 = 'nr_of_steps_up'
    hardcoded_feature13 = 'nr_of_steps_down'
    hardcoded_feature14 = 'nr_of_peaks'
    hardcoded_feature15 = 'nr_of_valleys'

    #check if there are as many features as spectograms in the train and test sets
    assert len(m_train) == len(features_train) == len(emitters_train), "Number of train instances do not match"
    assert len(m_test) == len(features_test) == len(emitters_test), "Number of test instances do not match"

    train_latent1 = m_train.numpy()
    test_latent1 = m_test.numpy()
    features_train1 = features_train
    features_test1 = features_test

    print('train latent shape:',train_latent1.shape)
    print('test latent shape:',test_latent1.shape) 
  
    #get the nearest neighbours of the test set in the train set
    neigh1 = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
    neigh1.fit(train_latent1)
    distances1, indices1 = neigh1.kneighbors(test_latent1)
    print("Indices shape: ", indices1.shape)
    print("Distances shape: ", distances1.shape)

    #get the features of the nearest neighbours
    features_nearest_neighbours1 = features_train1[indices1]
    print("Features nearest neighbours shape: ", features_nearest_neighbours1.shape)

    #determine the predicted features based on the nearest neighbours
    predicted_features1 = np.zeros((test_latent1.shape[0], features_train1.shape[1]))
    for i in range(test_latent1.shape[0]):
        #get the features of the nearest neighbours
        features_nearest_neighbours1 = features_train1[indices1[i]]
        #calculate the mean of the features of the nearest neighbours
        predicted_features1[i] = np.mean(features_nearest_neighbours1, axis=0)

    #assert if predicted features has the same shape as features test
    assert predicted_features1.shape == np.array(features_test1).shape, "Number of predicted features and test features do not match"

    #determine the rmse of the prediction for each feature and the total rmse
    rmse_per_feature1 = np.sqrt(np.mean((predicted_features1 - features_test1)**2, axis=0))
    total_rmse1 = np.sqrt(np.mean((predicted_features1 - features_test1)**2))

    print("Total RMSE: ", total_rmse1)
    print("RMSE per feature: ", rmse_per_feature1)

    #now permut the test features n times and calculate the mean rmse and the standard deviation of the rmse for the
    #overall permuted rmse and the permuted rmse for each feature
    permuted_rmses1 = np.zeros(n_permutations)
    permuted_rmses_per_feature1 = np.zeros((n_permutations, features_train1.shape[1]))
    for i in range(n_permutations):
        shuffled_features = features_train1.copy()
        np.random.shuffle(shuffled_features)

        permuted_predicted_features1 = np.zeros((test_latent1.shape[0], features_train1.shape[1]))
        for j in range(test_latent1.shape[0]):
            #get the features of the nearest neighbours
            features_nearest_neighbours1 = shuffled_features[indices1[j]]
            #calculate the mean of the features of the nearest neighbours
            permuted_predicted_features1[j] = np.mean(features_nearest_neighbours1, axis=0)
        
        permuted_rmses1[i] = np.sqrt(np.mean((permuted_predicted_features1 - features_test1)**2))
        permuted_rmses_per_feature1[i] = np.sqrt(np.mean((permuted_predicted_features1 - features_test1)**2, axis=0))

    variance_explained_per_permutation = 1 - (rmse_per_feature1 / permuted_rmses_per_feature1)
    mean_variance_explained_per_feature1 = np.mean(variance_explained_per_permutation, axis=0)
    std_variance_explained_per_feature1 = np.std(variance_explained_per_permutation, axis=0)

    #determine the variance explained by the model for each feature and total variance explained (variance explained = 1 - (rmse / mean_permuted_rmse))
    variance_explained_per_feature1 = mean_variance_explained_per_feature1

    #now do it individually for the pups and adults with pups = 0 and adults = 1
    pups_indices_train = np.where(emitters_train == 0)[0]
    adults_indices_train = np.where(emitters_train == 1)[0]
    pups_indices_test = np.where(emitters_test == 0)[0]
    adults_indices_test = np.where(emitters_test == 1)[0]
    #seperate the data into pups and adults
    train_latent_pups1 = train_latent1[pups_indices_train]
    test_latent_pups1 = test_latent1[pups_indices_test]
    features_train_pups1 = features_train1[pups_indices_train]
    features_test_pups1 = features_test1[pups_indices_test]
    train_latent_adults1 = train_latent1[adults_indices_train]
    test_latent_adults1 = test_latent1[adults_indices_test]
    features_train_adults1 = features_train1[adults_indices_train]
    features_test_adults1 = features_test1[adults_indices_test]
    print('Number of pups in train set:', len(train_latent_pups1))
    print('Number of adults in train set:', len(train_latent_adults1))
    print('Number of pups in test set:', len(test_latent_pups1))
    print('Number of adults in test set:', len(test_latent_adults1))


    #get the nearest neighbours of the test set in the train set
    neigh_pups1 = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
    neigh_pups1.fit(train_latent_pups1)
    distances_pups1, indices_pups1 = neigh_pups1.kneighbors(test_latent_pups1)
    print("Indices shape: ", indices_pups1.shape)
    print("Distances shape: ", distances_pups1.shape)

    #get the features of the nearest neighbours
    features_nearest_neighbours_pups1 = features_train_pups1[indices_pups1]
    print("Features nearest neighbours shape: ", features_nearest_neighbours_pups1.shape)

    #determine the predicted features based on the nearest neighbours
    predicted_features_pups1 = np.zeros((test_latent_pups1.shape[0], features_train_pups1.shape[1]))
    for i in range(test_latent_pups1.shape[0]):
        #get the features of the nearest neighbours
        features_nearest_neighbours_pups1 = features_train_pups1[indices_pups1[i]]
        #calculate the mean of the features of the nearest neighbours
        predicted_features_pups1[i] = np.mean(features_nearest_neighbours_pups1, axis=0)

    #assert if predicted features has the same shape as features test
    assert predicted_features_pups1.shape == np.array(features_test_pups1).shape, "Number of predicted features and test features do not match"

    #determine the rmse of the prediction for each feature and the total rmse
    rmse_per_feature_pups1 = np.sqrt(np.mean((predicted_features_pups1 - features_test_pups1)**2, axis=0))
    total_rmse_pups1 = np.sqrt(np.mean((predicted_features_pups1 - features_test_pups1)**2))

    print("Total RMSE: ", total_rmse_pups1)
    print("RMSE per feature: ", rmse_per_feature_pups1)

    #now permut the test features n times and calculate the mean rmse and the standard deviation of the rmse for the
    #overall permuted rmse and the permuted rmse for each feature
    permuted_rmses_pups1 = np.zeros(n_permutations)
    permuted_rmses_per_feature_pups1 = np.zeros((n_permutations, features_train_pups1.shape[1]))
    for i in range(n_permutations):
        shuffled_features = features_train_pups1.copy()
        np.random.shuffle(shuffled_features)

        permuted_predicted_features_pups1 = np.zeros((test_latent_pups1.shape[0], features_train_pups1.shape[1]))
        for j in range(test_latent_pups1.shape[0]):
            #get the features of the nearest neighbours
            features_nearest_neighbours_pups1 = shuffled_features[indices_pups1[j]]
            #calculate the mean of the features of the nearest neighbours
            permuted_predicted_features_pups1[j] = np.mean(features_nearest_neighbours_pups1, axis=0)
        
        permuted_rmses_pups1[i] = np.sqrt(np.mean((permuted_predicted_features_pups1 - features_test_pups1)**2))
        permuted_rmses_per_feature_pups1[i] = np.sqrt(np.mean((permuted_predicted_features_pups1 - features_test_pups1)**2, axis=0))

    variance_explained_per_permutation_pups = 1 - (rmse_per_feature_pups1 / permuted_rmses_per_feature_pups1)
    mean_variance_explained_per_feature_pups1 = np.mean(variance_explained_per_permutation_pups, axis=0)
    std_variance_explained_per_feature_pups1 = np.std(variance_explained_per_permutation_pups, axis=0)

    #determine the variance explained by the model for each feature and total variance explained (variance explained = 1 - (rmse / mean_permuted_rmse))
    variance_explained_per_feature_pups1 = mean_variance_explained_per_feature_pups1

    #get the nearest neighbours of the test set in the train set
    neigh_adults1 = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
    neigh_adults1.fit(train_latent_adults1)
    distances_adults1, indices_adults1 = neigh_adults1.kneighbors(test_latent_adults1)
    print("Indices shape: ", indices_adults1.shape)
    print("Distances shape: ", distances_adults1.shape)

    #get the features of the nearest neighbours
    features_nearest_neighbours_adults1 = features_train_adults1[indices_adults1]
    print("Features nearest neighbours shape: ", features_nearest_neighbours_adults1.shape)

    #determine the predicted features based on the nearest neighbours
    predicted_features_adults1 = np.zeros((test_latent_adults1.shape[0], features_train_adults1.shape[1]))
    for i in range(test_latent_adults1.shape[0]):
        #get the features of the nearest neighbours
        features_nearest_neighbours_adults1 = features_train_adults1[indices_adults1[i]]
        #calculate the mean of the features of the nearest neighbours
        predicted_features_adults1[i] = np.mean(features_nearest_neighbours_adults1, axis=0)

    #assert if predicted features has the same shape as features test
    assert predicted_features_adults1.shape == np.array(features_test_adults1).shape, "Number of predicted features and test features do not match"

    #determine the rmse of the prediction for each feature and the total rmse
    rmse_per_feature_adults1 = np.sqrt(np.mean((predicted_features_adults1 - features_test_adults1)**2, axis=0))
    total_rmse_adults1 = np.sqrt(np.mean((predicted_features_adults1 - features_test_adults1)**2))

    print("Total RMSE: ", total_rmse_adults1)
    print("RMSE per feature: ", rmse_per_feature_adults1)

    #now permut the test features n times and calculate the mean rmse and the standard deviation of the rmse for the
    #overall permuted rmse and the permuted rmse for each feature
    permuted_rmses_adults1 = np.zeros(n_permutations)
    permuted_rmses_per_feature_adults1 = np.zeros((n_permutations, features_train_adults1.shape[1]))
    for i in range(n_permutations):
        shuffled_features = features_train_adults1.copy()
        np.random.shuffle(shuffled_features)

        permuted_predicted_features_adults1 = np.zeros((test_latent_adults1.shape[0], features_train_adults1.shape[1]))
        for j in range(test_latent_adults1.shape[0]):
            #get the features of the nearest neighbours
            features_nearest_neighbours_adults1 = shuffled_features[indices_adults1[j]]
            #calculate the mean of the features of the nearest neighbours
            permuted_predicted_features_adults1[j] = np.mean(features_nearest_neighbours_adults1, axis=0)
        
        permuted_rmses_adults1[i] = np.sqrt(np.mean((permuted_predicted_features_adults1 - features_test_adults1)**2))
        permuted_rmses_per_feature_adults1[i] = np.sqrt(np.mean((permuted_predicted_features_adults1 - features_test_adults1)**2, axis=0))

    variance_explained_per_permutation_adults = 1 - (rmse_per_feature_adults1 / permuted_rmses_per_feature_adults1)
    mean_variance_explained_per_feature_adults1 = np.mean(variance_explained_per_permutation_adults, axis=0)
    std_variance_explained_per_feature_adults1 = np.std(variance_explained_per_permutation_adults, axis=0)

    #determine the variance explained by the model for each feature and total variance explained (variance explained = 1 - (rmse / mean_permuted_rmse))
    variance_explained_per_feature_adults1 = mean_variance_explained_per_feature_adults1


    #plot the variance explained per feature with error bars for the std of the permuted rmse per feature
    plt.figure(figsize=(10,5))
    bar_width = 0.25
    x = np.arange(len(variance_explained_per_feature1))
    #show total variance explained as a horizontal dotted line
    # plt.axhline(y=total_variance_explained2*100, color='r', linestyle='--', label='Total Variance Explained: {:.2f}%'.format(total_variance_explained2*100))
    plt.bar(x - bar_width, variance_explained_per_feature1*100, width = bar_width, yerr=std_variance_explained_per_feature1*100, alpha=1.0, label='Variance Explained Full Dataset')
    plt.bar(x, variance_explained_per_feature_pups1*100, width = bar_width, yerr=std_variance_explained_per_feature_pups1*100, alpha=1.0, label='Variance Explained Pups Dataset')
    plt.bar(x + bar_width, variance_explained_per_feature_adults1*100, width = bar_width, yerr=std_variance_explained_per_feature_adults1*100, alpha=1.0, label='Variance Explained Adults Dataset')
    plt.xticks(x, labels=[hardcoded_feature1, hardcoded_feature2, hardcoded_feature3, hardcoded_feature4, hardcoded_feature5, hardcoded_feature6, hardcoded_feature7, hardcoded_feature8, hardcoded_feature9, hardcoded_feature10, hardcoded_feature11, hardcoded_feature12, hardcoded_feature13, hardcoded_feature14, hardcoded_feature15], rotation=45, ha ='right')
    plt.title('Degree to which the spatial structure in the autoencoder space is explained by a given feature',fontweight='bold')
    plt.xlabel('Features')
    plt.ylabel('Variance Explained (%)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # plt.text(-2.0,-30.1 ,'Model_name: {}'.format(path_to_model))
    # plt.text(-2.0,-35.1 ,'Number of neighbours: {}'.format(n_neighbours))
    plt.ylim(0, 100)
    #show which model was used in the bottom left corner
    plt.figtext(0.01, 0.05 ,'Combined Model: shank3_pups_combined_8_v0', fontsize=8)
    plt.show()