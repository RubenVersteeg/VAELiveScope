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

def get_explained_variance_features_across_2_models(spec1,spec2, path_to_model1, path_to_model2, features1, features2, n_neighbours = 30, n_permutations =100, remove_double_specs = False):

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
    

    features1 = np.array(features1)
    print('features shape =', features1.shape)
    #load the trained parameters used in training
    parameters1 = joblib.load(path_to_model1 + '/parameters.pkl')

    features2 = np.array(features2)
    print('features shape =', features2.shape)
    #load the trained parameters used in training
    parameters2 = joblib.load(path_to_model2 + '/parameters.pkl')

    ##load spec_indices for train and test set
    spec_indices_train1 = torch.load(path_to_model1 + '/spec_indices_train')
    spec_indices_test1 = torch.load(path_to_model1 + '/spec_indices_test')
    spec_indices_train2 = torch.load(path_to_model2 + '/spec_indices_train')
    spec_indices_test2 = torch.load(path_to_model2 + '/spec_indices_test')

    if remove_double_specs:
        print('indices1 before removal of doubles =', len(spec_indices_train1))
        print('indices2 before removal of doubles =', len(spec_indices_train2))
        spec_indices_train1_after_removal = []
        spec_indices_train2_after_removal = []
        for i in range(len(spec_indices_train1)):
            if spec_indices_train1[i] not in spec_indices_train1_after_removal:
                spec_indices_train1_after_removal.append(spec_indices_train1[i])
        spec_indices_train1 = spec_indices_train1_after_removal
        for i in range(len(spec_indices_train2)):
            if spec_indices_train2[i] not in spec_indices_train2_after_removal:
                spec_indices_train2_after_removal.append(spec_indices_train2[i])
        spec_indices_train2 = spec_indices_train2_after_removal
        print('indices1 after removal of doubles =', len(spec_indices_train1_after_removal))
        print('indices2 after removal of doubles =', len(spec_indices_train2_after_removal))

    

    #get the spectograms
    train_specs1 = []
    test_specs1 = []

    for i in spec_indices_train1:
        train_specs1.append(spec1[i])
    for i in spec_indices_test1:
        test_specs1.append(spec1[i])

    train_specs2 = []
    test_specs2 = []
    for i in spec_indices_train2:
        train_specs2.append(spec2[i])
    for i in spec_indices_test2:
        test_specs2.append(spec2[i])

    #get the features for the train and test set
    features_train1 = []
    features_test1 = []
    for i in spec_indices_train1:
        features_train1.append(features1[i])
    for i in spec_indices_test1:
        features_test1.append(features1[i])

    features_train2 = []
    features_test2 = []
    for i in spec_indices_train2:
        features_train2.append(features2[i])
    for i in spec_indices_test2:
        features_test2.append(features2[i])

    features_train1 = np.array(features_train1)
    features_test1 = np.array(features_test1)
    features_train2 = np.array(features_train2)
    features_test2 = np.array(features_test2)

    #check if there are as many features as spectograms in the train and test sets
    assert len(train_specs1) == len(features_train1), "Number of train spectograms and labels do not match"
    assert len(test_specs1) == len(features_test1), "Number of test spectograms and labels do not match"
    assert len(train_specs2) == len(features_train2), "Number of train spectograms and labels do not match"
    assert len(test_specs2) == len(features_test2), "Number of test spectograms and labels do not match"

    #pad the spectograms according to our training procedure
    padding_length = 160
    train_padded_specs1 = create_padded_spectograms.pad_spectograms_sorted(train_specs1, padding_length)
    test_padded_specs1 = create_padded_spectograms.pad_spectograms_sorted(test_specs1, padding_length)
    train_padded_specs2 = create_padded_spectograms.pad_spectograms_sorted(train_specs2, padding_length)
    test_padded_specs2 = create_padded_spectograms.pad_spectograms_sorted(test_specs2, padding_length)

    #delete unnecesarry big files in RAM
    del spec1,spec2, train_specs1, test_specs1, train_specs2, test_specs2

    #normalize according to our training
    if parameters1['max_value_per_spec'] == 'true':
        max_value_per_spec = True
        print(max_value_per_spec)
    train_padded_specs1 = normalize_padded_spectograms.normalize_specs(train_padded_specs1, max_value_per_spec)
    test_padded_specs1 = normalize_padded_spectograms.normalize_specs(test_padded_specs1, max_value_per_spec)
    if parameters2['max_value_per_spec'] == 'true':
        max_value_per_spec = True
        print(max_value_per_spec)
    train_padded_specs2 = normalize_padded_spectograms.normalize_specs(train_padded_specs2, max_value_per_spec)
    test_padded_specs2 = normalize_padded_spectograms.normalize_specs(test_padded_specs2, max_value_per_spec)

    #convert to torch tensors
    latent_train_loader1 = torch.utils.data.DataLoader(train_padded_specs1, batch_size=1, shuffle=False)
    latent_test_loader1 = torch.utils.data.DataLoader(test_padded_specs1, batch_size=1, shuffle=False)
    latent_train_loader2 = torch.utils.data.DataLoader(train_padded_specs2, batch_size=1, shuffle=False)
    latent_test_loader2 = torch.utils.data.DataLoader(test_padded_specs2, batch_size=1, shuffle=False)

    #delete the padded spectograms from memory
    del train_padded_specs1, test_padded_specs1, train_padded_specs2, test_padded_specs2

    #initialize model
    latent_space_size1 = parameters1['latent_space_size']
    slope_leaky1 = parameters1['slope_leaky']
    learning_rate1 = parameters1['learning_rate']
    precision_model1 = parameters1['precision_model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = autoencoder_functions.VAE_KL_leaky(z_dim = latent_space_size1, device_name='cuda',slope=slope_leaky1,lr = learning_rate1, model_precision =precision_model1)
    model1 = model1.to(device)

    #load the model
    model_file_name1 = path_to_model1 + '/model.pt'
    model1.load_state_dict(torch.load(model_file_name1))

    train_latent1 = model1.get_latent(latent_train_loader1)
    test_latent1 = model1.get_latent(latent_test_loader1)

    # X_train1 = train_latent1
    # X_test1 = test_latent1
    # # #perform normalization on the train data and transform it to the test data
    # scaler = StandardScaler()
    # X_train1 = scaler.fit_transform(X_train1)
    # X_test1= scaler.transform(X_test1)

    # train_latent1 = X_train1
    # test_latent1 = X_test1


    # # #perform umap on train_latent and fit to test latent
    # reducer = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 2)
    # reducer.fit(train_latent1)
    # train_latent1 = reducer.transform(train_latent1)
    # test_latent1 = reducer.transform(test_latent1)

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
    predicted_features1 = np.zeros((test_latent1.shape[0], features1.shape[1]))
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
    permuted_rmses_per_feature1 = np.zeros((n_permutations, features1.shape[1]))
    for i in range(n_permutations):
        np.random.shuffle(features_train1)

        permuted_predicted_features1 = np.zeros((test_latent1.shape[0], features1.shape[1]))
        for j in range(test_latent1.shape[0]):
            #get the features of the nearest neighbours
            features_nearest_neighbours1 = features_train1[indices1[j]]
            #calculate the mean of the features of the nearest neighbours
            permuted_predicted_features1[j] = np.mean(features_nearest_neighbours1, axis=0)
        
        permuted_rmses1[i] = np.sqrt(np.mean((permuted_predicted_features1 - features_test1)**2))
        permuted_rmses_per_feature1[i] = np.sqrt(np.mean((permuted_predicted_features1 - features_test1)**2, axis=0))

    variance_explained_per_permutation = 1 - (rmse_per_feature1 / permuted_rmses_per_feature1)
    mean_variance_explained_per_feature1 = np.mean(variance_explained_per_permutation, axis=0)
    std_variance_explained_per_feature1 = np.std(variance_explained_per_permutation, axis=0)

    #determine the variance explained by the model for each feature and total variance explained (variance explained = 1 - (rmse / mean_permuted_rmse))
    variance_explained_per_feature1 = mean_variance_explained_per_feature1

    #initialize model
    latent_space_size2 = parameters2['latent_space_size']
    slope_leaky2 = parameters2['slope_leaky']
    learning_rate2 = parameters2['learning_rate']
    precision_model2 = parameters2['precision_model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2 = autoencoder_functions.VAE_KL_leaky(z_dim = latent_space_size2, device_name='cuda',slope=slope_leaky2,lr = learning_rate2, model_precision =precision_model2)
    model2 = model2.to(device)

    #load the model
    model_file_name2 = path_to_model2 + '/model.pt'
    model2.load_state_dict(torch.load(model_file_name2))

    train_latent2 = model2.get_latent(latent_train_loader2)
    test_latent2 = model2.get_latent(latent_test_loader2)

    X_train2 = train_latent2
    X_test2 = test_latent2
    # #perform normalization on the train data and transform it to the test data
    scaler = StandardScaler()
    X_train2 = scaler.fit_transform(X_train2)
    X_test2 = scaler.transform(X_test2)

    train_latent2 = X_train2
    test_latent2 = X_test2


    # # #perform umap on train_latent and fit to test latent
    # reducer = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 2)
    # reducer.fit(train_latent2)
    # train_latent2 = reducer.transform(train_latent2)
    # test_latent2 = reducer.transform(test_latent2)

    print('train latent shape:',train_latent2.shape)
    print('test latent shape:',test_latent2.shape)
  
    #get the nearest neighbours of the test set in the train set
    neigh2 = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
    neigh2.fit(train_latent2)
    distances2, indices2 = neigh2.kneighbors(test_latent2)
    print("Indices shape: ", indices2.shape)
    print("Distances shape: ", distances2.shape)

    #get the features of the nearest neighbours
    features_nearest_neighbours2 = features_train2[indices2]
    print("Features nearest neighbours shape: ", features_nearest_neighbours2.shape)

    #determine the predicted features based on the nearest neighbours
    predicted_features2 = np.zeros((test_latent2.shape[0], features2.shape[1]))
    for i in range(test_latent2.shape[0]):
        #get the features of the nearest neighbours
        features_nearest_neighbours2 = features_train2[indices2[i]]
        #calculate the mean of the features of the nearest neighbours
        predicted_features2[i] = np.mean(features_nearest_neighbours2, axis=0)

    #assert if predicted features has the same shape as features test
    assert predicted_features2.shape == np.array(features_test2).shape, "Number of predicted features and test features do not match"

    #determine the rmse of the prediction for each feature and the total rmse
    rmse_per_feature2 = np.sqrt(np.mean((predicted_features2 - features_test2)**2, axis=0))
    total_rmse2 = np.sqrt(np.mean((predicted_features2 - features_test2)**2))

    print("Total RMSE: ", total_rmse2)
    print("RMSE per feature: ", rmse_per_feature2)

    #now permut the test features n times and calculate the mean rmse and the standard deviation of the rmse for the
    #overall permuted rmse and the permuted rmse for each feature
    permuted_rmses2 = np.zeros(n_permutations)
    permuted_rmses_per_feature2 = np.zeros((n_permutations, features2.shape[1]))
    for i in range(n_permutations):
        np.random.shuffle(features_train2)

        permuted_predicted_features2 = np.zeros((test_latent2.shape[0], features2.shape[1]))
        for j in range(test_latent2.shape[0]):
            #get the features of the nearest neighbours
            features_nearest_neighbours2 = features_train2[indices2[j]]
            #calculate the mean of the features of the nearest neighbours
            permuted_predicted_features2[j] = np.mean(features_nearest_neighbours2, axis=0)
        
        permuted_rmses2[i] = np.sqrt(np.mean((permuted_predicted_features2 - features_test2)**2))
        permuted_rmses_per_feature2[i] = np.sqrt(np.mean((permuted_predicted_features2 - features_test2)**2, axis=0))
  
    variance_explained_per_permutation2 = 1 - (rmse_per_feature2 / permuted_rmses_per_feature2)
    mean_variance_explained_per_feature2 = np.mean(variance_explained_per_permutation2, axis=0)
    std_variance_explained_per_feature2 = np.std(variance_explained_per_permutation2, axis=0)

    #determine the variance explained by the model for each feature and total variance explained (variance explained = 1 - (rmse / mean_permuted_rmse))
    variance_explained_per_feature2 = mean_variance_explained_per_feature2

    #plot the variance explained per feature with error bars for the std of the permuted rmse per feature
    plt.figure(figsize=(10,5))
    bar_width = 0.35
    x = np.arange(len(variance_explained_per_feature1))
    #show total variance explained as a horizontal dotted line
    # plt.axhline(y=total_variance_explained2*100, color='r', linestyle='--', label='Total Variance Explained: {:.2f}%'.format(total_variance_explained2*100))
    plt.bar(x - bar_width/2, variance_explained_per_feature1*100, width = bar_width, yerr=std_variance_explained_per_feature1*100, alpha=1.0, label='Variance Explained Shank3 Model')
    plt.bar(x + bar_width/2, variance_explained_per_feature2*100, width = bar_width, yerr=std_variance_explained_per_feature2*100, alpha=1.0, label='Variance Explained Pups Model')
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
    plt.figtext(0.01, 0.05 ,'Shank3 Model: {}'.format(path_to_model1.split('/')[-1]), fontsize=8)
    plt.figtext(0.01, 0.01 ,'Pups Model: {}'.format(path_to_model2.split('/')[-1]), fontsize=8)
    plt.show()

def from_voc_to_latent_vis(spec, emitter, emitter_library, path_to_model, combine_train_test = False, use_strain = False):

        #load the trained parameters used in training
    parameters = joblib.load(path_to_model + '/parameters.pkl')

    ##load spec_indices for train and test set
    spec_indices_train = torch.load(path_to_model + '/spec_indices_train')
    spec_indices_test = torch.load(path_to_model + '/spec_indices_test')

    emitter_int = [emitter_library[e] for e in emitter]

    #remove doubles in train set  
    print('indices before removal of doubles =', len(spec_indices_train))
    spec_indices_train_after_removal = []
    for i in range(len(spec_indices_train)):
        if spec_indices_train[i] not in spec_indices_train_after_removal:
            spec_indices_train_after_removal.append(spec_indices_train[i])
    spec_indices_train = spec_indices_train_after_removal
    print('indices after removal of doubles =', len(spec_indices_train_after_removal))

    #get the spectograms and emitters
    train_specs = []
    test_specs = []
    labels_train = []
    labels_test = []

    for i in spec_indices_train:
        train_specs.append(spec[i])
        labels_train.append(emitter_int[i])
    for i in spec_indices_test:
        test_specs.append(spec[i])
        labels_test.append(emitter_int[i])

    #convert labels to integer numpy arrays
    labels_train = np.array(labels_train)
    labels_train = labels_train.astype(int)
    labels_test = np.array(labels_test)
    labels_test = labels_test.astype(int)

    #check if the same labels are present in labels_train and labels_test
    # Get the unique labels from each array
    unique_labels_train = np.unique(labels_train)
    unique_labels_test = np.unique(labels_test)

    # Check if the sets of unique labels are equal
    if np.array_equal(unique_labels_train, unique_labels_test):
        print("The same labels are present in both labels_train and labels_test.")
    else:
        print("The labels in labels_train and labels_test are different.")
        print("Unique labels in training set:", unique_labels_train)
        print("Unique labels in testing set:", unique_labels_test)

    if combine_train_test:
        test_specs = train_specs + test_specs
        labels_test = np.concatenate((labels_train, labels_test), axis=0)

    #print the size of test specs and labels test
    print('Total:',len(labels_test))
    print('Number of spectograms:',len(test_specs))

    #now pad and normalize the test specs
    padding_length = 160
    test_padded_specs = create_padded_spectograms.pad_spectograms_sorted(test_specs, padding_length)
    if parameters['max_value_per_spec'] == 'true':
        max_value_per_spec = True
        print(max_value_per_spec)
    test_padded_specs_normalized = normalize_padded_spectograms.normalize_specs(test_padded_specs, max_value_per_spec)

    #now choose a random spectogram from the test set and visualize it before and after padding and normalization
    random_index = random.randint(0, len(test_specs)-1)

    #get the spectral line image from the features_single python file for this random spectogram
    spectral_line_index = features_single.get_spectral_line_index(test_specs[random_index])
    spectral_line_index = cp.asnumpy(spectral_line_index)

    test_spec_that_will_be_used = test_specs[random_index]
    test_spec_that_will_be_used = np.array(test_spec_that_will_be_used)
    spectral_line_image = np.zeros(test_spec_that_will_be_used.shape)
    for i in range(len(spectral_line_index)):
        spectral_line_image[spectral_line_index[i],i] = 1
    
    # spectral_line_image = spectral_line_image * test_spec_that_will_be_used # use if you want to visualize the intensity of the line

    #get features for this spectogram
    features = features_single.get_usv_features(test_specs[random_index])
    #make a table with the features and their values
    feature_names = ['duration (s)', 'mean_freq (Hz)', 'min_freq (Hz)', 'max_freq (Hz)', 'bandwidth (Hz)', 'starting_freq (Hz)', 'stopping_freq (Hz)', 'directionality', 'variance', 'variance_spec_line', 'local_variability', 'nr_of_steps_up', 'nr_of_steps_down', 'nr_of_peaks', 'nr_of_valleys']
    feature_values = [features[0], features[1], features[2], features[3], features[4], features[5], features[6], features[7], features[8], features[9], features[10], features[11], features[12], features[13], features[14]]
    #remove brackets around feature_values
    feature_values = [item[0] for item in feature_values]
    #make the last 4 features integers and round the rest of the values to 2 decimal places except the first to 4
    feature_values[0] = np.round(feature_values[0], 4)
    feature_values[1] = np.round(feature_values[1], 2)
    feature_values[2] = np.round(feature_values[2], 2)
    feature_values[3] = np.round(feature_values[3], 2)
    feature_values[4] = np.round(feature_values[4], 2)
    feature_values[5] = np.round(feature_values[5], 2)
    feature_values[6] = np.round(feature_values[6], 2)
    feature_values[7] = np.round(feature_values[7], 2)
    feature_values[8] = np.round(feature_values[8], 2)
    feature_values[9] = np.round(feature_values[9], 2)
    feature_values[10] = np.round(feature_values[10], 2)
    feature_values[11] = int(feature_values[11])
    feature_values[12] = int(feature_values[12])
    feature_values[13] = int(feature_values[13])
    feature_values[14] = int(feature_values[14])
    table_data = [[name, value] for name, value in zip(feature_names, feature_values)]
    column_titles = ['Feature', 'Value']

    #get latent representation of the test spectograms
    test_padded_specs_tensor = torch.utils.data.DataLoader(test_padded_specs_normalized, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_space_size = parameters['latent_space_size']
    slope_leaky = parameters['slope_leaky']
    learning_rate = parameters['learning_rate']
    precision_model = parameters['precision_model']
    model = autoencoder_functions.VAE_KL_leaky(z_dim = latent_space_size, device_name='cuda',slope=slope_leaky,lr = learning_rate, model_precision =precision_model)
    model = model.to(device)
    model_file_name = path_to_model + '/model.pt'
    model.load_state_dict(torch.load(model_file_name))
    test_latent = model.get_latent(test_padded_specs_tensor)
    test_latent = cp.asnumpy(test_latent)

    #get the latent space of the random spectogram
    random_spectrogram_latent = test_latent[random_index]
    random_spectrogram_latent = random_spectrogram_latent.reshape(1, -1)  # Reshape to 2D array with one row

    #perform umap on the latent space to reduce it to 2 dimensions for visualization
    reducer = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 2)
    reducer.fit(test_latent)

    latent_2d = reducer.transform(test_latent)

    #make a a 100x100 grid around all points
    num_bins = 50
    # This is the correct way
    x_min, x_max = latent_2d[:, 0].min(), latent_2d[:, 0].max()
    y_min, y_max = latent_2d[:, 1].min(), latent_2d[:, 1].max()
    # 1. Calculate the range of each dimension
    x_range = x_max - x_min
    y_range = y_max - y_min

    # 2. Find the largest range
    max_range = max(x_range, y_range)

    # 3. Calculate the center of each dimension
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    # 4. Set the new, equal limits for both axes
    x_lim_min = x_center - max_range / 2
    x_lim_max = x_center + max_range / 2
    y_lim_min = y_center - max_range / 2
    y_lim_max = y_center + max_range / 2


    #give each emitter a color and make the random spectogram stand out with a very distinct color
    unique_emitters, counts = np.unique(labels_test, return_counts= True)

    if use_strain:
        points_label1 = latent_2d[labels_test == unique_emitters[0]]
        counts1, _, _ = np.histogram2d(points_label1[:,0], points_label1[:,1], bins = num_bins, range= [[x_min,x_max],[y_min,y_max]])
        points_label2 = latent_2d[labels_test == unique_emitters[1]]
        counts2, _, _ = np.histogram2d(points_label2[:,0], points_label2[:,1], bins = num_bins, range= [[x_min,x_max],[y_min,y_max]])
        weights1 = 1/counts[0]
        weights2 = 1/counts[1]
        counts1 = counts1 * weights1
        counts2 = counts2 * weights2
        density_difference = counts1 - counts2



    colors = plt.cm.get_cmap('tab20', len(unique_emitters))


    emitter_colors = {emitter: colors(i) for i, emitter in enumerate(unique_emitters)}
    #make sure the specific names in emitter _library are in the legend
    emitter_library_reversed = {v: k for k, v in emitter_library.items()}

    if use_strain:
        emitter_library_reversed = emitter_int
        # 0 is WT and 1 is KO
        emitter_colors = {0: 'blue', 1: 'red'}
        unique_emitters = [0, 1]


    fig = plt.figure(figsize=(15,15))
    fig.suptitle('Feature extraction pipeline', fontsize=20, fontweight='bold', y = 1.01)
    gs = GridSpec(9, 9, figure=fig)
    # subplot 1 original spectrogram
    ax1 = fig.add_subplot(gs[0:3, 0:3])
    im1 = ax1.imshow(test_specs[random_index], origin='lower')
    ax1.set_title('Original Spectrogram')
    fig.colorbar(im1, ax=ax1, shrink=0.5)
    ax1.set_xlabel('Time (bins)')
    ax1.set_ylabel('Frequency (bins)')
    ax1.set_aspect('auto')
    ax1.text(-0.18, 1.1, 'A', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    # subplot 2 padded spectrogram
    ax2 = fig.add_subplot(gs[0:3, 3:6])
    im2 = ax2.imshow(test_padded_specs[random_index], origin='lower')
    fig.colorbar(im2, ax=ax2, shrink=0.5)
    ax2.set_title('Padded Spectrogram')
    ax2.set_xlabel('Time (bins)')
    ax2.set_ylabel('Frequency (bins)')
    ax2.set_aspect('auto')
    ax2.text(-0.18, 1.1, 'B', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    # Add an arrow between ax1 and ax2
    ax1_pos = ax1.get_position()
    ax2_pos = ax2.get_position()
    plt.annotate('', xy=(ax2_pos.x0 -0.035, ax2_pos.y0 + ax2_pos.height / 2 -0.04), xytext=(ax1_pos.x1 -0.025, ax1_pos.y0 + ax1_pos.height / 2 -0.04),
        arrowprops=dict(facecolor='black', width=2, headwidth=8, headlength=8, shrink=0.05), xycoords='figure fraction', textcoords='figure fraction', annotation_clip= False)
    # subplot 3 normalized spectrogram
    ax3 = fig.add_subplot(gs[0:3, 6:9])
    im3 = ax3.imshow(test_padded_specs_normalized[random_index], origin='lower')
    ax3.set_title('Normalized Spectrogram')
    fig.colorbar(im3, ax=ax3, shrink=0.5)
    ax3.set_xlabel('Time (bins)')
    ax3.set_ylabel('Frequency (bins)')
    ax3.set_aspect('auto')
    ax3.text(-0.18, 1.1, 'C', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    # Add an arrow between ax2 and ax3
    ax2_pos = ax2.get_position()
    ax3_pos = ax3.get_position()
    plt.annotate('', xy=(ax3_pos.x0 +0.015, ax3_pos.y0 + ax3_pos.height / 2 -0.04), xytext=(ax2_pos.x1 +0.025, ax2_pos.y0 + ax2_pos.height / 2 -0.04),
        arrowprops=dict(facecolor='black', width=2, headwidth=8, headlength=8, shrink=0.05), xycoords='figure fraction', textcoords='figure fraction', annotation_clip= False)
    # subplot 4 spectral line image
    ax4 = fig.add_subplot(gs[3:6, 0:3])
    im4 = ax4.imshow(spectral_line_image, origin='lower')
    ax4.set_title('Spectral Line Image')
    fig.colorbar(im4, ax=ax4, shrink=0.5)  
    # ylabel zero bin and 128 bins that go up to 100000 with steps of 781.25
    ax4.set_yticks([0, 32, 64, 96, 128])
    ax4.set_yticklabels(['0', '25000', '50000', '75000', '100000'])
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_aspect('auto')
    ax4.text(-0.18, 1.1, 'D', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    #xticks with each bin being 227.559 ms, so givve xticks every 1000/227.559 = 4.39 bins for the duration, feature 0
    num_bins = test_specs[random_index].shape[1]
    xtick_positions = np.arange(0, num_bins, 10/(227.559/200)) #every 10 ms a bin
    #get length of xtick_position and for every length add a label for 1000 ms
    xtick_labels = [str(int(i * 10)) for i in range(len(xtick_positions))]
    ax4.set_xticks(xtick_positions)
    ax4.set_xticklabels(xtick_labels)
    ax4.set_xlabel('Time (ms)')
    #Add an arrow from ax1 to ax4
    ax1_pos = ax1.get_position()
    ax4_pos = ax4.get_position()
    plt.annotate('', xy=(ax4_pos.x0 + ax4_pos.width / 2, ax4_pos.y1 + 0.005), xytext=(ax1_pos.x0 + ax1_pos.width / 2, ax1_pos.y0 + 0.03),
        arrowprops=dict(facecolor='black', width=2, headwidth=8, headlength=8, shrink=0.05), xycoords='figure fraction', textcoords='figure fraction', annotation_clip= False)
    # subplot 5 and 6 latent representation visualize array as a 1,24 array with a colorbar spanning 2 subplots
    ax5 = fig.add_subplot(gs[3:4, 3:9])
    im5 = ax5.imshow(random_spectrogram_latent, cmap='viridis')
    ax5.set_title('Latent Representation')
    fig.colorbar(im5, ax=ax5, orientation='vertical')
    ax5.set_yticks([0])
    # ax5.set_yticklabels(['Latent Space'])
    ax5.set_xlabel('Latent Dimensions')
    ax5.text(-0.08, 3, 'E', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    #Add arrow from 4 to 5
    plt.annotate(
    '',
    xy=(0.825, 0.61), # Head of the arrow (x=0.25, y=0.45)
    xytext=(0.825, 0.65), # Tail of the arrow (x=0.25, y=0.65)
    arrowprops=dict(
        facecolor='black',
        width=2,
        headwidth=8,
        headlength=8,
        shrink=0.05
    ),
    xycoords='figure fraction',
    textcoords='figure fraction',
    annotation_clip=False
)
    # ax5.set_aspect('auto')

    # subplot 7 table with features and their values
    ax7 = fig.add_subplot(gs[6:9, 0:3])
    ax7.axis('off')
    ax7.table(cellText=table_data, colLabels=column_titles, loc='center')
    ax7.text(-0.15, 1.0, 'F', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    # ax7.set_title('Extracted Features')
    ax7.text(0.5, 0.9, 'Extracted Features', fontsize = 12, ha = 'center', va = 'center', transform=ax7.transAxes)
    #now add text beneath the table to show the emitter of this spectogram and the model used
    emitter_name = emitter_library_reversed[labels_test[random_index]]
    if use_strain:
        emitter_name = 'WT' if labels_test[random_index] == 0 else 'KO'
    ax7.text(0, 0, 'Emitter: {}'.format(emitter_name), fontsize = 10,  transform=ax7.transAxes)
    model_name = path_to_model.split('/')[-1]
    ax7.text(0, -0.05, 'Model: {}'.format(model_name), fontsize = 10,  transform=ax7.transAxes)
    # Add arrow from 4 to 7
    plt.annotate(
    '',
    xy=(0.225, 0.28), # Head of the arrow (x=0.25, y=0.45)
    xytext=(0.225, 0.34), # Tail of the arrow (x=0.25, y=0.65)
    arrowprops=dict(
        facecolor='black',
        width=2,
        headwidth=8,
        headlength=8,
        shrink=0.05
    ),
    xycoords='figure fraction',
    textcoords='figure fraction',
    annotation_clip=False
)
    # subplot 8 and 9 umap visualization of the latent space with the random spect
    ax8 = fig.add_subplot(gs[4:9, 3:9])
    if use_strain:
        vmax = max(abs(density_difference.min()), abs(density_difference.max()))
        original_cmap = plt.cm.get_cmap('RdBu_r')

        # Get the full list of colors from the original colormap's lookup table
        colors = original_cmap(np.linspace(0, 1, original_cmap.N))

        # Find the center of the color list (the pure white color)
        # The N // 2 trick handles both even and odd length colormaps
        center_index = original_cmap.N // 2

        # Set the middle color to pure white (RGB value is [1, 1, 1, 1])
        colors[center_index] = np.array([1, 1, 1, 1])

        # Create the new colormap from the modified color list
        my_cmap = LinearSegmentedColormap.from_list("my_diverging_cmap", colors)
        im8 = ax8.imshow(density_difference.T, extent= [x_min, x_max, y_min, y_max],origin='lower', cmap = my_cmap, vmin = -vmax, vmax = vmax)
        # Apply the new limits to your plot
        ax8.set_xlim(x_lim_min, x_lim_max)
        ax8.set_ylim(y_lim_min, y_lim_max)
        ax8.set_facecolor('white')
        ax8.set_title('Density Difference Emitters in UMAP Projection of Latent Space')
        ax8.set_xlabel('UMAP Dimension 1')
        ax8.set_ylabel('UMAP Dimension 2')
        cbar = fig.colorbar(im8, ax=ax8, shrink = 0.5)
        cbar.set_label(f'Higher Density of WT <--  --> Higher Density of KO')

    else:
        for emitter in unique_emitters:
            emitter_mask = labels_test == emitter
            ax8.scatter(latent_2d[emitter_mask, 0], latent_2d[emitter_mask, 1], color=emitter_colors[emitter], label=str(emitter), alpha=0.8)
        #highlight the random spectrogram with a black edge
        ax8.scatter(latent_2d[random_index, 0], latent_2d[random_index, 1], color='none', edgecolor='black', s=100, label='Selected Spectrogram')
        ax8.set_title('UMAP Projection of Latent Space')
        #make a legend for individual names and show the selected spectrogram in the legend
        handles, labels = ax8.get_legend_handles_labels()
        #change labels according to emitter_library_reversed
        new_labels = []
        for label in labels:
            if label.isdigit():  # Check if the label is a digit
                int_label = int(label)
                if int_label in emitter_library_reversed:
                    new_labels.append(emitter_library_reversed[int_label])
                else:
                    new_labels.append(label)  # If not found, keep the original label
            else:
                new_labels.append(label)  # If not a digit, keep the original label
        if use_strain:
            new_labels = ['WT' if lbl == '0' else 'KO' if lbl == '1' else lbl for lbl in new_labels]
        #set the legend inside the plot to the right
        ax8.legend(handles, new_labels, loc='upper right')
        if use_strain:
            ax8.legend(handles, ['WT', 'KO', 'Selected Spectrogram'], loc='upper right')
        ax8.set_xlabel('UMAP Dimension 1')
        ax8.set_ylabel('UMAP Dimension 2')
    ax8.text(-0.055, 1.05, 'G', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    # Add arrow from 5 to 8
    plt.annotate(
    '',
    xy=(0.825, 0.53), # Head of the arrow (x=0.25, y=0.45)
    xytext=(0.825, 0.57), # Tail of the arrow (x=0.25, y=0.65)
    arrowprops=dict(
        facecolor='black',
        width=2,
        headwidth=8,
        headlength=8,
        shrink=0.05
    ),
    xycoords='figure fraction',
    textcoords='figure fraction',
    annotation_clip=False)
    # plt.tight_layout()
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.95,
        wspace=0.5, # Increased horizontal space
        hspace=1  # Increased vertical space
    )
    plt.show()

def emitter_prediction_model_weighted_and_hardcoded(spec,features, emitter, emitter_library, path_to_model, n_neighbours = 30, n_permutations =100):

        #load the trained parameters used in training
    parameters = joblib.load(path_to_model + '/parameters.pkl')

    ##load spec_indices for train and test set
    spec_indices_train = torch.load(path_to_model + '/spec_indices_train')
    spec_indices_test = torch.load(path_to_model + '/spec_indices_test')

    emitter_int = [emitter_library[e] for e in emitter]

    #remove doubles in train set  
    print('indices before removal of doubles =', len(spec_indices_train))
    spec_indices_train_after_removal = []
    for i in range(len(spec_indices_train)):
        if spec_indices_train[i] not in spec_indices_train_after_removal:
            spec_indices_train_after_removal.append(spec_indices_train[i])
    spec_indices_train = spec_indices_train_after_removal
    print('indices after removal of doubles =', len(spec_indices_train_after_removal))

    #get the spectograms and emitters
    train_specs = []
    test_specs = []
    labels_train = []
    labels_test = []

    for i in spec_indices_train:
        train_specs.append(spec[i])
        labels_train.append(emitter_int[i])
    for i in spec_indices_test:
        test_specs.append(spec[i])
        labels_test.append(emitter_int[i])

    #convert labels to integer numpy arrays
    labels_train = np.array(labels_train)
    labels_train = labels_train.astype(int)
    labels_test = np.array(labels_test)
    labels_test = labels_test.astype(int)

    #check if the same labels are present in labels_train and labels_test
    # Get the unique labels from each array
    unique_labels_train = np.unique(labels_train)
    unique_labels_test = np.unique(labels_test)

    # Check if the sets of unique labels are equal
    if np.array_equal(unique_labels_train, unique_labels_test):
        print("The same labels are present in both labels_train and labels_test.")
    else:
        print("The labels in labels_train and labels_test are different.")
        print("Unique labels in training set:", unique_labels_train)
        print("Unique labels in testing set:", unique_labels_test)

    #make a weight list for the integers in labels train
    unique_labels, counts = np.unique(labels_train, return_counts=True)
    weights = {}
    for label, count in zip(unique_labels, counts):
        weights[label] = 1 / count

    print(weights)

    #check if there are as many labels as spectograms in the train and test sets
    assert len(train_specs) == len(labels_train), "Number of train spectograms and labels do not match"
    assert len(test_specs) == len(labels_test), "Number of test spectograms and labels do not match"

    #pad the spectograms according to our training procedure
    padding_length = 160
    train_padded_specs = create_padded_spectograms.pad_spectograms_sorted(train_specs, padding_length)
    test_padded_specs = create_padded_spectograms.pad_spectograms_sorted(test_specs, padding_length)

    #delete unnecesarry big files in RAM
    del spec, train_specs, test_specs

    #normalize according to our training
    if parameters['max_value_per_spec'] == 'true':
        max_value_per_spec = True
        print(max_value_per_spec)
    train_padded_specs = normalize_padded_spectograms.normalize_specs(train_padded_specs, max_value_per_spec)
    test_padded_specs = normalize_padded_spectograms.normalize_specs(test_padded_specs, max_value_per_spec)

    #convert to torch tensors
    latent_train_loader = torch.utils.data.DataLoader(train_padded_specs, batch_size=1, shuffle=False)
    latent_test_loader = torch.utils.data.DataLoader(test_padded_specs, batch_size=1, shuffle=False)

    #delete the padded spectograms from memory
    del train_padded_specs, test_padded_specs

    #initialize model
    latent_space_size = parameters['latent_space_size']
    slope_leaky = parameters['slope_leaky']
    learning_rate = parameters['learning_rate']
    precision_model = parameters['precision_model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = autoencoder_functions.VAE_KL_leaky(z_dim = latent_space_size, device_name='cuda',slope=slope_leaky,lr = learning_rate, model_precision =precision_model)
    model = model.to(device)

    #load the model
    model_file_name = path_to_model + '/model.pt'
    model.load_state_dict(torch.load(model_file_name))

    train_latent = model.get_latent(latent_train_loader)
    test_latent = model.get_latent(latent_test_loader)

    # X_train = train_latent
    # X_test = test_latent
    # # #perform normalization on the train data and transform it to the test data
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # train_latent = X_train
    # test_latent = X_test


    # # #perform umap on train_latent and fit to test latent
    # reducer = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 2)
    # reducer.fit(train_latent)
    # train_latent = reducer.transform(train_latent)
    # test_latent = reducer.transform(test_latent)

    print('train latent shape:',train_latent.shape)
    print('test latent shape:',test_latent.shape) 
  
    #get the nearest neighbours of the test set in the train set
    neigh = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
    neigh.fit(train_latent)
    distances, indices = neigh.kneighbors(test_latent)
    print("Indices shape: ", indices.shape)
    print("Distances shape: ", distances.shape)

    #get the labels of the nearest neighbours
    labels_nearest_neighbours = labels_train[indices]
    print("Labels nearest neighbours shape: ", labels_nearest_neighbours.shape)

    #determine the predicted label based on the nearest neighbours
    predicted_labels = np.zeros(test_latent.shape[0])
    for i in range(test_latent.shape[0]):
        #get the labels of the nearest neighbours
        labels_nearest_neighbours = labels_train[indices[i]]
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
    accuracy = np.sum(predicted_labels == labels_test) / len(labels_test)

    # Get a complete list of all unique labels from both train and test sets
    all_labels = np.unique(np.concatenate((labels_train, labels_test), axis=0))

    # Initialize accuracy dictionary for all possible labels
    accuracy_per_class = {label: 0.0 for label in all_labels}
    
    # Calculate accuracy for each class that exists in the labels_test
    unique_labels_present = np.unique(labels_test)
    for label in unique_labels_present:
        accuracy_per_class[label] = np.sum(predicted_labels[labels_test == label] == labels_test[labels_test == label]) / len(labels_test[labels_test == label])

    print("Overall accuracy: ", accuracy)
    print("Accuracy per class: ", accuracy_per_class)

    #now permut the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy for the
    #overall permuted accurcacy and the permuted accuracy for each class
    permuted_accuracies = np.zeros(n_permutations)
    permuted_accuracies_per_class = {label: np.zeros(n_permutations) for label in all_labels}
    for i in range(n_permutations):
        np.random.shuffle(labels_train)

        permuted_predicted_labels = np.zeros(test_latent.shape[0])
        for j in range(test_latent.shape[0]):
            #get the labels of the nearest neighbours
            labels_nearest_neighbours = labels_train[indices[j]]
            #get the unique labels and their counts
            unique_labels_nn, counts = np.unique(labels_nearest_neighbours, return_counts=True)
            #calculate weighted counts
            weighted_counts = np.zeros(len(unique_labels_nn))
            for k in range(len(unique_labels_nn)):
                label = unique_labels_nn[k]
                weighted_counts[k] = counts[k] * weights[label]
            #get the index of the label with the most counts
            permuted_predicted_labels[j] = unique_labels_nn[np.argmax(weighted_counts)]
        
        permuted_accuracies[i] = np.sum(permuted_predicted_labels == labels_test) / len(labels_test)

        # Calculate and store permuted accuracy for each class that exists in this permutation
        unique_labels_present_in_perm = np.unique(labels_test)
        for label in unique_labels_present_in_perm:
                permuted_accuracies_per_class[label][i] = np.sum(permuted_predicted_labels[labels_test == label] == labels_test[labels_test == label]) / len(labels_test[labels_test == label])

    mean_permuted_accuracies_per_class = {label: np.mean(permuted_accuracies_per_class[label]) for label in all_labels}
    std_permuted_accuracies_per_class = {label: np.std(permuted_accuracies_per_class[label]) for label in all_labels}
    
    mean_permuted_accuracy = np.mean(permuted_accuracies)
    std_permuted_accuracy = np.std(permuted_accuracies)

    #calculate the p-values for every individual emitter per class and the p-value for the total accuracy
    count_higher_per_class = {label: np.sum(permuted_accuracies_per_class[label] >= accuracy_per_class[label]) for label in all_labels}
    count_higher = np.sum(permuted_accuracies >= accuracy)
    p_value_per_class = {label: count_higher_per_class[label] / n_permutations for label in all_labels}
    p_value = count_higher / n_permutations 
    
    print("Mean permuted accuracy: ", mean_permuted_accuracy)
    print("Std permuted accuracy: ", std_permuted_accuracy)
    print("Mean permuted accuracies per class: ", mean_permuted_accuracies_per_class)
    print("Std permuted accuracies per class: ", std_permuted_accuracies_per_class)


    #create an indices list with len emitter
    labels2 = []
    for i in emitter:
        label2 = emitter_library[i]
        labels2.append(label2)
    print(np.unique(labels2, return_counts=True))

    print('Amount of vocalisations',len(labels2))

    indices2 = []
    labels_after_removal2 = []
    #remove vocalisations from either starting with capital F or starting with mother or starting with pupx, when checking emitter library
    for i, emit in enumerate(emitter):
        if not emit.startswith('F') and not emit.startswith('mother') and not emit.startswith('pupx'):
            indices2.append(i)
            labels_after_removal2.append(labels2[i])


    labels2 = labels_after_removal2
    labels2 = np.array(labels2)
    indices2 = np.array(indices2)
    print('Amount of vocalisations after females removed:',len(indices2))
    print('Females removed:',len(emitter)-len(indices2))

    #remove features not in indices
    features_after_removal2 = []
    for i in indices2:
        features_after_removal2.append(features[i])

    features2 = features_after_removal2

    print('labels after removal =',np.unique(labels2,return_counts=True))
    #seperate the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features2, labels2, test_size=0.2, random_state=42)
    print('Total:',len(y_train))
    print('Number of features:',len(X_train[0]))
    print('Total:',len(y_test))
    print('Number of features:',len(X_test[0]))

    #perform normalization on the train data and transform it to the test data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_latent2 = X_train
    test_latent2 = X_test
    labels_train2 = y_train
    labels_test2 = y_test

    #make a weight list for the integers in labels train
    unique_labels2, counts2 = np.unique(labels_train2, return_counts=True)
    weights2 = {}
    for label, count in zip(unique_labels2, counts2):
        weights2[label] = 1 / count

    print('weights =',weights2)

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
        unique_labels, counts = np.unique(labels_nearest_neighbours2, return_counts=True)
        #calculate wieghted counts
        weighted_counts = np.zeros(len(unique_labels))
        for j in range(len(unique_labels)):
            label2 = unique_labels[j]
            weighted_counts[j] = counts[j] * weights2[label2]
        #get the index of the label with the most counts
        predicted_labels2[i] = unique_labels[np.argmax(weighted_counts)]      

    #determine the accuracy of the total prediction and the accuracy of the prediction for each class
    accuracy2 = np.sum(predicted_labels2 == labels_test2) / len(labels_test2)

    # Get a complete list of all unique labels from both train and test sets
    all_labels2 = np.unique(np.concatenate((labels_train2, labels_test2), axis=0))

    # Initialize accuracy dictionary for all possible labels
    accuracy_per_class2 = {label2: 0.0 for label2 in all_labels2}
    
    # Calculate accuracy for each class that exists in the labels_test
    unique_labels_present2 = np.unique(labels_test2)
    for label2 in unique_labels_present2:
        accuracy_per_class2[label2] = np.sum(predicted_labels2[labels_test2 == label2] == labels_test2[labels_test2 == label2]) / len(labels_test2[labels_test2 == label2])

    print("Overall accuracy: ", accuracy2)
    print("Accuracy per class: ", accuracy_per_class2)

    #now permut the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy for the
    #overall permuted accurcacy and the permuted accuracy for each class
    permuted_accuracies2 = np.zeros(n_permutations)
    permuted_accuracies_per_class2 = {label2: np.zeros(n_permutations) for label2 in all_labels2}
    for i in range(n_permutations):
        np.random.shuffle(labels_train2)

        permuted_predicted_labels2 = np.zeros(test_latent2.shape[0])
        for j in range(test_latent2.shape[0]):
            #get the labels of the nearest neighbours
            labels_nearest_neighbours2 = labels_train2[indices2[j]]
            #get the unique labels and their counts
            unique_labels_nn, counts = np.unique(labels_nearest_neighbours2, return_counts=True)
            #calculate weighted counts
            weighted_counts = np.zeros(len(unique_labels_nn))
            for k in range(len(unique_labels_nn)):
                label = unique_labels_nn[k]
                weighted_counts[k] = counts[k] * weights2[label]
            #get the index of the label with the most counts
            permuted_predicted_labels2[j] = unique_labels_nn[np.argmax(weighted_counts)]
        
        permuted_accuracies2[i] = np.sum(permuted_predicted_labels2 == labels_test2) / len(labels_test2)

        # Calculate and store permuted accuracy for each class that exists in this permutation
        unique_labels_present_in_perm2 = np.unique(labels_test2)
        for label2 in unique_labels_present_in_perm2:
                permuted_accuracies_per_class2[label2][i] = np.sum(permuted_predicted_labels2[labels_test2 == label2] == labels_test2[labels_test2 == label2]) / len(labels_test2[labels_test2 == label2])

    mean_permuted_accuracies_per_class2 = {label2: np.mean(permuted_accuracies_per_class2[label2]) for label2 in all_labels2}
    std_permuted_accuracies_per_class2 = {label2: np.std(permuted_accuracies_per_class2[label2]) for label2 in all_labels2}
    
    mean_permuted_accuracy2 = np.mean(permuted_accuracies2)
    std_permuted_accuracy2 = np.std(permuted_accuracies2)
    
    #calculate the p-values for every individual emitter per class and the p-value for the total accuracy
    count_higher_per_class2 = {label2: np.sum(permuted_accuracies_per_class2[label2] >= accuracy_per_class2[label2]) for label2 in all_labels2}
    count_higher2 = np.sum(permuted_accuracies2 >= accuracy2)
    p_value_per_class2 = {label2: count_higher_per_class2[label2] / n_permutations for label2 in all_labels2}
    p_value2 = count_higher2 / n_permutations

    print("Mean permuted accuracy: ", mean_permuted_accuracy2)
    print("Std permuted accuracy: ", std_permuted_accuracy2)
    print("Mean permuted accuracies per class: ", mean_permuted_accuracies_per_class2)
    print("Std permuted accuracies per class: ", std_permuted_accuracies_per_class2)
    
    # all_labels = list(all_labels)
    # all_labels = all_labels[0:6] + all_labels[8:] + all_labels[6:8]
    # all_labels2 = list(all_labels2)
    # all_labels2 = all_labels2[0:6] + all_labels2[8:] + all_labels2[6:8]
   #now plot the accurcay per class and permuted accuracy per class in subplot 1 and the overall accuracy and permuted accuracy in subplot 2
    plt.figure(figsize=(12, 10)) # It's a good practice to set a figure size for better readability
    plt.suptitle('Emitter prediction for dataset', fontweight='bold')

    plt.subplot(2, 1, 1)
    normal_accuracies = [accuracy_per_class[label] for label in all_labels]
    permuted_accuracies = [mean_permuted_accuracies_per_class[label] for label in all_labels]
    permuted_std_errors = [std_permuted_accuracies_per_class[label] for label in all_labels]
    x_labels = [str(label) for label in all_labels]
    p_value_per_class = [p_value_per_class[label] for label in all_labels]

    # Add the overall accuracy as an extra bar
    # Corrected: Use append without re-assignment
    normal_accuracies.append(accuracy)
    permuted_accuracies.append(mean_permuted_accuracy)
    permuted_std_errors.append(std_permuted_accuracy)
    x_labels.append('Overall accuracy')

    plt.bar(x_labels, normal_accuracies, yerr=[0]*len(normal_accuracies), label='Accuracy')
    plt.bar(x_labels, permuted_accuracies, yerr=permuted_std_errors, alpha=0.9, label='Permuted Accuracy')

    # For every bar in normal_accuracies, add a text with the p-value above the bar
    for i in range(len(p_value_per_class)):
        if p_value_per_class[i] < 0.01:
            plt.text(i, normal_accuracies[i] + 0.05, 'p < 0.01', ha='center', fontsize=8, rotation=90)
        else:
            plt.text(i, normal_accuracies[i] + 0.05, 'p = {:.2f}'.format(p_value_per_class[i]), ha='center', fontsize=8, rotation=90)
            
    # Corrected: Use the correct index for the last bar
    if p_value < 0.01:
        plt.text(len(normal_accuracies) - 1, normal_accuracies[-1] + 0.05, 'p < 0.01', ha='center', fontsize=8, rotation=90)
    else:
        plt.text(len(normal_accuracies) - 1, normal_accuracies[-1] + 0.05, 'p = {:.2f}'.format(p_value), ha='center', fontsize=8, rotation=90)

    plt.title('Prediction accuracy based latent features')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.text(-0.1, 1.0, 'A', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.ylim(0, 1)

    emitter_library = {v: k for k, v in emitter_library.items()}
    x_changed_labels = [emitter_library[label] for label in all_labels]
    # Corrected: Add the overall accuracy label to the changed labels list
    x_changed_labels.append('Total accuracy')
    plt.xticks(range(len(x_labels)), x_changed_labels, rotation=90)
    plt.grid()

    plt.subplot(2, 1, 2)
    normal_accuracies2 = [accuracy_per_class2[label2] for label2 in all_labels2]
    permuted_accuracies2 = [mean_permuted_accuracies_per_class2[label2] for label2 in all_labels2]
    permuted_std_errors2 = [std_permuted_accuracies_per_class2[label2] for label2 in all_labels2]
    x_labels2 = [str(label2) for label2 in all_labels2]
    p_value_per_class2 = [p_value_per_class2[label2] for label2 in all_labels2]


    # Add the overall accuracy as an extra bar
    # Corrected: Use append without re-assignment
    normal_accuracies2.append(accuracy2)
    permuted_accuracies2.append(mean_permuted_accuracy2)
    permuted_std_errors2.append(std_permuted_accuracy2)
    x_labels2.append('Overall accuracy')

    plt.bar(x_labels2, normal_accuracies2, yerr=[0]*len(normal_accuracies2), label='Accuracy')
    plt.bar(x_labels2, permuted_accuracies2, yerr=permuted_std_errors2, alpha=0.9, label='Permuted Accuracy')

    # For every bar in normal_accuracies, add a text with the p-value above the bar
    for i in range(len(p_value_per_class2)):
        if p_value_per_class2[i] < 0.01:
            plt.text(i, normal_accuracies2[i] + 0.05, 'p < 0.01', ha='center', fontsize=8, rotation=90)
        else:
            plt.text(i, normal_accuracies2[i] + 0.05, 'p = {:.2f}'.format(p_value_per_class2[i]), ha='center', fontsize=8, rotation=90) # Corrected label

    # Corrected: Use the correct index for the last bar
    if p_value2 < 0.01:
        plt.text(len(normal_accuracies2) - 1, normal_accuracies2[-1] + 0.05, 'p < 0.01', ha='center', fontsize=8, rotation=90)
    else:
        plt.text(len(normal_accuracies2) - 1, normal_accuracies2[-1] + 0.05, 'p = {:.2f}'.format(p_value2), ha='center', fontsize=8, rotation=90) # Corrected label

    plt.title('Prediction accuracy based on traditional features')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    x_changed_labels2 = [emitter_library[label2] for label2 in all_labels2]
    # Corrected: Add the overall accuracy label to the changed labels list
    x_changed_labels2.append('Total accuracy')
    plt.xticks(range(len(x_labels2)), x_changed_labels2, rotation=90)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.text(-0.1, 1.0, 'B', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)

    # Print model name used in bottom left corner of the plot
    plt.figtext(0.01, 0.01 ,'Model_name: {}'.format(path_to_model), fontsize=8, verticalalignment='bottom')
    plt.show()



def strain_prediction_model_weighted_and_hardcoded(spec,features, emitter,emitter_library,path_to_model, n_neighbours = 30, n_permutations =100):

    #load the trained parameters used in training
    parameters = joblib.load(path_to_model + '/parameters.pkl')

    ##load spec_indices for train and test set
    spec_indices_train = torch.load(path_to_model + '/spec_indices_train')
    spec_indices_test = torch.load(path_to_model + '/spec_indices_test')

    #load labels_train and labels_test based on the spec_indices
    labels_train = torch.load(path_to_model + '/labels_train')
    labels_test = torch.load(path_to_model + '/labels_test')

    print('indices before removal of doubles =', len(spec_indices_train))

    #remove doubles indices from spec_indices_train and remove them from labels_train as well
    spec_indices_train_after_removal = []
    labels_train_after_removal = []



    for i in range(len(spec_indices_train)):
        if spec_indices_train[i] not in spec_indices_train_after_removal:
            spec_indices_train_after_removal.append(spec_indices_train[i])
            labels_train_after_removal.append(labels_train[i])

    spec_indices_train = spec_indices_train_after_removal
    labels_train = labels_train_after_removal

    print('indices after removal of doubles =', len(spec_indices_train_after_removal))
    labels_train = np.array(labels_train)
    labels_train = labels_train.astype(int)
    labels_test = np.array(labels_test)
    labels_test = labels_test.astype(int)

    weight_WT = 1/np.sum(labels_train==0)
    weight_KO = 1/np.sum(labels_train==1)
    print('Weight WT =',weight_WT)
    print('Weight KO =', weight_KO)
    weights = {0: weight_WT, 1: weight_KO}

    #get the spectograms
    train_specs = []
    test_specs = []

    for i in spec_indices_train:
        train_specs.append(spec[i])
    for i in spec_indices_test:
        test_specs.append(spec[i])

    

    #check if there are as many labels as spectograms in the train and test sets
    assert len(train_specs) == len(labels_train), "Number of train spectograms and labels do not match"
    assert len(test_specs) == len(labels_test), "Number of test spectograms and labels do not match"

    #pad the spectograms according to our training procedure
    padding_length = 160
    train_padded_specs = create_padded_spectograms.pad_spectograms_sorted(train_specs, padding_length)
    test_padded_specs = create_padded_spectograms.pad_spectograms_sorted(test_specs, padding_length)

    #delete unnecesarry big files in RAM
    del spec, train_specs, test_specs

    #normalize according to our training
    if parameters['max_value_per_spec'] == 'true':
        max_value_per_spec = True
        print(max_value_per_spec)
    train_padded_specs = normalize_padded_spectograms.normalize_specs(train_padded_specs, max_value_per_spec)
    test_padded_specs = normalize_padded_spectograms.normalize_specs(test_padded_specs, max_value_per_spec)

    #convert to torch tensors
    latent_train_loader = torch.utils.data.DataLoader(train_padded_specs, batch_size=1, shuffle=False)
    latent_test_loader = torch.utils.data.DataLoader(test_padded_specs, batch_size=1, shuffle=False)

    #delete the padded spectograms from memory
    del train_padded_specs, test_padded_specs

    #initialize model
    latent_space_size = parameters['latent_space_size']
    slope_leaky = parameters['slope_leaky']
    learning_rate = parameters['learning_rate']
    precision_model = parameters['precision_model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = autoencoder_functions.VAE_KL_leaky(z_dim = latent_space_size, device_name='cuda',slope=slope_leaky,lr = learning_rate, model_precision =precision_model)
    model = model.to(device)

    #load the model
    model_file_name = path_to_model + '/model.pt'
    model.load_state_dict(torch.load(model_file_name))

    train_latent = model.get_latent(latent_train_loader)
    test_latent = model.get_latent(latent_test_loader)

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
    labels_nearest_neighbours = labels_train[indices]
    print("Labels nearest neighbours shape: ", labels_nearest_neighbours.shape)

    #determine the predicted label based on the nearest neighbours
    predicted_labels = np.zeros(test_latent.shape[0])
    for i in range(test_latent.shape[0]):
        #get the labels of the nearest neighbours
        labels_nearest_neighbours = labels_train[indices[i]]
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
    accuracy = np.sum(predicted_labels == labels_test) / len(labels_test)
    accuracy_WT = np.sum(predicted_labels[labels_test == 0] == labels_test[labels_test == 0]) / len(labels_test[labels_test == 0])
    accuracy_KO = np.sum(predicted_labels[labels_test == 1] == labels_test[labels_test == 1]) / len(labels_test[labels_test == 1])
    print("Overall accuracy: ", accuracy)
    #now permute the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy
    permuted_accuracies = np.zeros(n_permutations)
    permuted_accuracy_WT = np.zeros(n_permutations)
    permuted_accuracy_KO = np.zeros(n_permutations)
    for i in range(n_permutations):
        np.random.shuffle(labels_train)
         #determine the predicted label based on the nearest neighbours
        predicted_labels = np.zeros(test_latent.shape[0])
        for j in range(test_latent.shape[0]):
            #get the labels of the nearest neighbours
            labels_nearest_neighbours = labels_train[indices[j]]
            #get the unique labels and their counts
            unique_labels, counts = np.unique(labels_nearest_neighbours, return_counts=True)
            #calculate weighted counts
            weighted_counts = np.zeros(len(unique_labels))
            for k in range(len(unique_labels)):
                label = unique_labels[k]
                weighted_counts[k] = counts[k] * weights[label]
            #get the index of the label with the most counts
            predicted_labels[j] = unique_labels[np.argmax(weighted_counts)]   
        permuted_accuracies[i] = np.sum(predicted_labels == labels_test) / len(labels_test)
        permuted_accuracy_WT[i] = np.sum(predicted_labels[labels_test == 0] == labels_test[labels_test == 0]) / len(labels_test[labels_test == 0])
        permuted_accuracy_KO[i] = np.sum(predicted_labels[labels_test == 1] == labels_test[labels_test == 1]) / len(labels_test[labels_test == 1])
    print(permuted_accuracies)
    mean_permuted_accuracy = np.mean(permuted_accuracies)
    mean_permuted_accuracy_WT = np.mean(permuted_accuracy_WT)
    mean_permuted_accuracy_KO = np.mean(permuted_accuracy_KO)
    std_permuted_accuracy = np.std(permuted_accuracies)
    std_permuted_accuracy_WT = np.std(permuted_accuracy_WT)
    std_permuted_accuracy_KO = np.std(permuted_accuracy_KO)

    #calculate the p-values for every individual emitter and the p-value for the total accuracy
    count_higher = np.sum(permuted_accuracies >= accuracy)
    count_higher_WT = np.sum(permuted_accuracy_WT >= accuracy_WT)
    count_higher_KO = np.sum(permuted_accuracy_KO >= accuracy_KO)
    p_value = count_higher / n_permutations
    p_value_WT = count_higher_WT / n_permutations
    p_value_KO = count_higher_KO / n_permutations

     #create an indices list with len emitter
    labels2 = []
    for i in emitter:
        label2 = emitter_library[i]
        labels2.append(label2)
    
    print('Amount of vocalisations',len(labels2))

    indices2 = []
    labels_after_removal2 = []
    for i in range(len(labels2)):
        if labels2[i] == 0:
            indices2.append(i)
            labels_after_removal2.append(0)
        elif labels2[i] == 1:
            indices2.append(i)
            labels_after_removal2.append(1)

    labels2 = labels_after_removal2
    labels2 = np.array(labels2)
    indices2 = np.array(indices2)
    print('Amount of vocalisations after females removed:',len(indices2))
    print('Females removed:',len(emitter)-len(indices2))

    #remove features not in indices
    features_after_removal2 = []
    for i in indices2:
        features_after_removal2.append(features[i])

    features = features_after_removal2



    print('Number of instances in each class')
    print('WT:',np.sum(labels2==0))
    print('KO:',np.sum(labels2==1))

    #seperate the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels2, test_size=0.2, random_state=42)
    print('Number of instances in each class in training set')
    print('WT:',np.sum(y_train==0))
    print('KO:',np.sum(y_train==1))
    print('Total:',len(y_train))
    print('Number of features:',len(X_train[0]))
    print('Number of instances in each class in testing set')
    print('WT:',np.sum(y_test==0))
    print('KO:',np.sum(y_test==1))
    print('Total:',len(y_test))
    print('Number of features:',len(X_test[0]))
    weight_WT = 1/np.sum(y_train==0)
    weight_KO = 1/np.sum(y_train==1)
    print('Weight WT =',weight_WT)
    print('Weight KO =', weight_KO)
    weights2 = {0: weight_WT, 1: weight_KO}

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
    accuracy_WT2 = np.sum(predicted_labels2[labels_test2 == 0] == labels_test2[labels_test2 == 0]) / len(labels_test2[labels_test2 == 0])
    accuracy_KO2 = np.sum(predicted_labels2[labels_test2 == 1] == labels_test2[labels_test2 == 1]) / len(labels_test2[labels_test2 == 1])
    print("Overall accuracy: ", accuracy2)
    #now permute the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy
    permuted_accuracies2 = np.zeros(n_permutations)
    permuted_accuracy_WT2 = np.zeros(n_permutations)
    permuted_accuracy_KO2 = np.zeros(n_permutations)
    for i in range(n_permutations):
        np.random.shuffle(labels_train2)
         #determine the predicted label based on the nearest neighbours
        predicted_labels2 = np.zeros(test_latent2.shape[0])
        for j in range(test_latent2.shape[0]):
            #get the labels of the nearest neighbours
            labels_nearest_neighbours2 = labels_train2[indices2[j]]
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
        permuted_accuracy_WT2[i] = np.sum(predicted_labels2[labels_test2 == 0] == labels_test2[labels_test2 == 0]) / len(labels_test2[labels_test2 == 0])
        permuted_accuracy_KO2[i] = np.sum(predicted_labels2[labels_test2 == 1] == labels_test2[labels_test2 == 1]) / len(labels_test2[labels_test2 == 1])
    print(permuted_accuracies2)
    mean_permuted_accuracy2 = np.mean(permuted_accuracies2)
    mean_permuted_accuracy_WT2 = np.mean(permuted_accuracy_WT2)
    mean_permuted_accuracy_KO2 = np.mean(permuted_accuracy_KO2)
    std_permuted_accuracy2 = np.std(permuted_accuracies2)
    std_permuted_accuracy_WT2 = np.std(permuted_accuracy_WT2)
    std_permuted_accuracy_KO2 = np.std(permuted_accuracy_KO2)

    #count how many times the permuted accuracy is higher than the normal accuracy for p-value calculation
    count_higher2 = np.sum(permuted_accuracies2 >= accuracy2)
    count_higher_WT2 = np.sum(permuted_accuracy_WT2 >= accuracy_WT2)
    count_higher_KO2 = np.sum(permuted_accuracy_KO2 >= accuracy_KO2)
    p_value2 = count_higher2 / n_permutations
    p_value_WT2 = count_higher_WT2 / n_permutations
    p_value_KO2 = count_higher_KO2 / n_permutations


    #now plot the overall accuracy in the first subplot and the accuracy for each class in the second subplot
    plt.figure()
    plt.suptitle('Strain prediction for dataset', fontweight='bold')
    plt.subplot(2, 1, 1)
    labels = ['WT', 'KO', 'Total']
    normal_accuracies = [accuracy_WT, accuracy_KO, accuracy]
    permuted_accuracies = [mean_permuted_accuracy_WT, mean_permuted_accuracy_KO, mean_permuted_accuracy]
    permuted_std_errors = [std_permuted_accuracy_WT, std_permuted_accuracy_KO, std_permuted_accuracy]
    x_labels = ['Accuracy WT', 'Accuracy KO', 'Accuracy Total']
    plt.bar(x_labels, normal_accuracies, yerr=[0, 0,0], label='Accuracy')
    plt.bar(x_labels, permuted_accuracies, yerr=permuted_std_errors, alpha=0.75, label='Permuted Accuracy')
    if p_value_WT < 0.01:
        plt.text(0, normal_accuracies[0] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(0, normal_accuracies[0] + 0.05, 'p = {:.2f}'.format(p_value_WT), ha='center')
    if p_value_KO < 0.01:
        plt.text(1, normal_accuracies[1] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(1, normal_accuracies[1] + 0.05, 'p = {:.2f}'.format(p_value_KO), ha='center')
    if p_value < 0.01:
        plt.text(2, normal_accuracies[2] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(2, normal_accuracies[2] + 0.05, 'p = {:.2f}'.format(p_value), ha='center')

    plt.title('Prediction Accuracy based on latent features')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.text(-0.1, 1.0, 'A', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    plt.tight_layout() #adjust plot to fit legend.
    plt.ylim(0, 1)
    plt.xticks([0, 1,2], ['Accuracy WT', 'Accuracy KO', 'Accuracy Total'])
    plt.grid()
    plt.subplot(2, 1, 2)
    labels = ['WT', 'KO', 'Total']
    normal_accuracies2 = [accuracy_WT2, accuracy_KO2, accuracy2]
    permuted_accuracies2 = [mean_permuted_accuracy_WT2, mean_permuted_accuracy_KO2, mean_permuted_accuracy2]
    permuted_std_errors2 = [std_permuted_accuracy_WT2, std_permuted_accuracy_KO2, std_permuted_accuracy2]
    x_labels2 = ['Accuracy WT', 'Accuracy KO', 'Accuracy Total']
    plt.bar(x_labels2, normal_accuracies2, yerr=[0, 0,0], label='Accuracy')
    plt.bar(x_labels2, permuted_accuracies2, yerr=permuted_std_errors2, alpha=0.75, label='Permuted Accuracy')
    if p_value_WT2 < 0.01:
        plt.text(0, normal_accuracies2[0] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(0, normal_accuracies2[0] + 0.05, 'p = {:.2f}'.format(p_value_WT2), ha='center')
    if p_value_KO2 < 0.01:
        plt.text(1, normal_accuracies2[1] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(1, normal_accuracies2[1] + 0.05, 'p = {:.2f}'.format(p_value_KO2), ha='center')
    if p_value2 < 0.01:
        plt.text(2, normal_accuracies2[2] + 0.05, 'p < 0.01', ha='center')
    else:
        plt.text(2, normal_accuracies2[2] + 0.05, 'p = {:.2f}'.format(p_value2), ha='center')

    plt.title('Prediction Accuracy based on traditional features')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.ylim(0, 1)
    plt.xticks([0, 1,2], ['Accuracy WT', 'Accuracy KO', 'Accuracy Total'])
    plt.grid()
    plt.tight_layout() 
    plt.text(-0.1, 1.0, 'B', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    #print model used in the bottom left_corner of the figure
    plt.figtext(0.01, 0.01, 'Model_name: {}'.format(path_to_model),  fontsize=8, verticalalignment='bottom')
    plt.show()


def from_voc_to_latent_vis_strain(spec, emitter, emitter_library, path_to_model, combine_train_test = False):

        #load the trained parameters used in training
    parameters = joblib.load(path_to_model + '/parameters.pkl')

    ##load spec_indices for train and test set
    spec_indices_train = torch.load(path_to_model + '/spec_indices_train')
    spec_indices_test = torch.load(path_to_model + '/spec_indices_test')

    emitter_int = [emitter_library[e] for e in emitter]

    #remove doubles in train set  
    print('indices before removal of doubles =', len(spec_indices_train))
    spec_indices_train_after_removal = []
    for i in range(len(spec_indices_train)):
        if spec_indices_train[i] not in spec_indices_train_after_removal:
            spec_indices_train_after_removal.append(spec_indices_train[i])
    spec_indices_train = spec_indices_train_after_removal
    print('indices after removal of doubles =', len(spec_indices_train_after_removal))

    #get the spectograms and emitters
    train_specs = []
    test_specs = []
    labels_train = []
    labels_test = []

    for i in spec_indices_train:
        train_specs.append(spec[i])
        labels_train.append(emitter_int[i])
    for i in spec_indices_test:
        test_specs.append(spec[i])
        labels_test.append(emitter_int[i])

    #convert labels to integer numpy arrays
    labels_train = np.array(labels_train)
    labels_train = labels_train.astype(int)
    labels_test = np.array(labels_test)
    labels_test = labels_test.astype(int)

    #check if the same labels are present in labels_train and labels_test
    # Get the unique labels from each array
    unique_labels_train = np.unique(labels_train)
    unique_labels_test = np.unique(labels_test)

    # Check if the sets of unique labels are equal
    if np.array_equal(unique_labels_train, unique_labels_test):
        print("The same labels are present in both labels_train and labels_test.")
    else:
        print("The labels in labels_train and labels_test are different.")
        print("Unique labels in training set:", unique_labels_train)
        print("Unique labels in testing set:", unique_labels_test)

    if combine_train_test:
        test_specs = train_specs + test_specs
        labels_test = np.concatenate((labels_train, labels_test), axis=0)

    #print the size of test specs and labels test
    print('Total:',len(labels_test))
    print('Number of spectograms:',len(test_specs))

    #now pad and normalize the test specs
    padding_length = 160
    test_padded_specs = create_padded_spectograms.pad_spectograms_sorted(test_specs, padding_length)
    if parameters['max_value_per_spec'] == 'true':
        max_value_per_spec = True
        print(max_value_per_spec)
    test_padded_specs_normalized = normalize_padded_spectograms.normalize_specs(test_padded_specs, max_value_per_spec)

    #now choose a random spectogram from the test set and visualize it before and after padding and normalization
    while True:
        random_index = random.randint(0, len(test_specs) - 1)
        
        # Calculate the length of the item at the random index
        # (assuming test_specs[random_index] is the item you are checking)
        current_len = len(test_specs[random_index].T)
        
        # Check if the condition is met (length is 25 or greater)
        if current_len >= 25:
            # If the condition is met, exit the loop
            break
            
        # If the condition is NOT met (current_len < 25), the loop repeats
        
    print('random_index:', random_index)
    print(len(test_specs[random_index].T))
    random_index = 6195

    #get the spectral line image from the features_single python file for this random spectogram
    spectral_line_index = features_single.get_spectral_line_index(test_specs[random_index])
    spectral_line_index = cp.asnumpy(spectral_line_index)

    test_spec_that_will_be_used = test_specs[random_index]
    test_spec_that_will_be_used = np.array(test_spec_that_will_be_used)
    spectral_line_image = np.zeros(test_spec_that_will_be_used.shape)
    for i in range(len(spectral_line_index)):
        spectral_line_image[spectral_line_index[i],i] = 1
    
    # spectral_line_image = spectral_line_image * test_spec_that_will_be_used # use if you want to visualize the intensity of the line

    #get features for this spectogram
    features = features_single.get_usv_features(test_specs[random_index])
    #make a table with the features and their values
    feature_names = ['duration (s)', 'mean_freq (Hz)', 'min_freq (Hz)', 'max_freq (Hz)', 'bandwidth (Hz)', 'starting_freq (Hz)', 'stopping_freq (Hz)', 'directionality', 'coefficient of variation', 'normalized irregularity', 'local_variability', 'nr_of_steps_up', 'nr_of_steps_down', 'nr_of_peaks', 'nr_of_valleys']
    feature_values = [features[0], features[1], features[2], features[3], features[4], features[5], features[6], features[7], features[8], features[9], features[10], features[11], features[12], features[13], features[14]]
    #remove brackets around feature_values
    feature_values = [item[0] for item in feature_values]
    #make the last 4 features integers and round the rest of the values to 2 decimal places except the first to 4
    feature_values[0] = np.round(feature_values[0], 4)
    feature_values[1] = np.round(feature_values[1], 2)
    feature_values[2] = np.round(feature_values[2], 2)
    feature_values[3] = np.round(feature_values[3], 2)
    feature_values[4] = np.round(feature_values[4], 2)
    feature_values[5] = np.round(feature_values[5], 2)
    feature_values[6] = np.round(feature_values[6], 2)
    feature_values[7] = np.round(feature_values[7], 2)
    feature_values[8] = np.round(feature_values[8], 2)
    feature_values[9] = np.round(feature_values[9], 2)
    feature_values[10] = np.round(feature_values[10], 2)
    feature_values[11] = int(feature_values[11])
    feature_values[12] = int(feature_values[12])
    feature_values[13] = int(feature_values[13])
    feature_values[14] = int(feature_values[14])
    table_data = [[name, value] for name, value in zip(feature_names, feature_values)]
    column_titles = ['Feature', 'Value']

    #get latent representation of the test spectograms
    test_padded_specs_tensor = torch.utils.data.DataLoader(test_padded_specs_normalized, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_space_size = parameters['latent_space_size']
    slope_leaky = parameters['slope_leaky']
    learning_rate = parameters['learning_rate']
    precision_model = parameters['precision_model']
    model = autoencoder_functions.VAE_KL_leaky(z_dim = latent_space_size, device_name='cuda',slope=slope_leaky,lr = learning_rate, model_precision =precision_model)
    model = model.to(device)
    model_file_name = path_to_model + '/model.pt'
    model.load_state_dict(torch.load(model_file_name))
    test_latent = model.get_latent(test_padded_specs_tensor)
    test_latent = cp.asnumpy(test_latent)

    #get the latent space of the random spectogram
    random_spectrogram_latent = test_latent[random_index]
    random_spectrogram_latent = random_spectrogram_latent.reshape(1, -1)  # Reshape to 2D array with one row

    #perform umap on the latent space to reduce it to 2 dimensions for visualization
    reducer = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 2)
    reducer.fit(test_latent)

    latent_2d = reducer.transform(test_latent)

    #make a a 100x100 grid around all points
    num_bins = 50
    # This is the correct way
    x_min, x_max = latent_2d[:, 0].min(), latent_2d[:, 0].max()
    y_min, y_max = latent_2d[:, 1].min(), latent_2d[:, 1].max()
    # 1. Calculate the range of each dimension
    x_range = x_max - x_min
    y_range = y_max - y_min

    # 2. Find the largest range
    max_range = max(x_range, y_range)

    # 3. Calculate the center of each dimension
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    # 4. Set the new, equal limits for both axes
    x_lim_min = x_center - max_range / 2
    x_lim_max = x_center + max_range / 2
    y_lim_min = y_center - max_range / 2
    y_lim_max = y_center + max_range / 2


    #give each emitter a color and make the random spectogram stand out with a very distinct color
    unique_emitters, counts = np.unique(labels_test, return_counts= True)

 
    points_label1 = latent_2d[labels_test == unique_emitters[0]]
    counts1, _, _ = np.histogram2d(points_label1[:,0], points_label1[:,1], bins = num_bins, range= [[x_min,x_max],[y_min,y_max]])
    points_label2 = latent_2d[labels_test == unique_emitters[1]]
    counts2, _, _ = np.histogram2d(points_label2[:,0], points_label2[:,1], bins = num_bins, range= [[x_min,x_max],[y_min,y_max]])
    weights1 = 1/counts[0]
    weights2 = 1/counts[1]
    counts1 = counts1 * weights1
    counts2 = counts2 * weights2
    density_difference = counts1 - counts2

    sigma = 1.5
    density_difference_smoothed = gaussian_filter(density_difference, sigma = sigma)



    colors = plt.cm.get_cmap('tab20', len(unique_emitters))


    emitter_colors = {emitter: colors(i) for i, emitter in enumerate(unique_emitters)}
    #make sure the specific names in emitter _library are in the legend
    emitter_library_reversed = {v: k for k, v in emitter_library.items()}

    emitter_library_reversed = emitter_int
    # 0 is WT and 1 is KO
    emitter_colors = {0: 'blue', 1: 'red'}
    unique_emitters = [0, 1]
    nan_indices =[]

    #make a umap representation of the quantitative features of the test spectograms
    q_features = []
    for i in range(len(test_specs)):
        features_q = features_single.get_usv_features(test_specs[i])
        features_q = features_q.get()

        if np.isnan(features_q).any():
            nan_indices.append(i)

        q_features.append(features_q)
    
    print(nan_indices)
    q_features = np.stack(q_features)
    q_features = np.squeeze(q_features)

    print(q_features.shape)

    stds = np.std(q_features, axis=0)
    zero_std_cols = np.where(stds == 0)[0]
    print(f"!!! Zero standard deviation detected in columns: {zero_std_cols} !!!")

    scaler = StandardScaler()
    q_features = scaler.fit_transform(q_features)

    #perform umap on the latent space to reduce it to 2 dimensions for visualization
    reducer2 = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 2)
    reducer2.fit(q_features)

    latent_2d2 = reducer2.transform(q_features)
    print(latent_2d2.shape)
    print(latent_2d2)

    if np.isnan(latent_2d2).any():
        print("!!! NaN detected in final UMAP output. This is highly unusual if input was clean. !!!")
    #make a a 100x100 grid around all points
    num_bins = 50
    # This is the correct way
    x_min2, x_max2 = latent_2d2[:, 0].min(), latent_2d2[:, 0].max()
    y_min2, y_max2 = latent_2d2[:, 1].min(), latent_2d2[:, 1].max()
    print(x_min2, y_min2)
    print(x_max2, y_max2)
    # 1. Calculate the range of each dimension
    x_range2 = x_max2 - x_min2
    y_range2 = y_max2 - y_min2

    # 2. Find the largest range
    max_range2 = max(x_range2, y_range2)

    # 3. Calculate the center of each dimension
    x_center2 = (x_max2 + x_min2) / 2
    y_center2 = (y_max2 + y_min2) / 2

    # 4. Set the new, equal limits for both axes
    x_lim_min2 = x_center2 - max_range2 / 2
    x_lim_max2 = x_center2 + max_range2 / 2
    y_lim_min2 = y_center2 - max_range2 / 2
    y_lim_max2 = y_center2 + max_range2 / 2


    #give each emitter a color and make the random spectogram stand out with a very distinct color
    unique_emitters, counts = np.unique(labels_test, return_counts= True)

 
    points2_label1 = latent_2d2[labels_test == unique_emitters[0]]
    counts2_1, _, _ = np.histogram2d(points2_label1[:,0], points2_label1[:,1], bins = num_bins, range= [[x_min2,x_max2],[y_min2,y_max2]])
    points2_label2 = latent_2d2[labels_test == unique_emitters[1]]
    counts2_2, _, _ = np.histogram2d(points2_label2[:,0], points2_label2[:,1], bins = num_bins, range= [[x_min2,x_max2],[y_min2,y_max2]])
    weights2_1 = 1/counts[0]
    weights2_2 = 1/counts[1]
    counts2_1 = counts2_1 * weights2_1
    counts2_2 = counts2_2 * weights2_2
    density_difference2 = counts2_1 - counts2_2

    sigma = 1.5
    density_difference_smoothed2 = gaussian_filter(density_difference2, sigma = sigma)
    
  

    fig = plt.figure(figsize=(15,15))
    fig.suptitle('Feature Extraction Pipeline', fontsize=20, fontweight='bold', y = 1.01)
    gs = GridSpec(9, 9, figure=fig)
    # subplot 1 original spectrogram
    ax1 = fig.add_subplot(gs[0:3, 0:3])
    im1 = ax1.imshow(test_specs[random_index], origin='lower')
    ax1.set_title('Original Spectrogram')
    cbar = fig.colorbar(im1, ax=ax1, shrink=0.5)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    cbar.ax.yaxis.set_major_formatter(formatter) 
    offset_text = cbar.ax.get_yaxis().get_offset_text()
    offset_text.set_x(2.5)
    ax1.set_xlabel('Time (bins)')
    ax1.set_ylabel('Frequency (bins)')
    ax1.set_aspect('auto')
    ax1.text(-0.18, 1.1, 'A', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    # subplot 2 padded spectrogram
    ax2 = fig.add_subplot(gs[0:3, 3:6])
    im2 = ax2.imshow(test_padded_specs[random_index], origin='lower')
    cbar = fig.colorbar(im2, ax=ax2, shrink=0.5)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    cbar.ax.yaxis.set_major_formatter(formatter) 
    offset_text = cbar.ax.get_yaxis().get_offset_text()
    offset_text.set_x(2.5)
    ax2.set_title('Padded Spectrogram')
    ax2.set_xlabel('Time (bins)')
    ax2.set_ylabel('Frequency (bins)')
    ax2.set_aspect('auto')
    ax2.text(-0.18, 1.1, 'B', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    # Add an arrow between ax1 and ax2
    ax1_pos = ax1.get_position()
    ax2_pos = ax2.get_position()
    plt.annotate('', xy=(ax2_pos.x0 -0.035, ax2_pos.y0 + ax2_pos.height / 2 -0.04), xytext=(ax1_pos.x1 -0.025, ax1_pos.y0 + ax1_pos.height / 2 -0.04),
        arrowprops=dict(facecolor='black', width=2, headwidth=8, headlength=8, shrink=0.05), xycoords='figure fraction', textcoords='figure fraction', annotation_clip= False)
    # subplot 3 normalized spectrogram
    ax3 = fig.add_subplot(gs[0:3, 6:9])
    im3 = ax3.imshow(test_padded_specs_normalized[random_index], origin='lower')
    ax3.set_title('Normalized Spectrogram')
    cbar = fig.colorbar(im3, ax=ax3, shrink=0.5)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    offset_text = cbar.ax.get_yaxis().get_offset_text()
    offset_text.set_x(2.5)
    cbar.ax.yaxis.set_major_formatter(formatter) 
    ax3.set_xlabel('Time (bins)')
    ax3.set_ylabel('Frequency (bins)')
    ax3.set_aspect('auto')
    ax3.text(-0.18, 1.1, 'C', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    # Add an arrow between ax2 and ax3
    ax2_pos = ax2.get_position()
    ax3_pos = ax3.get_position()
    plt.annotate('', xy=(ax3_pos.x0 +0.015, ax3_pos.y0 + ax3_pos.height / 2 -0.04), xytext=(ax2_pos.x1 +0.025, ax2_pos.y0 + ax2_pos.height / 2 -0.04),
        arrowprops=dict(facecolor='black', width=2, headwidth=8, headlength=8, shrink=0.05), xycoords='figure fraction', textcoords='figure fraction', annotation_clip= False)
    # subplot 4 spectral line image
    ax4 = fig.add_subplot(gs[3:6, 0:3])
    im4 = ax4.imshow(spectral_line_image, origin='lower')
    ax4.set_title('Spectral Line Image')
    cbar = fig.colorbar(im4, ax=ax4, shrink=0.5)  
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    cbar.ax.yaxis.set_major_formatter(formatter) 
    offset_text = cbar.ax.get_yaxis().get_offset_text()
    offset_text.set_x(2.5)
    # ylabel zero bin and 128 bins that go up to 100000 with steps of 781.25
    ax4.set_yticks([0, 32, 64, 96, 128])
    ax4.set_yticklabels(['0', '25000', '50000', '75000', '100000'])
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_aspect('auto')
    ax4.text(-0.18, 1.1, 'D', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    #xticks with each bin being 227.559 ms, so givve xticks every 1000/227.559 = 4.39 bins for the duration, feature 0
    num_bins = test_specs[random_index].shape[1]
    xtick_positions = np.arange(0, num_bins, 10/(227.559/200)) #every 10 ms a bin
    #get length of xtick_position and for every length add a label for 1000 ms
    xtick_labels = [str(int(i * 10)) for i in range(len(xtick_positions))]
    ax4.set_xticks(xtick_positions)
    ax4.set_xticklabels(xtick_labels)
    ax4.set_xlabel('Time (ms)')
    #Add an arrow from ax1 to ax4
    ax1_pos = ax1.get_position()
    ax4_pos = ax4.get_position()
    plt.annotate('', xy=(ax4_pos.x0 + ax4_pos.width / 2, ax4_pos.y1 + 0.005), xytext=(ax1_pos.x0 + ax1_pos.width / 2, ax1_pos.y0 + 0.03),
        arrowprops=dict(facecolor='black', width=2, headwidth=8, headlength=8, shrink=0.05), xycoords='figure fraction', textcoords='figure fraction', annotation_clip= False)
    # subplot 5 and 6 latent representation visualize array as a 1,24 array with a colorbar spanning 2 subplots
    ax5 = fig.add_subplot(gs[3:6, 3:9])
    im5 = ax5.imshow(random_spectrogram_latent, cmap='viridis')
    ax5.set_title('Latent Representation')
    fig.colorbar(im5, ax=ax5, orientation='vertical', shrink = 0.5)
    ax5.set_yticks([0])
    # ax5.set_yticklabels(['Latent Space'])
    ax5.set_xlabel('Latent Dimensions')
    ax5.text(-0.04, 3, 'E', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    #Add arrow from 4 to 5
    plt.annotate(
    '',
    xy=(0.8, 0.52), # Head of the arrow (x=0.25, y=0.45)
    xytext=(0.8, 0.64), # Tail of the arrow (x=0.25, y=0.65)
    arrowprops=dict(
        facecolor='black',
        width=2,
        headwidth=8,
        headlength=8,
        shrink=0.05
    ),
    xycoords='figure fraction',
    textcoords='figure fraction',
    annotation_clip=False
)
    # ax5.set_aspect('auto')

    # subplot 7 table with features and their values
    ax7 = fig.add_subplot(gs[6:9, 0:3])
    ax7.axis('off')
    ax7.table(cellText=table_data, colLabels=column_titles, loc='center')
    ax7.set_aspect('auto')
    ax7.text(-0.15, 1.0, 'F', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    # ax7.set_title('Extracted Features')
    ax7.text(0.5, 0.9, 'Extracted Features', fontsize = 12, ha = 'center', va = 'center', transform=ax7.transAxes)
    #now add text beneath the table to show the emitter of this spectogram and the model used
    emitter_name = 'WT' if labels_test[random_index] == 0 else 'KO'
    ax7.text(0, 0, 'Emitter: {}'.format(emitter_name), fontsize = 10,  transform=ax7.transAxes)
    model_name = path_to_model.split('/')[-1]
    ax7.text(0, -0.05, 'Model: {}'.format(model_name), fontsize = 10,  transform=ax7.transAxes)
    # Add arrow from 4 to 7
    plt.annotate(
    '',
    xy=(0.225, 0.28), # Head of the arrow (x=0.25, y=0.45)
    xytext=(0.225, 0.34), # Tail of the arrow (x=0.25, y=0.65)
    arrowprops=dict(
        facecolor='black',
        width=2,
        headwidth=8,
        headlength=8,
        shrink=0.05
    ),
    xycoords='figure fraction',
    textcoords='figure fraction',
    annotation_clip=False
)
    # subplot 8 and 9 umap visualization of the latent space with the random spect
    ax8 = fig.add_subplot(gs[6:9, 6:9])
    vmax = max(abs(density_difference_smoothed.min()), abs(density_difference_smoothed.max()))
    original_cmap = plt.cm.get_cmap('RdBu_r')

    # Get the full list of colors from the original colormap's lookup table
    colors = original_cmap(np.linspace(0, 1, original_cmap.N))

    # Find the center of the color list (the pure white color)
    # The N // 2 trick handles both even and odd length colormaps
    center_index = original_cmap.N // 2

    # Set the middle color to pure white (RGB value is [1, 1, 1, 1])
    colors[center_index] = np.array([1, 1, 1, 1])

    # Create the new colormap from the modified color list
    my_cmap = LinearSegmentedColormap.from_list("my_diverging_cmap", colors)
    im8 = ax8.imshow(density_difference_smoothed.T, extent= [x_min, x_max, y_min, y_max],origin='lower', cmap = my_cmap, vmin = -vmax, vmax = vmax)
    # Apply the new limits to your plot if we want a square
    # ax8.set_xlim(x_lim_min, x_lim_max)
    # ax8.set_ylim(y_lim_min, y_lim_max)
    ax8.set_facecolor('white')
    ax8.set_title('Latent Feature Density Difference (UMAP)', fontsize = 11)
    ax8.set_xlabel('UMAP Dimension 1')
    ax8.set_ylabel('UMAP Dimension 2')
    ax8.set_aspect('auto')
    cbar = fig.colorbar(im8, ax=ax8, shrink = 0.5)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    cbar.ax.yaxis.set_major_formatter(formatter) 
    offset_text = cbar.ax.get_yaxis().get_offset_text()
    offset_text.set_x(2.5)
    cbar.set_label(f'Higher Density of WT <--  --> Higher Density of KO', fontsize = 8)
    ax8.text(-0.055, 1.05, 'H', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)

    ax9 = fig.add_subplot(gs[6:9, 3:6])
    vmax = max(abs(density_difference_smoothed2.min()), abs(density_difference_smoothed2.max()))

    im9 = ax9.imshow(density_difference_smoothed2.T, extent= [x_min2, x_max2, y_min2, y_max2],origin='lower', cmap = my_cmap, vmin = -vmax, vmax = vmax)
    # Apply the new limits to your plot if we want a square
    # ax8.set_xlim(x_lim_min, x_lim_max)
    # ax8.set_ylim(y_lim_min, y_lim_max)
    ax9.set_facecolor('white')
    ax9.set_title('Traditional Feature Density Difference (UMAP)', fontsize = 11)
    ax9.set_xlabel('UMAP Dimension 1')
    ax9.set_ylabel('UMAP Dimension 2')
    ax9.set_aspect('auto')
    cbar = fig.colorbar(im9, ax=ax9, shrink = 0.5)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    cbar.ax.yaxis.set_major_formatter(formatter) 
    offset_text = cbar.ax.get_yaxis().get_offset_text()
    offset_text.set_x(2.5)
    cbar.set_label(f'Higher Density of WT <--  --> Higher Density of KO', fontsize = 8)
    ax9.text(-0.055, 1.05, 'G', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)

    # Add arrow from 5 to 8
    plt.annotate(
    '',
    xy=(0.8, 0.35), # Head of the arrow (x=0.25, y=0.45)
    xytext=(0.8, 0.47), # Tail of the arrow (x=0.25, y=0.65)
    arrowprops=dict(
        facecolor='black',
        width=2,
        headwidth=8,
        headlength=8,
        shrink=0.05
    ),
    xycoords='figure fraction',
    textcoords='figure fraction',
    annotation_clip=False)

    # 1. Start Point (Tail)
    START_X = 0.25
    START_Y = 0.07
    xytext_point = (START_X, START_Y)

    # 2. End Point (Head)
    # To control length:
    # - The arrow will go DOWN by (START_Y - END_Y) amount.
    # - The arrow will go RIGHT by (END_X - START_X) amount.
    END_X = 0.35  # Move further right than START_X (0.55) -> Longer horizontal part
    END_Y = 0.05  # Move further down than START_Y (0.60) -> Longer vertical part
    xy_point = (END_X, END_Y)

    # --- Plot the Angled Arrow ---
    plt.annotate(
        '',  # No text needed, just the arrow
        xy=xy_point,
        xytext=xytext_point,
        arrowprops=dict(
            arrowstyle="-|>",
            # Use angle style for a 90-degree hook:
            # angleA=270: Start pointing down (-90 degrees)
            # angleB=0: End pointing right (0 degrees)
            # rad=0: Makes the corner a sharp 90-degree hook (no rounding)
            connectionstyle="angle,angleA=270,angleB=0,rad=0",
            facecolor='black',
            lw=2,
            # Shrinkage (gap from start/end points to arrow line) in points
            shrinkA=5,
            shrinkB=5,
            # Controls the size/thickness of the line and head
            mutation_scale=25
        ),
        # Set both to figure fraction for free movement across the figure
        xycoords='figure fraction',
        textcoords='figure fraction',
        annotation_clip=False
    )
    # plt.tight_layout()
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.95,
        wspace=0.5, # Increased horizontal space
        hspace=1  # Increased vertical space
    )
    plt.show()

def emitter_prediction_model_weighted_and_hardcoded_pups(spec,features, emitter, emitter_library, path_to_model, n_neighbours = 30, n_permutations =100):

        #load the trained parameters used in training
    parameters = joblib.load(path_to_model + '/parameters.pkl')

    ##load spec_indices for train and test set
    spec_indices_train = torch.load(path_to_model + '/spec_indices_train')
    spec_indices_test = torch.load(path_to_model + '/spec_indices_test')

    emitter_int = [emitter_library[e] for e in emitter]

    #remove doubles in train set  
    print('indices before removal of doubles =', len(spec_indices_train))
    spec_indices_train_after_removal = []
    for i in range(len(spec_indices_train)):
        if spec_indices_train[i] not in spec_indices_train_after_removal:
            spec_indices_train_after_removal.append(spec_indices_train[i])
    spec_indices_train = spec_indices_train_after_removal
    print('indices after removal of doubles =', len(spec_indices_train_after_removal))

    #get the spectograms and emitters
    train_specs = []
    test_specs = []
    labels_train = []
    labels_test = []

    for i in spec_indices_train:
        train_specs.append(spec[i])
        labels_train.append(emitter_int[i])
    for i in spec_indices_test:
        test_specs.append(spec[i])
        labels_test.append(emitter_int[i])

    #convert labels to integer numpy arrays
    labels_train = np.array(labels_train)
    labels_train = labels_train.astype(int)
    labels_test = np.array(labels_test)
    labels_test = labels_test.astype(int)

    #check if the same labels are present in labels_train and labels_test
    # Get the unique labels from each array
    unique_labels_train = np.unique(labels_train)
    unique_labels_test = np.unique(labels_test)

    # Check if the sets of unique labels are equal
    if np.array_equal(unique_labels_train, unique_labels_test):
        print("The same labels are present in both labels_train and labels_test.")
    else:
        print("The labels in labels_train and labels_test are different.")
        print("Unique labels in training set:", unique_labels_train)
        print("Unique labels in testing set:", unique_labels_test)

    #make a weight list for the integers in labels train
    unique_labels, counts = np.unique(labels_train, return_counts=True)
    weights = {}
    for label, count in zip(unique_labels, counts):
        weights[label] = 1 / count

    print(weights)

    #check if there are as many labels as spectograms in the train and test sets
    assert len(train_specs) == len(labels_train), "Number of train spectograms and labels do not match"
    assert len(test_specs) == len(labels_test), "Number of test spectograms and labels do not match"

    #pad the spectograms according to our training procedure
    padding_length = 160
    train_padded_specs = create_padded_spectograms.pad_spectograms_sorted(train_specs, padding_length)
    test_padded_specs = create_padded_spectograms.pad_spectograms_sorted(test_specs, padding_length)

    #delete unnecesarry big files in RAM
    del spec, train_specs, test_specs

    #normalize according to our training
    if parameters['max_value_per_spec'] == 'true':
        max_value_per_spec = True
        print(max_value_per_spec)
    train_padded_specs = normalize_padded_spectograms.normalize_specs(train_padded_specs, max_value_per_spec)
    test_padded_specs = normalize_padded_spectograms.normalize_specs(test_padded_specs, max_value_per_spec)

    #convert to torch tensors
    latent_train_loader = torch.utils.data.DataLoader(train_padded_specs, batch_size=1, shuffle=False)
    latent_test_loader = torch.utils.data.DataLoader(test_padded_specs, batch_size=1, shuffle=False)

    #delete the padded spectograms from memory
    del train_padded_specs, test_padded_specs

    #initialize model
    latent_space_size = parameters['latent_space_size']
    slope_leaky = parameters['slope_leaky']
    learning_rate = parameters['learning_rate']
    precision_model = parameters['precision_model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = autoencoder_functions.VAE_KL_leaky(z_dim = latent_space_size, device_name='cuda',slope=slope_leaky,lr = learning_rate, model_precision =precision_model)
    model = model.to(device)

    #load the model
    model_file_name = path_to_model + '/model.pt'
    model.load_state_dict(torch.load(model_file_name))

    train_latent = model.get_latent(latent_train_loader)
    test_latent = model.get_latent(latent_test_loader)

    # X_train = train_latent
    # X_test = test_latent
    # # #perform normalization on the train data and transform it to the test data
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # train_latent = X_train
    # test_latent = X_test


    # # #perform umap on train_latent and fit to test latent
    # reducer = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 2)
    # reducer.fit(train_latent)
    # train_latent = reducer.transform(train_latent)
    # test_latent = reducer.transform(test_latent)

    print('train latent shape:',train_latent.shape)
    print('test latent shape:',test_latent.shape) 
  
    #get the nearest neighbours of the test set in the train set
    neigh = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
    neigh.fit(train_latent)
    distances, indices = neigh.kneighbors(test_latent)
    print("Indices shape: ", indices.shape)
    print("Distances shape: ", distances.shape)

    #get the labels of the nearest neighbours
    labels_nearest_neighbours = labels_train[indices]
    print("Labels nearest neighbours shape: ", labels_nearest_neighbours.shape)

    #determine the predicted label based on the nearest neighbours
    predicted_labels = np.zeros(test_latent.shape[0])
    for i in range(test_latent.shape[0]):
        #get the labels of the nearest neighbours
        labels_nearest_neighbours = labels_train[indices[i]]
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
    accuracy = np.sum(predicted_labels == labels_test) / len(labels_test)

    # Get a complete list of all unique labels from both train and test sets
    all_labels = np.unique(np.concatenate((labels_train, labels_test), axis=0))

    # Initialize accuracy dictionary for all possible labels
    accuracy_per_class = {label: 0.0 for label in all_labels}
    
    # Calculate accuracy for each class that exists in the labels_test
    unique_labels_present = np.unique(labels_test)
    for label in unique_labels_present:
        accuracy_per_class[label] = np.sum(predicted_labels[labels_test == label] == labels_test[labels_test == label]) / len(labels_test[labels_test == label])

    print("Overall accuracy: ", accuracy)
    print("Accuracy per class: ", accuracy_per_class)

    #now permut the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy for the
    #overall permuted accurcacy and the permuted accuracy for each class
    permuted_accuracies = np.zeros(n_permutations)
    permuted_accuracies_per_class = {label: np.zeros(n_permutations) for label in all_labels}
    for i in range(n_permutations):
        np.random.shuffle(labels_train)

        permuted_predicted_labels = np.zeros(test_latent.shape[0])
        for j in range(test_latent.shape[0]):
            #get the labels of the nearest neighbours
            labels_nearest_neighbours = labels_train[indices[j]]
            #get the unique labels and their counts
            unique_labels_nn, counts = np.unique(labels_nearest_neighbours, return_counts=True)
            #calculate weighted counts
            weighted_counts = np.zeros(len(unique_labels_nn))
            for k in range(len(unique_labels_nn)):
                label = unique_labels_nn[k]
                weighted_counts[k] = counts[k] * weights[label]
            #get the index of the label with the most counts
            permuted_predicted_labels[j] = unique_labels_nn[np.argmax(weighted_counts)]
        
        permuted_accuracies[i] = np.sum(permuted_predicted_labels == labels_test) / len(labels_test)

        # Calculate and store permuted accuracy for each class that exists in this permutation
        unique_labels_present_in_perm = np.unique(labels_test)
        for label in unique_labels_present_in_perm:
                permuted_accuracies_per_class[label][i] = np.sum(permuted_predicted_labels[labels_test == label] == labels_test[labels_test == label]) / len(labels_test[labels_test == label])

    mean_permuted_accuracies_per_class = {label: np.mean(permuted_accuracies_per_class[label]) for label in all_labels}
    std_permuted_accuracies_per_class = {label: np.std(permuted_accuracies_per_class[label]) for label in all_labels}
    
    mean_permuted_accuracy = np.mean(permuted_accuracies)
    std_permuted_accuracy = np.std(permuted_accuracies)

    #calculate the p-values for every individual emitter per class and the p-value for the total accuracy
    count_higher_per_class = {label: np.sum(permuted_accuracies_per_class[label] >= accuracy_per_class[label]) for label in all_labels}
    count_higher = np.sum(permuted_accuracies >= accuracy)
    p_value_per_class = {label: count_higher_per_class[label] / n_permutations for label in all_labels}
    p_value = count_higher / n_permutations 
    
    print("Mean permuted accuracy: ", mean_permuted_accuracy)
    print("Std permuted accuracy: ", std_permuted_accuracy)
    print("Mean permuted accuracies per class: ", mean_permuted_accuracies_per_class)
    print("Std permuted accuracies per class: ", std_permuted_accuracies_per_class)


    #create an indices list with len emitter
    labels2 = []
    for i in emitter:
        label2 = emitter_library[i]
        labels2.append(label2)
    print(np.unique(labels2, return_counts=True))

    print('Amount of vocalisations',len(labels2))

    indices2 = []
    labels_after_removal2 = []
    #remove vocalisations from either starting with capital F or starting with mother or starting with pupx, when checking emitter library
    for i, emit in enumerate(emitter):
        if not emit.startswith('F') and not emit.startswith('mother') and not emit.startswith('pupx'):
            indices2.append(i)
            labels_after_removal2.append(labels2[i])


    labels2 = labels_after_removal2
    labels2 = np.array(labels2)
    indices2 = np.array(indices2)
    print('Amount of vocalisations after females removed:',len(indices2))
    print('Females removed:',len(emitter)-len(indices2))

    #remove features not in indices
    features_after_removal2 = []
    for i in indices2:
        features_after_removal2.append(features[i])

    features2 = features_after_removal2

    print('labels after removal =',np.unique(labels2,return_counts=True))
    #seperate the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features2, labels2, test_size=0.2, random_state=42)
    print('Total:',len(y_train))
    print('Number of features:',len(X_train[0]))
    print('Total:',len(y_test))
    print('Number of features:',len(X_test[0]))

    #perform normalization on the train data and transform it to the test data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_latent2 = X_train
    test_latent2 = X_test
    labels_train2 = y_train
    labels_test2 = y_test

    #make a weight list for the integers in labels train
    unique_labels2, counts2 = np.unique(labels_train2, return_counts=True)
    weights2 = {}
    for label, count in zip(unique_labels2, counts2):
        weights2[label] = 1 / count

    print('weights =',weights2)

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
        unique_labels, counts = np.unique(labels_nearest_neighbours2, return_counts=True)
        #calculate wieghted counts
        weighted_counts = np.zeros(len(unique_labels))
        for j in range(len(unique_labels)):
            label2 = unique_labels[j]
            weighted_counts[j] = counts[j] * weights2[label2]
        #get the index of the label with the most counts
        predicted_labels2[i] = unique_labels[np.argmax(weighted_counts)]      

    #determine the accuracy of the total prediction and the accuracy of the prediction for each class
    accuracy2 = np.sum(predicted_labels2 == labels_test2) / len(labels_test2)

    # Get a complete list of all unique labels from both train and test sets
    all_labels2 = np.unique(np.concatenate((labels_train2, labels_test2), axis=0))

    # Initialize accuracy dictionary for all possible labels
    accuracy_per_class2 = {label2: 0.0 for label2 in all_labels2}
    
    # Calculate accuracy for each class that exists in the labels_test
    unique_labels_present2 = np.unique(labels_test2)
    for label2 in unique_labels_present2:
        accuracy_per_class2[label2] = np.sum(predicted_labels2[labels_test2 == label2] == labels_test2[labels_test2 == label2]) / len(labels_test2[labels_test2 == label2])

    print("Overall accuracy: ", accuracy2)
    print("Accuracy per class: ", accuracy_per_class2)

    #now permut the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy for the
    #overall permuted accurcacy and the permuted accuracy for each class
    permuted_accuracies2 = np.zeros(n_permutations)
    permuted_accuracies_per_class2 = {label2: np.zeros(n_permutations) for label2 in all_labels2}
    for i in range(n_permutations):
        np.random.shuffle(labels_train2)

        permuted_predicted_labels2 = np.zeros(test_latent2.shape[0])
        for j in range(test_latent2.shape[0]):
            #get the labels of the nearest neighbours
            labels_nearest_neighbours2 = labels_train2[indices2[j]]
            #get the unique labels and their counts
            unique_labels_nn, counts = np.unique(labels_nearest_neighbours2, return_counts=True)
            #calculate weighted counts
            weighted_counts = np.zeros(len(unique_labels_nn))
            for k in range(len(unique_labels_nn)):
                label = unique_labels_nn[k]
                weighted_counts[k] = counts[k] * weights2[label]
            #get the index of the label with the most counts
            permuted_predicted_labels2[j] = unique_labels_nn[np.argmax(weighted_counts)]
        
        permuted_accuracies2[i] = np.sum(permuted_predicted_labels2 == labels_test2) / len(labels_test2)

        # Calculate and store permuted accuracy for each class that exists in this permutation
        unique_labels_present_in_perm2 = np.unique(labels_test2)
        for label2 in unique_labels_present_in_perm2:
                permuted_accuracies_per_class2[label2][i] = np.sum(permuted_predicted_labels2[labels_test2 == label2] == labels_test2[labels_test2 == label2]) / len(labels_test2[labels_test2 == label2])

    mean_permuted_accuracies_per_class2 = {label2: np.mean(permuted_accuracies_per_class2[label2]) for label2 in all_labels2}
    std_permuted_accuracies_per_class2 = {label2: np.std(permuted_accuracies_per_class2[label2]) for label2 in all_labels2}
    
    mean_permuted_accuracy2 = np.mean(permuted_accuracies2)
    std_permuted_accuracy2 = np.std(permuted_accuracies2)
    
    #calculate the p-values for every individual emitter per class and the p-value for the total accuracy
    count_higher_per_class2 = {label2: np.sum(permuted_accuracies_per_class2[label2] >= accuracy_per_class2[label2]) for label2 in all_labels2}
    count_higher2 = np.sum(permuted_accuracies2 >= accuracy2)
    p_value_per_class2 = {label2: count_higher_per_class2[label2] / n_permutations for label2 in all_labels2}
    p_value2 = count_higher2 / n_permutations

    print("Mean permuted accuracy: ", mean_permuted_accuracy2)
    print("Std permuted accuracy: ", std_permuted_accuracy2)
    print("Mean permuted accuracies per class: ", mean_permuted_accuracies_per_class2)
    print("Std permuted accuracies per class: ", std_permuted_accuracies_per_class2)
    
    all_labels = list(all_labels)
    all_labels = all_labels[0:6] + all_labels[8:] + all_labels[6:8]
    all_labels2 = list(all_labels2)
    all_labels2 = all_labels2[0:6] + all_labels2[8:] + all_labels2[6:8]
   #now plot the accurcay per class and permuted accuracy per class in subplot 1 and the overall accuracy and permuted accuracy in subplot 2
    plt.figure(figsize=(12, 10)) # It's a good practice to set a figure size for better readability
    plt.suptitle('Emitter prediction for dataset', fontweight='bold')

    plt.subplot(2, 1, 1)
    normal_accuracies = [accuracy_per_class[label] for label in all_labels]
    permuted_accuracies = [mean_permuted_accuracies_per_class[label] for label in all_labels]
    permuted_std_errors = [std_permuted_accuracies_per_class[label] for label in all_labels]
    x_labels = [str(label) for label in all_labels]
    p_value_per_class = [p_value_per_class[label] for label in all_labels]

    # Add the overall accuracy as an extra bar
    # Corrected: Use append without re-assignment
    normal_accuracies.append(accuracy)
    permuted_accuracies.append(mean_permuted_accuracy)
    permuted_std_errors.append(std_permuted_accuracy)
    x_labels.append('Overall accuracy')

    plt.bar(x_labels, normal_accuracies, yerr=[0]*len(normal_accuracies), label='Accuracy')
    plt.bar(x_labels, permuted_accuracies, yerr=permuted_std_errors, alpha=0.9, label='Permuted Accuracy')

    # For every bar in normal_accuracies, add a text with the p-value above the bar
    for i in range(len(p_value_per_class)):
        if p_value_per_class[i] < 0.01:
            plt.text(i, normal_accuracies[i] + 0.05, 'p < 0.01', ha='center', fontsize=8, rotation=90)
        else:
            plt.text(i, normal_accuracies[i] + 0.05, 'p = {:.2f}'.format(p_value_per_class[i]), ha='center', fontsize=8, rotation=90)
            
    # Corrected: Use the correct index for the last bar
    if p_value < 0.01:
        plt.text(len(normal_accuracies) - 1, normal_accuracies[-1] + 0.05, 'p < 0.01', ha='center', fontsize=8, rotation=90)
    else:
        plt.text(len(normal_accuracies) - 1, normal_accuracies[-1] + 0.05, 'p = {:.2f}'.format(p_value), ha='center', fontsize=8, rotation=90)

    plt.title('Prediction accuracy based latent features')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.text(-0.1, 1.0, 'A', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.ylim(0, 1)

    emitter_library = {v: k for k, v in emitter_library.items()}
    x_changed_labels = [emitter_library[label] for label in all_labels]
    # Corrected: Add the overall accuracy label to the changed labels list
    x_changed_labels.append('Total accuracy')
    plt.xticks(range(len(x_labels)), x_changed_labels, rotation=90)
    plt.grid()

    plt.subplot(2, 1, 2)
    normal_accuracies2 = [accuracy_per_class2[label2] for label2 in all_labels2]
    permuted_accuracies2 = [mean_permuted_accuracies_per_class2[label2] for label2 in all_labels2]
    permuted_std_errors2 = [std_permuted_accuracies_per_class2[label2] for label2 in all_labels2]
    x_labels2 = [str(label2) for label2 in all_labels2]
    p_value_per_class2 = [p_value_per_class2[label2] for label2 in all_labels2]


    # Add the overall accuracy as an extra bar
    # Corrected: Use append without re-assignment
    normal_accuracies2.append(accuracy2)
    permuted_accuracies2.append(mean_permuted_accuracy2)
    permuted_std_errors2.append(std_permuted_accuracy2)
    x_labels2.append('Overall accuracy')

    plt.bar(x_labels2, normal_accuracies2, yerr=[0]*len(normal_accuracies2), label='Accuracy')
    plt.bar(x_labels2, permuted_accuracies2, yerr=permuted_std_errors2, alpha=0.9, label='Permuted Accuracy')

    # For every bar in normal_accuracies, add a text with the p-value above the bar
    for i in range(len(p_value_per_class2)):
        if p_value_per_class2[i] < 0.01:
            plt.text(i, normal_accuracies2[i] + 0.05, 'p < 0.01', ha='center', fontsize=8, rotation=90)
        else:
            plt.text(i, normal_accuracies2[i] + 0.05, 'p = {:.2f}'.format(p_value_per_class2[i]), ha='center', fontsize=8, rotation=90) # Corrected label

    # Corrected: Use the correct index for the last bar
    if p_value2 < 0.01:
        plt.text(len(normal_accuracies2) - 1, normal_accuracies2[-1] + 0.05, 'p < 0.01', ha='center', fontsize=8, rotation=90)
    else:
        plt.text(len(normal_accuracies2) - 1, normal_accuracies2[-1] + 0.05, 'p = {:.2f}'.format(p_value2), ha='center', fontsize=8, rotation=90) # Corrected label

    plt.title('Prediction accuracy based on traditional features')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    x_changed_labels2 = [emitter_library[label2] for label2 in all_labels2]
    # Corrected: Add the overall accuracy label to the changed labels list
    x_changed_labels2.append('Total accuracy')
    plt.xticks(range(len(x_labels2)), x_changed_labels2, rotation=90)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.text(-0.1, 1.0, 'B', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)

    # Print model name used in bottom left corner of the plot
    plt.figtext(0.01, 0.01 ,'Model_name: {}'.format(path_to_model), fontsize=8, verticalalignment='bottom')
    plt.show()

# def strain_prediction_model(spec, emitter, path_to_model, n_neighbours = 30, n_permutations =100):

#     #load the trained parameters used in training
#     parameters = joblib.load(path_to_model + '/parameters.pkl')

#     ##load spec_indices for train and test set
#     spec_indices_train = torch.load(path_to_model + '/spec_indices_train')
#     spec_indices_test = torch.load(path_to_model + '/spec_indices_test')

#     #load labels_train and labels_test based on the spec_indices
#     labels_train = torch.load(path_to_model + '/labels_train')
#     labels_test = torch.load(path_to_model + '/labels_test')

#     #get the spectograms
#     train_specs = []
#     test_specs = []

#     for i in spec_indices_train:
#         train_specs.append(spec[i])
#     for i in spec_indices_test:
#         test_specs.append(spec[i])

    

#     #check if there are as many labels as spectograms in the train and test sets
#     assert len(train_specs) == len(labels_train), "Number of train spectograms and labels do not match"
#     assert len(test_specs) == len(labels_test), "Number of test spectograms and labels do not match"

#     #pad the spectograms according to our training procedure
#     padding_length = 160
#     train_padded_specs = create_padded_spectograms.pad_spectograms_sorted(train_specs, padding_length)
#     test_padded_specs = create_padded_spectograms.pad_spectograms_sorted(test_specs, padding_length)

#     #delete unnecesarry big files in RAM
#     del spec, train_specs, test_specs

#     #normalize according to our training
#     if parameters['max_value_per_spec'] == 'true':
#         max_value_per_spec = True
#         print(max_value_per_spec)
#     train_padded_specs = normalize_padded_spectograms.normalize_specs(train_padded_specs, max_value_per_spec)
#     test_padded_specs = normalize_padded_spectograms.normalize_specs(test_padded_specs, max_value_per_spec)

#     #convert to torch tensors
#     latent_train_loader = torch.utils.data.DataLoader(train_padded_specs, batch_size=1, shuffle=False)
#     latent_test_loader = torch.utils.data.DataLoader(test_padded_specs, batch_size=1, shuffle=False)

#     #delete the padded spectograms from memory
#     del train_padded_specs, test_padded_specs

#     #initialize model
#     latent_space_size = parameters['latent_space_size']
#     slope_leaky = parameters['slope_leaky']
#     learning_rate = parameters['learning_rate']
#     precision_model = parameters['precision_model']
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = autoencoder_functions.VAE_KL_leaky(z_dim = latent_space_size, device_name='cuda',slope=slope_leaky,lr = learning_rate, model_precision =precision_model)
#     model = model.to(device)

#     #load the model
#     model_file_name = path_to_model + '/model.pt'
#     model.load_state_dict(torch.load(model_file_name))

#     train_latent = model.get_latent(latent_train_loader)
#     test_latent = model.get_latent(latent_test_loader)

#     print('train latent shape:',train_latent.shape)
#     print('test latent shape:',test_latent.shape) 
  
#     #get the nearest neighbours of the test set in the train set
#     neigh = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
#     neigh.fit(train_latent)
#     distances, indices = neigh.kneighbors(test_latent)
#     print("Indices shape: ", indices.shape)
#     print("Distances shape: ", distances.shape)

#     #get the labels of the nearest neighbours
#     labels_nearest_neighbours = labels_train[indices]
#     print("Labels nearest neighbours shape: ", labels_nearest_neighbours.shape)

#     #determine the predicted label based on the nearest neighbours
#     predicted_labels = np.zeros(test_latent.shape[0])
#     for i in range(test_latent.shape[0]):
#         #get the labels of the nearest neighbours
#         labels_nearest_neighbours = labels_train[indices[i]]
#         #get the unique labels and their counts
#         unique_labels, counts = np.unique(labels_nearest_neighbours, return_counts=True)
#         #get the index of the label with the most counts
#         predicted_labels[i] = unique_labels[np.argmax(counts)]       

#     #determine the accuracy of the total prediction and the accuracy of the prediction for each class
#     accuracy = np.sum(predicted_labels == labels_test) / len(labels_test)
#     accuracy_WT = np.sum(predicted_labels[labels_test == 0] == labels_test[labels_test == 0]) / len(labels_test[labels_test == 0])
#     accuracy_KO = np.sum(predicted_labels[labels_test == 1] == labels_test[labels_test == 1]) / len(labels_test[labels_test == 1])

#     #now permute the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy
#     permuted_accuracies = np.zeros(n_permutations)
#     permuted_accuracy_WT = np.zeros(n_permutations)
#     permuted_accuracy_KO = np.zeros(n_permutations)
#     for i in range(n_permutations):
#         np.random.shuffle(labels_train)
#          #determine the predicted label based on the nearest neighbours
#         predicted_labels = np.zeros(test_latent.shape[0])
#         for j in range(test_latent.shape[0]):
#             #get the labels of the nearest neighbours
#             labels_nearest_neighbours = labels_train[indices[j]]
#             #get the unique labels and their counts
#             unique_labels, counts = np.unique(labels_nearest_neighbours, return_counts=True)
#             #get the index of the label with the most counts
#             predicted_labels[j] = unique_labels[np.argmax(counts)]   
#         permuted_accuracies[i] = np.sum(predicted_labels == labels_test) / len(labels_test)
#         permuted_accuracy_WT[i] = np.sum(predicted_labels[labels_test == 0] == labels_test[labels_test == 0]) / len(labels_test[labels_test == 0])
#         permuted_accuracy_KO[i] = np.sum(predicted_labels[labels_test == 1] == labels_test[labels_test == 1]) / len(labels_test[labels_test == 1])
#     print(permuted_accuracies)
#     mean_permuted_accuracy = np.mean(permuted_accuracies)
#     mean_permuted_accuracy_WT = np.mean(permuted_accuracy_WT)
#     mean_permuted_accuracy_KO = np.mean(permuted_accuracy_KO)
#     std_permuted_accuracy = np.std(permuted_accuracies)
#     std_permuted_accuracy_WT = np.std(permuted_accuracy_WT)
#     std_permuted_accuracy_KO = np.std(permuted_accuracy_KO)

#     #now plot the overall accuracy in the first subplot and the accuracy for each class in the second subplot
#     plt.figure(figsize=(10, 5))
#     plt.suptitle('Strain prediction based on model')
#     plt.subplot(2, 1, 1)
#     labels = ['WT', 'KO']
#     normal_accuracies = [accuracy_WT, accuracy_KO]
#     permuted_accuracies = [mean_permuted_accuracy_WT, mean_permuted_accuracy_KO]
#     permuted_std_errors = [std_permuted_accuracy_WT, std_permuted_accuracy_KO]
#     x_labels = ['Accuracy WT', 'Accuracy KO']
#     plt.bar(x_labels, normal_accuracies, yerr=[0, 0], label='Accuracy')
#     plt.bar(x_labels, permuted_accuracies, yerr=permuted_std_errors, alpha=0.75, label='Permuted Accuracy')
#     plt.title('Accuracy for each class')
#     plt.ylabel('Accuracy')
#     plt.legend() 
#     plt.text(-0.1, 1.0, 'A', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
#     plt.tight_layout()
#     plt.ylim(0, 1)
#     plt.xticks([0, 1], ['Accuracy WT', 'Accuracy KO'])
#     plt.grid()
#     plt.subplot(2, 1, 2)
#     plt.bar(['Accuracy', 'Permuted Accuracy'], [accuracy, mean_permuted_accuracy], yerr=[0, std_permuted_accuracy])
#     plt.title('Overall accuracy')
#     plt.ylabel('Accuracy')
#     plt.ylim(0, 1)
#     plt.xticks([0, 1], ['Accuracy', 'Permuted Accuracy'])
#     plt.grid()
#     plt.tight_layout()
#     plt.text(-0.1, 1.0, 'B', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
#     # plt.text(0,-0.3 ,'Model_name: {}'.format(path_to_model))
#     # plt.text(0,-0.4 ,'Number of neighbours: {}'.format(n_neighbours))
#     plt.show()

# def strain_prediction_hardcoded(features, emitter,emitter_library, n_neighbours = 30, n_permutations =100, string_to_print = ''):

#     #create an indices list with len emitter
#     labels = []
#     for i in emitter:
#         label = emitter_library[i]
#         labels.append(label)
    
#     print('Amount of vocalisations',len(labels))

#     indices = []
#     labels_after_removal = []
#     for i in range(len(labels)):
#         if labels[i] == 0:
#             indices.append(i)
#             labels_after_removal.append(0)
#         elif labels[i] == 1:
#             indices.append(i)
#             labels_after_removal.append(1)

#     labels = labels_after_removal
#     labels = np.array(labels)
#     indices = np.array(indices)
#     print('Amount of vocalisations after females removed:',len(indices))
#     print('Females removed:',len(emitter)-len(indices))

#     #remove features not in indices
#     features_after_removal = []
#     for i in indices:
#         features_after_removal.append(features[i])

#     features = features_after_removal



#     print('Number of instances in each class')
#     print('WT:',np.sum(labels==0))
#     print('KO:',np.sum(labels==1))

#     #seperate the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#     print('Number of instances in each class in training set')
#     print('WT:',np.sum(y_train==0))
#     print('KO:',np.sum(y_train==1))
#     print('Total:',len(y_train))
#     print('Number of features:',len(X_train[0]))
#     print('Number of instances in each class in testing set')
#     print('WT:',np.sum(y_test==0))
#     print('KO:',np.sum(y_test==1))
#     print('Total:',len(y_test))
#     print('Number of features:',len(X_test[0]))
#     weight_WT = 1/np.sum(y_train==0)
#     weight_KO = 1/np.sum(y_train==1)
#     print('Weight WT =',weight_WT)
#     print('Weight KO =', weight_KO)
#     weights = {0: weight_WT, 1: weight_KO}

#     #perform normalization on the train data and transform it to the test data
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     train_latent = X_train
#     test_latent = X_test
#     labels_train = y_train
#     labels_test = y_test

#     #perform umap on train_latent and fit to test latent
#     reducer = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 2)
#     reducer.fit(train_latent)
#     train_latent = reducer.transform(train_latent)
#     test_latent = reducer.transform(test_latent)

#     print('Train latent shape after umap =',np.shape(train_latent))
#     print('Test latent shape after umap =',np.shape(test_latent))

#        #get the nearest neighbours of the test set in the train set
#     neigh = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
#     neigh.fit(train_latent)
#     distances, indices = neigh.kneighbors(test_latent)
#     print("Indices shape: ", indices.shape)
#     print("Distances shape: ", distances.shape)

#     #get the labels of the nearest neighbours
#     labels_nearest_neighbours = labels_train[indices]
#     print("Labels nearest neighbours shape: ", labels_nearest_neighbours.shape)

#     #determine the predicted label based on the nearest neighbours
#     predicted_labels = np.zeros(test_latent.shape[0])
#     for i in range(test_latent.shape[0]):
#         #get the labels of the nearest neighbours
#         labels_nearest_neighbours = labels_train[indices[i]]
#         #get the unique labels and their counts
#         unique_labels, counts = np.unique(labels_nearest_neighbours, return_counts=True)
#         #calculate wieghted counts
#         weighted_counts = np.zeros(len(unique_labels))
#         for j in range(len(unique_labels)):
#             label = unique_labels[j]
#             weighted_counts[j] = counts[j] * weights[label]
#         #get the index of the label with the most counts
#         predicted_labels[i] = unique_labels[np.argmax(weighted_counts)]       

#     #determine the accuracy of the total prediction and the accuracy of the prediction for each class
#     accuracy = np.sum(predicted_labels == labels_test) / len(labels_test)
#     accuracy_WT = np.sum(predicted_labels[labels_test == 0] == labels_test[labels_test == 0]) / len(labels_test[labels_test == 0])
#     accuracy_KO = np.sum(predicted_labels[labels_test == 1] == labels_test[labels_test == 1]) / len(labels_test[labels_test == 1])

#     #now permute the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy
#     permuted_accuracies = np.zeros(n_permutations)
#     permuted_accuracy_WT = np.zeros(n_permutations)
#     permuted_accuracy_KO = np.zeros(n_permutations)
#     for i in range(n_permutations):
#         np.random.shuffle(labels_train)
#          #determine the predicted label based on the nearest neighbours
#         predicted_labels = np.zeros(test_latent.shape[0])
#         for j in range(test_latent.shape[0]):
#             #get the labels of the nearest neighbours
#             labels_nearest_neighbours = labels_train[indices[j]]
#             #get the unique labels and their counts
#             unique_labels, counts = np.unique(labels_nearest_neighbours, return_counts=True)
#             #calculate weighted counts
#             weighted_counts = np.zeros(len(unique_labels))
#             for k in range(len(unique_labels)):
#                 label = unique_labels[k]
#                 weighted_counts[k] = counts[k] * weights[label]
#             #get the index of the label with the most counts
#             predicted_labels[j] = unique_labels[np.argmax(weighted_counts)]   
#         permuted_accuracies[i] = np.sum(predicted_labels == labels_test) / len(labels_test)
#         permuted_accuracy_WT[i] = np.sum(predicted_labels[labels_test == 0] == labels_test[labels_test == 0]) / len(labels_test[labels_test == 0])
#         permuted_accuracy_KO[i] = np.sum(predicted_labels[labels_test == 1] == labels_test[labels_test == 1]) / len(labels_test[labels_test == 1])
#     print(permuted_accuracies)
#     mean_permuted_accuracy = np.mean(permuted_accuracies)
#     mean_permuted_accuracy_WT = np.mean(permuted_accuracy_WT)
#     mean_permuted_accuracy_KO = np.mean(permuted_accuracy_KO)
#     std_permuted_accuracy = np.std(permuted_accuracies)
#     std_permuted_accuracy_WT = np.std(permuted_accuracy_WT)
#     std_permuted_accuracy_KO = np.std(permuted_accuracy_KO)

#     #count how many times the permuted accuracy is higher than the normal accuracy for p-value calculation
#     count_higher = np.sum(permuted_accuracies >= accuracy)
#     count_higher_WT = np.sum(permuted_accuracy_WT >= accuracy_WT)
#     count_higher_KO = np.sum(permuted_accuracy_KO >= accuracy_KO)
#     p_value = count_higher / n_permutations
#     p_value_WT = count_higher_WT / n_permutations
#     p_value_KO = count_higher_KO / n_permutations


#     #now plot the overall accuracy in the first subplot and the accuracy for each class in the second subplot
#     plt.figure()
#     plt.suptitle('Strain prediction based on hardcoded features', fontweight='bold')
#     plt.subplot(2, 1, 1)
#     labels = ['WT', 'KO']
#     normal_accuracies = [accuracy_WT, accuracy_KO]
#     permuted_accuracies = [mean_permuted_accuracy_WT, mean_permuted_accuracy_KO]
#     permuted_std_errors = [std_permuted_accuracy_WT, std_permuted_accuracy_KO]
#     x_labels = ['Accuracy WT', 'Accuracy KO']
#     plt.bar(x_labels, normal_accuracies, yerr=[0, 0], label='Accuracy')
#     plt.bar(x_labels, permuted_accuracies, yerr=permuted_std_errors, alpha=0.75, label='Permuted Accuracy')
#     #above each bar show the p-value for WT and KO
#     if p_value_WT < 0.01:
#         plt.text(0, normal_accuracies[0] + 0.05, 'p < 0.01', ha='center')
#     else:
#         plt.text(0, normal_accuracies[0] + 0.05, 'p = {:.2f}'.format(p_value_WT), ha='center')
#     if p_value_KO < 0.01:
#         plt.text(1, normal_accuracies[1] + 0.05, 'p < 0.01', ha='center')
#     else:
#         plt.text(1, normal_accuracies[1] + 0.05, 'p = {:.2f}'.format(p_value_KO), ha='center')
#     plt.title('Accuracy for each class')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.tight_layout() 
#     plt.ylim(0, 1)
#     plt.xticks([0, 1], ['Accuracy WT', 'Accuracy KO'])
#     plt.grid()
#     plt.text(-0.1, 1.0, 'A', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
#     plt.subplot(2, 1, 2)
#     plt.bar(['Accuracy', 'Permuted Accuracy'], [accuracy, mean_permuted_accuracy], yerr=[0, std_permuted_accuracy])
#     #show p-value in the top-left corner of the subplot
#     if p_value < 0.01:
#         plt.text(0.01, 0.95, 'p < 0.01', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
#     else:
#         plt.text(0.01, 0.95, 'p = {:.2f}'.format(p_value), transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
#     plt.title('Overall accuracy')
#     plt.ylabel('Accuracy')
#     plt.ylim(0, 1)
#     plt.xticks([0, 1], ['Accuracy', 'Permuted Accuracy'])
#     plt.grid()
#     plt.text(-0.1, 1.0, 'B', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
#     plt.tight_layout() 
#     #print which dataset was used in the bottom left corner of the figure
#     plt.figtext(0.01, 0.01, 'Dataset used: {}'.format(string_to_print), fontsize=8, verticalalignment='bottom')
#     plt.show()

# def strain_prediction_model_weighted(spec, emitter, path_to_model, n_neighbours = 30, n_permutations =100):

#     #load the trained parameters used in training
#     parameters = joblib.load(path_to_model + '/parameters.pkl')

#     ##load spec_indices for train and test set
#     spec_indices_train = torch.load(path_to_model + '/spec_indices_train')
#     spec_indices_test = torch.load(path_to_model + '/spec_indices_test')

#     #load labels_train and labels_test based on the spec_indices
#     labels_train = torch.load(path_to_model + '/labels_train')
#     labels_test = torch.load(path_to_model + '/labels_test')

#     print('indices before removal of doubles =', len(spec_indices_train))

#     #remove doubles indices from spec_indices_train and remove them from labels_train as well
#     spec_indices_train_after_removal = []
#     labels_train_after_removal = []



#     for i in range(len(spec_indices_train)):
#         if spec_indices_train[i] not in spec_indices_train_after_removal:
#             spec_indices_train_after_removal.append(spec_indices_train[i])
#             labels_train_after_removal.append(labels_train[i])

#     spec_indices_train = spec_indices_train_after_removal
#     labels_train = labels_train_after_removal

#     print('indices after removal of doubles =', len(spec_indices_train_after_removal))
#     labels_train = np.array(labels_train)
#     labels_train = labels_train.astype(int)
#     labels_test = np.array(labels_test)
#     labels_test = labels_test.astype(int)

#     weight_WT = 1/np.sum(labels_train==0)
#     weight_KO = 1/np.sum(labels_train==1)
#     print('Weight WT =',weight_WT)
#     print('Weight KO =', weight_KO)
#     weights = {0: weight_WT, 1: weight_KO}

#     #get the spectograms
#     train_specs = []
#     test_specs = []

#     for i in spec_indices_train:
#         train_specs.append(spec[i])
#     for i in spec_indices_test:
#         test_specs.append(spec[i])

    

#     #check if there are as many labels as spectograms in the train and test sets
#     assert len(train_specs) == len(labels_train), "Number of train spectograms and labels do not match"
#     assert len(test_specs) == len(labels_test), "Number of test spectograms and labels do not match"

#     #pad the spectograms according to our training procedure
#     padding_length = 160
#     train_padded_specs = create_padded_spectograms.pad_spectograms_sorted(train_specs, padding_length)
#     test_padded_specs = create_padded_spectograms.pad_spectograms_sorted(test_specs, padding_length)

#     #delete unnecesarry big files in RAM
#     del spec, train_specs, test_specs

#     #normalize according to our training
#     if parameters['max_value_per_spec'] == 'true':
#         max_value_per_spec = True
#         print(max_value_per_spec)
#     train_padded_specs = normalize_padded_spectograms.normalize_specs(train_padded_specs, max_value_per_spec)
#     test_padded_specs = normalize_padded_spectograms.normalize_specs(test_padded_specs, max_value_per_spec)

#     #convert to torch tensors
#     latent_train_loader = torch.utils.data.DataLoader(train_padded_specs, batch_size=1, shuffle=False)
#     latent_test_loader = torch.utils.data.DataLoader(test_padded_specs, batch_size=1, shuffle=False)

#     #delete the padded spectograms from memory
#     del train_padded_specs, test_padded_specs

#     #initialize model
#     latent_space_size = parameters['latent_space_size']
#     slope_leaky = parameters['slope_leaky']
#     learning_rate = parameters['learning_rate']
#     precision_model = parameters['precision_model']
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = autoencoder_functions.VAE_KL_leaky(z_dim = latent_space_size, device_name='cuda',slope=slope_leaky,lr = learning_rate, model_precision =precision_model)
#     model = model.to(device)

#     #load the model
#     model_file_name = path_to_model + '/model.pt'
#     model.load_state_dict(torch.load(model_file_name))

#     train_latent = model.get_latent(latent_train_loader)
#     test_latent = model.get_latent(latent_test_loader)

#     print('train latent shape:',train_latent.shape)
#     print('test latent shape:',test_latent.shape) 
#     X_train = train_latent
#     X_test = test_latent
#     # #perform normalization on the train data and transform it to the test data
#     # scaler = StandardScaler()
#     # X_train = scaler.fit_transform(X_train)
#     # X_test = scaler.transform(X_test)

#     # train_latent = X_train
#     # test_latent = X_test


#     # #perform umap on train_latent and fit to test latent
#     # reducer = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 2)
#     # reducer.fit(train_latent)
#     # train_latent = reducer.transform(train_latent)
#     # test_latent = reducer.transform(test_latent)

#     # print('Train latent shape after umap =',np.shape(train_latent))
#     # print('Test latent shape after umap =',np.shape(test_latent))
#     #get the nearest neighbours of the test set in the train set
#     neigh = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
#     neigh.fit(train_latent)
#     distances, indices = neigh.kneighbors(test_latent)
#     print("Indices shape: ", indices.shape)
#     print("Distances shape: ", distances.shape)

#     #get the labels of the nearest neighbours
#     labels_nearest_neighbours = labels_train[indices]
#     print("Labels nearest neighbours shape: ", labels_nearest_neighbours.shape)

#     #determine the predicted label based on the nearest neighbours
#     predicted_labels = np.zeros(test_latent.shape[0])
#     for i in range(test_latent.shape[0]):
#         #get the labels of the nearest neighbours
#         labels_nearest_neighbours = labels_train[indices[i]]
#         #get the unique labels and their counts
#         unique_labels, counts = np.unique(labels_nearest_neighbours, return_counts=True)
#         #calculate wieghted counts
#         weighted_counts = np.zeros(len(unique_labels))
#         for j in range(len(unique_labels)):
#             label = unique_labels[j]
#             weighted_counts[j] = counts[j] * weights[label]
#         #get the index of the label with the most counts
#         predicted_labels[i] = unique_labels[np.argmax(weighted_counts)]       

#     #determine the accuracy of the total prediction and the accuracy of the prediction for each class
#     accuracy = np.sum(predicted_labels == labels_test) / len(labels_test)
#     accuracy_WT = np.sum(predicted_labels[labels_test == 0] == labels_test[labels_test == 0]) / len(labels_test[labels_test == 0])
#     accuracy_KO = np.sum(predicted_labels[labels_test == 1] == labels_test[labels_test == 1]) / len(labels_test[labels_test == 1])

#     #now permute the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy
#     permuted_accuracies = np.zeros(n_permutations)
#     permuted_accuracy_WT = np.zeros(n_permutations)
#     permuted_accuracy_KO = np.zeros(n_permutations)
#     for i in range(n_permutations):
#         np.random.shuffle(labels_train)
#          #determine the predicted label based on the nearest neighbours
#         predicted_labels = np.zeros(test_latent.shape[0])
#         for j in range(test_latent.shape[0]):
#             #get the labels of the nearest neighbours
#             labels_nearest_neighbours = labels_train[indices[j]]
#             #get the unique labels and their counts
#             unique_labels, counts = np.unique(labels_nearest_neighbours, return_counts=True)
#             #calculate weighted counts
#             weighted_counts = np.zeros(len(unique_labels))
#             for k in range(len(unique_labels)):
#                 label = unique_labels[k]
#                 weighted_counts[k] = counts[k] * weights[label]
#             #get the index of the label with the most counts
#             predicted_labels[j] = unique_labels[np.argmax(weighted_counts)]   
#         permuted_accuracies[i] = np.sum(predicted_labels == labels_test) / len(labels_test)
#         permuted_accuracy_WT[i] = np.sum(predicted_labels[labels_test == 0] == labels_test[labels_test == 0]) / len(labels_test[labels_test == 0])
#         permuted_accuracy_KO[i] = np.sum(predicted_labels[labels_test == 1] == labels_test[labels_test == 1]) / len(labels_test[labels_test == 1])
#     print(permuted_accuracies)
#     mean_permuted_accuracy = np.mean(permuted_accuracies)
#     mean_permuted_accuracy_WT = np.mean(permuted_accuracy_WT)
#     mean_permuted_accuracy_KO = np.mean(permuted_accuracy_KO)
#     std_permuted_accuracy = np.std(permuted_accuracies)
#     std_permuted_accuracy_WT = np.std(permuted_accuracy_WT)
#     std_permuted_accuracy_KO = np.std(permuted_accuracy_KO)

#     #calculate the p-values for every individual emitter and the p-value for the total accuracy
#     count_higher = np.sum(permuted_accuracies >= accuracy)
#     count_higher_WT = np.sum(permuted_accuracy_WT >= accuracy_WT)
#     count_higher_KO = np.sum(permuted_accuracy_KO >= accuracy_KO)
#     p_value = count_higher / n_permutations
#     p_value_WT = count_higher_WT / n_permutations
#     p_value_KO = count_higher_KO / n_permutations

#     #now plot the overall accuracy in the first subplot and the accuracy for each class in the second subplot
#     plt.figure()
#     plt.suptitle('Strain prediction based on model', fontweight='bold')
#     plt.subplot(2, 1, 1)
#     labels = ['WT', 'KO']
#     normal_accuracies = [accuracy_WT, accuracy_KO]
#     permuted_accuracies = [mean_permuted_accuracy_WT, mean_permuted_accuracy_KO]
#     permuted_std_errors = [std_permuted_accuracy_WT, std_permuted_accuracy_KO]
#     x_labels = ['Accuracy WT', 'Accuracy KO']
#     plt.bar(x_labels, normal_accuracies, yerr=[0, 0], label='Accuracy')
#     plt.bar(x_labels, permuted_accuracies, yerr=permuted_std_errors, alpha=0.75, label='Permuted Accuracy')
#     if p_value_WT < 0.01:
#         plt.text(0, normal_accuracies[0] + 0.05, 'p < 0.01', ha='center')
#     else:
#         plt.text(0, normal_accuracies[0] + 0.05, 'p = {:.2f}'.format(p_value_WT), ha='center')
#     if p_value_KO < 0.01:
#         plt.text(1, normal_accuracies[1] + 0.05, 'p < 0.01', ha='center')
#     else:
#         plt.text(1, normal_accuracies[1] + 0.05, 'p = {:.2f}'.format(p_value_KO), ha='center')
#     plt.title('Accuracy for each class')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.text(-0.1, 1.0, 'A', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
#     plt.tight_layout() #adjust plot to fit legend.
#     plt.ylim(0, 1)
#     plt.xticks([0, 1], ['Accuracy WT', 'Accuracy KO'])
#     plt.grid()
#     plt.subplot(2, 1, 2)
#     plt.bar(['Accuracy', 'Permuted Accuracy'], [accuracy, mean_permuted_accuracy], yerr=[0, std_permuted_accuracy])
#     if p_value < 0.01:
#         plt.text(0.01, 0.95, 'p < 0.01', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
#     else:
#         plt.text(0.01, 0.95, 'p = {:.2f}'.format(p_value), transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
#     plt.title('Overall accuracy')
#     plt.ylabel('Accuracy')
#     plt.ylim(0, 1)
#     plt.xticks([0, 1], ['Accuracy', 'Permuted Accuracy'])
#     plt.grid()
#     plt.tight_layout() 
#     plt.text(-0.1, 1.0, 'B', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
#     #print model used in the bottom left_corner of the figure
#     plt.figtext(0.01, 0.01, 'Model_name: {}'.format(path_to_model),  fontsize=8, verticalalignment='bottom')
#     plt.show()
  
# def emitter_prediction_model(spec, emitter, emitter_library, path_to_model, n_neighbours = 30, n_permutations =100):

#         #load the trained parameters used in training
#     parameters = joblib.load(path_to_model + '/parameters.pkl')

#     ##load spec_indices for train and test set
#     spec_indices_train = torch.load(path_to_model + '/spec_indices_train')
#     spec_indices_test = torch.load(path_to_model + '/spec_indices_test')

#     emitter_int = [emitter_library[e] for e in emitter]

#     #get the spectograms and emitters
#     train_specs = []
#     test_specs = []
#     labels_train = []
#     labels_test = []

#     for i in spec_indices_train:
#         train_specs.append(spec[i])
#         labels_train.append(emitter_int[i])
#     for i in spec_indices_test:
#         test_specs.append(spec[i])
#         labels_test.append(emitter_int[i])

#     #convert labels to integer numpy arrays
#     labels_train = np.array(labels_train)
#     labels_train = labels_train.astype(int)
#     labels_test = np.array(labels_test)
#     labels_test = labels_test.astype(int)

    
#     #check if there are as many labels as spectograms in the train and test sets
#     assert len(train_specs) == len(labels_train), "Number of train spectograms and labels do not match"
#     assert len(test_specs) == len(labels_test), "Number of test spectograms and labels do not match"

#     #pad the spectograms according to our training procedure
#     padding_length = 160
#     train_padded_specs = create_padded_spectograms.pad_spectograms_sorted(train_specs, padding_length)
#     test_padded_specs = create_padded_spectograms.pad_spectograms_sorted(test_specs, padding_length)

#     #delete unnecesarry big files in RAM
#     del spec, train_specs, test_specs

#     #normalize according to our training
#     if parameters['max_value_per_spec'] == 'true':
#         max_value_per_spec = True
#         print(max_value_per_spec)
#     train_padded_specs = normalize_padded_spectograms.normalize_specs(train_padded_specs, max_value_per_spec)
#     test_padded_specs = normalize_padded_spectograms.normalize_specs(test_padded_specs, max_value_per_spec)

#     #convert to torch tensors
#     latent_train_loader = torch.utils.data.DataLoader(train_padded_specs, batch_size=1, shuffle=False)
#     latent_test_loader = torch.utils.data.DataLoader(test_padded_specs, batch_size=1, shuffle=False)

#     #delete the padded spectograms from memory
#     del train_padded_specs, test_padded_specs

#     #initialize model
#     latent_space_size = parameters['latent_space_size']
#     slope_leaky = parameters['slope_leaky']
#     learning_rate = parameters['learning_rate']
#     precision_model = parameters['precision_model']
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = autoencoder_functions.VAE_KL_leaky(z_dim = latent_space_size, device_name='cuda',slope=slope_leaky,lr = learning_rate, model_precision =precision_model)
#     model = model.to(device)

#     #load the model
#     model_file_name = path_to_model + '/model.pt'
#     model.load_state_dict(torch.load(model_file_name))

#     train_latent = model.get_latent(latent_train_loader)
#     test_latent = model.get_latent(latent_test_loader)

#     print('train latent shape:',train_latent.shape)
#     print('test latent shape:',test_latent.shape) 
  
#     #get the nearest neighbours of the test set in the train set
#     neigh = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
#     neigh.fit(train_latent)
#     distances, indices = neigh.kneighbors(test_latent)
#     print("Indices shape: ", indices.shape)
#     print("Distances shape: ", distances.shape)

#     #get the labels of the nearest neighbours
#     labels_nearest_neighbours = labels_train[indices]
#     print("Labels nearest neighbours shape: ", labels_nearest_neighbours.shape)

#     #determine the predicted label based on the nearest neighbours
#     predicted_labels = np.zeros(test_latent.shape[0])
#     for i in range(test_latent.shape[0]):
#         #get the labels of the nearest neighbours
#         labels_nearest_neighbours = labels_train[indices[i]]
#         #get the unique labels and their counts
#         unique_labels, counts = np.unique(labels_nearest_neighbours, return_counts=True)
#         #get the index of the label with the most counts
#         predicted_labels[i] = unique_labels[np.argmax(counts)]       

#     #determine the accuracy of the total prediction and the accuracy of the prediction for each class
#     accuracy = np.sum(predicted_labels == labels_test) / len(labels_test)

#     # Get a complete list of all unique labels from both train and test sets
#     all_labels = np.unique(np.concatenate((labels_train, labels_test), axis=0))

#     # Initialize accuracy dictionary for all possible labels
#     accuracy_per_class = {label: 0.0 for label in all_labels}
    
#     # Calculate accuracy for each class that exists in the labels_test
#     unique_labels_present = np.unique(labels_test)
#     for label in unique_labels_present:
#         accuracy_per_class[label] = np.sum(predicted_labels[labels_test == label] == labels_test[labels_test == label]) / len(labels_test[labels_test == label])

#     print("Overall accuracy: ", accuracy)
#     print("Accuracy per class: ", accuracy_per_class)

#     #now permut the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy for the
#     #overall permuted accurcacy and the permuted accuracy for each class
#     permuted_accuracies = np.zeros(n_permutations)
#     permuted_accuracies_per_class = {label: np.zeros(n_permutations) for label in all_labels}
#     for i in range(n_permutations):
#         np.random.shuffle(labels_train)

#         permuted_predicted_labels = np.zeros(test_latent.shape[0])
#         for j in range(test_latent.shape[0]):
#             #get the labels of the nearest neighbours
#             labels_nearest_neighbours = labels_train[indices[j]]
#             #get the unique labels and their counts
#             unique_labels_nn, counts = np.unique(labels_nearest_neighbours, return_counts=True)
#             #get the index of the label with the most counts
#             # if remove_double_specs and use_weights:
#             #     counts = counts * np.array([weights[label] for label in unique_labels_nn])
#             permuted_predicted_labels[j] = unique_labels_nn[np.argmax(counts)]
        
#         permuted_accuracies[i] = np.sum(permuted_predicted_labels == labels_test) / len(labels_test)

#         # Calculate and store permuted accuracy for each class that exists in this permutation
#         unique_labels_present_in_perm = np.unique(labels_test)
#         for label in unique_labels_present_in_perm:
#                 permuted_accuracies_per_class[label][i] = np.sum(permuted_predicted_labels[labels_test == label] == labels_test[labels_test == label]) / len(labels_test[labels_test == label])

#     mean_permuted_accuracies_per_class = {label: np.mean(permuted_accuracies_per_class[label]) for label in all_labels}
#     std_permuted_accuracies_per_class = {label: np.std(permuted_accuracies_per_class[label]) for label in all_labels}
    
#     mean_permuted_accuracy = np.mean(permuted_accuracies)
#     std_permuted_accuracy = np.std(permuted_accuracies)
    
#     print("Mean permuted accuracy: ", mean_permuted_accuracy)
#     print("Std permuted accuracy: ", std_permuted_accuracy)
#     print("Mean permuted accuracies per class: ", mean_permuted_accuracies_per_class)
#     print("Std permuted accuracies per class: ", std_permuted_accuracies_per_class)
    
#     #now plot the accurcay per class and permuted accuracy per class in subplot 1 and the overall accuracy and permuted accuracy in subplot 2
#     plt.figure()
#     plt.suptitle('Emitter prediction based on model')
#     plt.subplot(2, 1, 1)
#     normal_accuracies = [accuracy_per_class[label] for label in all_labels]
#     permuted_accuracies = [mean_permuted_accuracies_per_class[label] for label in all_labels]
#     permuted_std_errors = [std_permuted_accuracies_per_class[label] for label in all_labels]
#     x_labels = [str(label) for label in all_labels]
#     plt.bar(x_labels, normal_accuracies, yerr=[0]*len(normal_accuracies), label='Accuracy')
#     plt.bar(x_labels, permuted_accuracies, yerr=permuted_std_errors, alpha=0.9, label='Permuted Accuracy')
#     plt.title('Accuracy for each class')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.text(-0.1, 1.0, 'A', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
#     plt.tight_layout()
#     plt.ylim(0, 1)
#     emitter_library = {v: k for k, v in emitter_library.items()}
#     x_changed_labels = [emitter_library[label] for label in all_labels]
#     plt.xticks(range(len(x_labels)), x_changed_labels, rotation=90)
#     plt.grid()
#     plt.subplot(2, 1, 2)
#     plt.bar(['Accuracy', 'Permuted Accuracy'], [accuracy, mean_permuted_accuracy], yerr=[0, std_permuted_accuracy])
#     plt.title('Overall accuracy')
#     plt.ylabel('Accuracy')
#     plt.ylim(0, 1)
#     plt.xticks([0, 1], ['Accuracy', 'Permuted Accuracy'])
#     plt.grid()
#     plt.tight_layout()
#     plt.text(-0.1, 1.0, 'B', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
#     # plt.text(0,-0.4 ,'Number of neighbours: {}'.format(n_neighbours))
#     plt.show()

# def emitter_prediction_model_weighted(spec, emitter, emitter_library, path_to_model, n_neighbours = 30, n_permutations =100):

#         #load the trained parameters used in training
#     parameters = joblib.load(path_to_model + '/parameters.pkl')

#     ##load spec_indices for train and test set
#     spec_indices_train = torch.load(path_to_model + '/spec_indices_train')
#     spec_indices_test = torch.load(path_to_model + '/spec_indices_test')

#     emitter_int = [emitter_library[e] for e in emitter]

#     #remove doubles in train set  
#     print('indices before removal of doubles =', len(spec_indices_train))
#     spec_indices_train_after_removal = []
#     for i in range(len(spec_indices_train)):
#         if spec_indices_train[i] not in spec_indices_train_after_removal:
#             spec_indices_train_after_removal.append(spec_indices_train[i])
#     spec_indices_train = spec_indices_train_after_removal
#     print('indices after removal of doubles =', len(spec_indices_train_after_removal))

#     #get the spectograms and emitters
#     train_specs = []
#     test_specs = []
#     labels_train = []
#     labels_test = []

#     for i in spec_indices_train:
#         train_specs.append(spec[i])
#         labels_train.append(emitter_int[i])
#     for i in spec_indices_test:
#         test_specs.append(spec[i])
#         labels_test.append(emitter_int[i])

#     #convert labels to integer numpy arrays
#     labels_train = np.array(labels_train)
#     labels_train = labels_train.astype(int)
#     labels_test = np.array(labels_test)
#     labels_test = labels_test.astype(int)

#     #check if the same labels are present in labels_train and labels_test
#     # Get the unique labels from each array
#     unique_labels_train = np.unique(labels_train)
#     unique_labels_test = np.unique(labels_test)

#     # Check if the sets of unique labels are equal
#     if np.array_equal(unique_labels_train, unique_labels_test):
#         print("The same labels are present in both labels_train and labels_test.")
#     else:
#         print("The labels in labels_train and labels_test are different.")
#         print("Unique labels in training set:", unique_labels_train)
#         print("Unique labels in testing set:", unique_labels_test)

#     #make a weight list for the integers in labels train
#     unique_labels, counts = np.unique(labels_train, return_counts=True)
#     weights = {}
#     for label, count in zip(unique_labels, counts):
#         weights[label] = 1 / count

#     print(weights)

#     #check if there are as many labels as spectograms in the train and test sets
#     assert len(train_specs) == len(labels_train), "Number of train spectograms and labels do not match"
#     assert len(test_specs) == len(labels_test), "Number of test spectograms and labels do not match"

#     #pad the spectograms according to our training procedure
#     padding_length = 160
#     train_padded_specs = create_padded_spectograms.pad_spectograms_sorted(train_specs, padding_length)
#     test_padded_specs = create_padded_spectograms.pad_spectograms_sorted(test_specs, padding_length)

#     #delete unnecesarry big files in RAM
#     del spec, train_specs, test_specs

#     #normalize according to our training
#     if parameters['max_value_per_spec'] == 'true':
#         max_value_per_spec = True
#         print(max_value_per_spec)
#     train_padded_specs = normalize_padded_spectograms.normalize_specs(train_padded_specs, max_value_per_spec)
#     test_padded_specs = normalize_padded_spectograms.normalize_specs(test_padded_specs, max_value_per_spec)

#     #convert to torch tensors
#     latent_train_loader = torch.utils.data.DataLoader(train_padded_specs, batch_size=1, shuffle=False)
#     latent_test_loader = torch.utils.data.DataLoader(test_padded_specs, batch_size=1, shuffle=False)

#     #delete the padded spectograms from memory
#     del train_padded_specs, test_padded_specs

#     #initialize model
#     latent_space_size = parameters['latent_space_size']
#     slope_leaky = parameters['slope_leaky']
#     learning_rate = parameters['learning_rate']
#     precision_model = parameters['precision_model']
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = autoencoder_functions.VAE_KL_leaky(z_dim = latent_space_size, device_name='cuda',slope=slope_leaky,lr = learning_rate, model_precision =precision_model)
#     model = model.to(device)

#     #load the model
#     model_file_name = path_to_model + '/model.pt'
#     model.load_state_dict(torch.load(model_file_name))

#     train_latent = model.get_latent(latent_train_loader)
#     test_latent = model.get_latent(latent_test_loader)

#     print('train latent shape:',train_latent.shape)
#     print('test latent shape:',test_latent.shape) 
  
#     #get the nearest neighbours of the test set in the train set
#     neigh = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
#     neigh.fit(train_latent)
#     distances, indices = neigh.kneighbors(test_latent)
#     print("Indices shape: ", indices.shape)
#     print("Distances shape: ", distances.shape)

#     #get the labels of the nearest neighbours
#     labels_nearest_neighbours = labels_train[indices]
#     print("Labels nearest neighbours shape: ", labels_nearest_neighbours.shape)

#     #determine the predicted label based on the nearest neighbours
#     predicted_labels = np.zeros(test_latent.shape[0])
#     for i in range(test_latent.shape[0]):
#         #get the labels of the nearest neighbours
#         labels_nearest_neighbours = labels_train[indices[i]]
#         #get the unique labels and their counts
#         unique_labels, counts = np.unique(labels_nearest_neighbours, return_counts=True)
#         #calculate wieghted counts
#         weighted_counts = np.zeros(len(unique_labels))
#         for j in range(len(unique_labels)):
#             label = unique_labels[j]
#             weighted_counts[j] = counts[j] * weights[label]
#         #get the index of the label with the most counts
#         predicted_labels[i] = unique_labels[np.argmax(weighted_counts)]      

#     #determine the accuracy of the total prediction and the accuracy of the prediction for each class
#     accuracy = np.sum(predicted_labels == labels_test) / len(labels_test)

#     # Get a complete list of all unique labels from both train and test sets
#     all_labels = np.unique(np.concatenate((labels_train, labels_test), axis=0))

#     # Initialize accuracy dictionary for all possible labels
#     accuracy_per_class = {label: 0.0 for label in all_labels}
    
#     # Calculate accuracy for each class that exists in the labels_test
#     unique_labels_present = np.unique(labels_test)
#     for label in unique_labels_present:
#         accuracy_per_class[label] = np.sum(predicted_labels[labels_test == label] == labels_test[labels_test == label]) / len(labels_test[labels_test == label])

#     print("Overall accuracy: ", accuracy)
#     print("Accuracy per class: ", accuracy_per_class)

#     #now permut the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy for the
#     #overall permuted accurcacy and the permuted accuracy for each class
#     permuted_accuracies = np.zeros(n_permutations)
#     permuted_accuracies_per_class = {label: np.zeros(n_permutations) for label in all_labels}
#     for i in range(n_permutations):
#         np.random.shuffle(labels_train)

#         permuted_predicted_labels = np.zeros(test_latent.shape[0])
#         for j in range(test_latent.shape[0]):
#             #get the labels of the nearest neighbours
#             labels_nearest_neighbours = labels_train[indices[j]]
#             #get the unique labels and their counts
#             unique_labels_nn, counts = np.unique(labels_nearest_neighbours, return_counts=True)
#             #calculate weighted counts
#             weighted_counts = np.zeros(len(unique_labels_nn))
#             for k in range(len(unique_labels_nn)):
#                 label = unique_labels_nn[k]
#                 weighted_counts[k] = counts[k] * weights[label]
#             #get the index of the label with the most counts
#             permuted_predicted_labels[j] = unique_labels_nn[np.argmax(weighted_counts)]
        
#         permuted_accuracies[i] = np.sum(permuted_predicted_labels == labels_test) / len(labels_test)

#         # Calculate and store permuted accuracy for each class that exists in this permutation
#         unique_labels_present_in_perm = np.unique(labels_test)
#         for label in unique_labels_present_in_perm:
#                 permuted_accuracies_per_class[label][i] = np.sum(permuted_predicted_labels[labels_test == label] == labels_test[labels_test == label]) / len(labels_test[labels_test == label])

#     mean_permuted_accuracies_per_class = {label: np.mean(permuted_accuracies_per_class[label]) for label in all_labels}
#     std_permuted_accuracies_per_class = {label: np.std(permuted_accuracies_per_class[label]) for label in all_labels}
    
#     mean_permuted_accuracy = np.mean(permuted_accuracies)
#     std_permuted_accuracy = np.std(permuted_accuracies)

#     #calculate the p-values for every individual emitter per class and the p-value for the total accuracy
#     count_higher_per_class = {label: np.sum(permuted_accuracies_per_class[label] >= accuracy_per_class[label]) for label in all_labels}
#     count_higher = np.sum(permuted_accuracies >= accuracy)
#     p_value_per_class = {label: count_higher_per_class[label] / n_permutations for label in all_labels}
#     p_value = count_higher / n_permutations 
    
#     print("Mean permuted accuracy: ", mean_permuted_accuracy)
#     print("Std permuted accuracy: ", std_permuted_accuracy)
#     print("Mean permuted accuracies per class: ", mean_permuted_accuracies_per_class)
#     print("Std permuted accuracies per class: ", std_permuted_accuracies_per_class)
    
#     #now plot the accurcay per class and permuted accuracy per class in subplot 1 and the overall accuracy and permuted accuracy in subplot 2
#     plt.figure()
#     plt.suptitle('Emitter prediction based on model',fontweight='bold')
#     plt.subplot(2, 1, 1)
#     normal_accuracies = [accuracy_per_class[label] for label in all_labels]
#     permuted_accuracies = [mean_permuted_accuracies_per_class[label] for label in all_labels]
#     permuted_std_errors = [std_permuted_accuracies_per_class[label] for label in all_labels]
#     x_labels = [str(label) for label in all_labels]
#     plt.bar(x_labels, normal_accuracies, yerr=[0]*len(normal_accuracies), label='Accuracy')
#     plt.bar(x_labels, permuted_accuracies, yerr=permuted_std_errors, alpha=0.9, label='Permuted Accuracy')
#     #for every bar in normal_accuracies, add a text with the p-value above the bar
#     for i, label in enumerate(all_labels):
#         if p_value_per_class[label] < 0.01:
#             plt.text(i, normal_accuracies[i] + 0.05, 'p < 0.01', ha='center', fontsize=8, rotation=90)
#         else:
#             plt.text(i, normal_accuracies[i] + 0.05, 'p = {:.2f}'.format(p_value_per_class[label]), ha='center', fontsize=8, rotation=90)
#     plt.title('Accuracy for each class')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.text(-0.1, 1.0, 'A', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
#     plt.tight_layout()
#     plt.ylim(0, 1)
#     emitter_library = {v: k for k, v in emitter_library.items()}
#     x_changed_labels = [emitter_library[label] for label in all_labels]
#     plt.xticks(range(len(x_labels)), x_changed_labels, rotation=90)
#     plt.grid()
#     plt.subplot(2, 1, 2)
#     plt.bar(['Accuracy', 'Permuted Accuracy'], [accuracy, mean_permuted_accuracy], yerr=[0, std_permuted_accuracy])
#     if p_value < 0.01:
#         plt.text(0.01, 0.95, 'p < 0.01', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
#     else:
#         plt.text(0.01, 0.95, 'p = {:.2f}'.format(p_value), transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
#     plt.title('Overall accuracy')
#     plt.ylabel('Accuracy')
#     plt.ylim(0, 1)
#     plt.xticks([0, 1], ['Accuracy', 'Permuted Accuracy'])
#     plt.grid()
#     plt.tight_layout()
#     plt.text(-0.1, 1.0, 'B', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
#     #print model name used in bottom left corner of the plot
#     plt.figtext(0.01, 0.01 ,'Model_name: {}'.format(path_to_model),  fontsize=8, verticalalignment='bottom')
#     plt.show()


# def emitter_prediction_hardcoded(features, emitter,emitter_library, n_neighbours = 30, n_permutations =100, string_to_print = ''):

#     #create an indices list with len emitter
#     labels = []
#     for i in emitter:
#         label = emitter_library[i]
#         labels.append(label)
#     print(np.unique(labels, return_counts=True))

#     print('Amount of vocalisations',len(labels))

#     indices = []
#     labels_after_removal = []
#     #remove vocalisations from either starting with capital F or starting with mother or starting with pupx, when checking emitter library
#     for i, emit in enumerate(emitter):
#         if not emit.startswith('F') and not emit.startswith('mother') and not emit.startswith('pupx'):
#             indices.append(i)
#             labels_after_removal.append(labels[i])


#     labels = labels_after_removal
#     labels = np.array(labels)
#     indices = np.array(indices)
#     print('Amount of vocalisations after females removed:',len(indices))
#     print('Females removed:',len(emitter)-len(indices))

#     #remove features not in indices
#     features_after_removal = []
#     for i in indices:
#         features_after_removal.append(features[i])

#     features = features_after_removal

#     print('labels after removal =',np.unique(labels,return_counts=True))
#     #seperate the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#     print('Total:',len(y_train))
#     print('Number of features:',len(X_train[0]))
#     print('Total:',len(y_test))
#     print('Number of features:',len(X_test[0]))

#     #perform normalization on the train data and transform it to the test data
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     train_latent = X_train
#     test_latent = X_test
#     labels_train = y_train
#     labels_test = y_test

#     #make a weight list for the integers in labels train
#     unique_labels, counts = np.unique(labels_train, return_counts=True)
#     weights = {}
#     for label, count in zip(unique_labels, counts):
#         weights[label] = 1 / count

#     print('weights =',weights)

#     #perform umap on train_latent and fit to test latent
#     reducer = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 2)
#     reducer.fit(train_latent)
#     train_latent = reducer.transform(train_latent)
#     test_latent = reducer.transform(test_latent)

#     print('Train latent shape after umap =',np.shape(train_latent))
#     print('Test latent shape after umap =',np.shape(test_latent))

#     #get the nearest neighbours of the test set in the train set
#     neigh = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
#     neigh.fit(train_latent)
#     distances, indices = neigh.kneighbors(test_latent)
#     print("Indices shape: ", indices.shape)
#     print("Distances shape: ", distances.shape)

#     #get the labels of the nearest neighbours
#     labels_nearest_neighbours = labels_train[indices]
#     print("Labels nearest neighbours shape: ", labels_nearest_neighbours.shape)

#     #determine the predicted label based on the nearest neighbours
#     predicted_labels = np.zeros(test_latent.shape[0])
#     for i in range(test_latent.shape[0]):
#         #get the labels of the nearest neighbours
#         labels_nearest_neighbours = labels_train[indices[i]]
#         #get the unique labels and their counts
#         unique_labels, counts = np.unique(labels_nearest_neighbours, return_counts=True)
#         #calculate wieghted counts
#         weighted_counts = np.zeros(len(unique_labels))
#         for j in range(len(unique_labels)):
#             label = unique_labels[j]
#             weighted_counts[j] = counts[j] * weights[label]
#         #get the index of the label with the most counts
#         predicted_labels[i] = unique_labels[np.argmax(weighted_counts)]      

#     #determine the accuracy of the total prediction and the accuracy of the prediction for each class
#     accuracy = np.sum(predicted_labels == labels_test) / len(labels_test)

#     # Get a complete list of all unique labels from both train and test sets
#     all_labels = np.unique(np.concatenate((labels_train, labels_test), axis=0))

#     # Initialize accuracy dictionary for all possible labels
#     accuracy_per_class = {label: 0.0 for label in all_labels}
    
#     # Calculate accuracy for each class that exists in the labels_test
#     unique_labels_present = np.unique(labels_test)
#     for label in unique_labels_present:
#         accuracy_per_class[label] = np.sum(predicted_labels[labels_test == label] == labels_test[labels_test == label]) / len(labels_test[labels_test == label])

#     print("Overall accuracy: ", accuracy)
#     print("Accuracy per class: ", accuracy_per_class)

#     #now permut the test labels n times and calculate the mean accuracy and the standard deviation of the accuracy for the
#     #overall permuted accurcacy and the permuted accuracy for each class
#     permuted_accuracies = np.zeros(n_permutations)
#     permuted_accuracies_per_class = {label: np.zeros(n_permutations) for label in all_labels}
#     for i in range(n_permutations):
#         np.random.shuffle(labels_train)

#         permuted_predicted_labels = np.zeros(test_latent.shape[0])
#         for j in range(test_latent.shape[0]):
#             #get the labels of the nearest neighbours
#             labels_nearest_neighbours = labels_train[indices[j]]
#             #get the unique labels and their counts
#             unique_labels_nn, counts = np.unique(labels_nearest_neighbours, return_counts=True)
#             #calculate weighted counts
#             weighted_counts = np.zeros(len(unique_labels_nn))
#             for k in range(len(unique_labels_nn)):
#                 label = unique_labels_nn[k]
#                 weighted_counts[k] = counts[k] * weights[label]
#             #get the index of the label with the most counts
#             permuted_predicted_labels[j] = unique_labels_nn[np.argmax(weighted_counts)]
        
#         permuted_accuracies[i] = np.sum(permuted_predicted_labels == labels_test) / len(labels_test)

#         # Calculate and store permuted accuracy for each class that exists in this permutation
#         unique_labels_present_in_perm = np.unique(labels_test)
#         for label in unique_labels_present_in_perm:
#                 permuted_accuracies_per_class[label][i] = np.sum(permuted_predicted_labels[labels_test == label] == labels_test[labels_test == label]) / len(labels_test[labels_test == label])

#     mean_permuted_accuracies_per_class = {label: np.mean(permuted_accuracies_per_class[label]) for label in all_labels}
#     std_permuted_accuracies_per_class = {label: np.std(permuted_accuracies_per_class[label]) for label in all_labels}
    
#     mean_permuted_accuracy = np.mean(permuted_accuracies)
#     std_permuted_accuracy = np.std(permuted_accuracies)
    
#     #calculate the p-values for every individual emitter per class and the p-value for the total accuracy
#     count_higher_per_class = {label: np.sum(permuted_accuracies_per_class[label] >= accuracy_per_class[label]) for label in all_labels}
#     count_higher = np.sum(permuted_accuracies >= accuracy)
#     p_value_per_class = {label: count_higher_per_class[label] / n_permutations for label in all_labels}
#     p_value = count_higher / n_permutations

#     print("Mean permuted accuracy: ", mean_permuted_accuracy)
#     print("Std permuted accuracy: ", std_permuted_accuracy)
#     print("Mean permuted accuracies per class: ", mean_permuted_accuracies_per_class)
#     print("Std permuted accuracies per class: ", std_permuted_accuracies_per_class)
    
#     #now plot the accurcay per class and permuted accuracy per class in subplot 1 and the overall accuracy and permuted accuracy in subplot 2
#     plt.figure()
#     plt.suptitle('Emitter prediction based on hardcoded features',fontweight='bold')
#     plt.subplot(2, 1, 1)
#     normal_accuracies = [accuracy_per_class[label] for label in all_labels]
#     permuted_accuracies = [mean_permuted_accuracies_per_class[label] for label in all_labels]
#     permuted_std_errors = [std_permuted_accuracies_per_class[label] for label in all_labels]
#     x_labels = [str(label) for label in all_labels]
#     plt.bar(x_labels, normal_accuracies, yerr=[0]*len(normal_accuracies), label='Accuracy')
#     plt.bar(x_labels, permuted_accuracies, yerr=permuted_std_errors, alpha=0.9, label='Permuted Accuracy')
#     #for every bar in normal_accuracies, add a text with the p-value above the bar
#     for i, label in enumerate(all_labels):
#         if p_value_per_class[label] < 0.01:
#             plt.text(i, normal_accuracies[i] + 0.05, 'p < 0.01', ha='center', fontsize = 8, rotation=90)
#         else:
#             plt.text(i, normal_accuracies[i] + 0.05, 'p = {:.2f}'.format(p_value_per_class[label]), ha='center', fontsize = 8, rotation=90)
#     plt.title('Accuracy for each class')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.text(-0.1, 1.0, 'A', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
#     plt.tight_layout()
#     plt.ylim(0, 1)
#     emitter_library = {v: k for k, v in emitter_library.items()}
#     x_changed_labels = [emitter_library[label] for label in all_labels]
#     plt.xticks(range(len(x_labels)), x_changed_labels, rotation=90)
#     plt.grid()
#     plt.subplot(2, 1, 2)
#     plt.bar(['Accuracy', 'Permuted Accuracy'], [accuracy, mean_permuted_accuracy], yerr=[0, std_permuted_accuracy])
#     if p_value < 0.01:
#         plt.text(0.01, 0.95, 'p < 0.01', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
#     else:
#         plt.text(0.01, 0.95, 'p = {:.2f}'.format(p_value), transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
#     plt.title('Overall accuracy')
#     plt.ylabel('Accuracy')
#     plt.ylim(0, 1)
#     plt.xticks([0, 1], ['Accuracy', 'Permuted Accuracy'])
#     plt.grid()
#     plt.tight_layout()
#     plt.text(-0.1, 1.0, 'B', fontsize=16, fontweight='bold', va='top', ha='right', transform=plt.gca().transAxes)
#     # plt.text(0,-0.4 ,'Number of neighbours: {}'.format(n_neighbours))
#     #show which emitters was used in the bottom left corner from the name emitter from the input firt make emitter a string
#     plt.figtext(0.01, 0.01, 'Dataset used: {}'.format(string_to_print),  fontsize=8, verticalalignment='bottom')
    

#     plt.show()

# def get_explained_variance_features_across_model(spec, path_to_model, features, n_neighbours = 30, n_permutations =100, remove_double_specs = False):

#     hardcoded_feature1 = 'duration'
#     hardcoded_feature2 = 'mean_freq'
#     hardcoded_feature3 = 'min_freq'
#     hardcoded_feature4 = 'max_freq'
#     hardcoded_feature5 = 'bandwidth'
#     hardcoded_feature6 = 'starting_freq'
#     hardcoded_feature7 = 'stopping_freq'
#     hardcoded_feature8 = 'directionality'
#     hardcoded_feature9 = 'coefficient_of_variation'
#     hardcoded_feature10 = 'normalized_irregularity'
#     hardcoded_feature11 = 'local_variability'
#     hardcoded_feature12 = 'nr_of_steps_up'
#     hardcoded_feature13 = 'nr_of_steps_down'
#     hardcoded_feature14 = 'nr_of_peaks'
#     hardcoded_feature15 = 'nr_of_valleys'
    

#     features = np.array(features)
#     print('features shape =', features.shape)
#     #load the trained parameters used in training
#     parameters = joblib.load(path_to_model + '/parameters.pkl')

#     ##load spec_indices for train and test set
#     spec_indices_train = torch.load(path_to_model + '/spec_indices_train')
#     spec_indices_test = torch.load(path_to_model + '/spec_indices_test')

#     if remove_double_specs:
#         print('indices before removal of doubles =', len(spec_indices_train))
#         spec_indices_train_after_removal = []
#         for i in range(len(spec_indices_train)):
#             if spec_indices_train[i] not in spec_indices_train_after_removal:
#                 spec_indices_train_after_removal.append(spec_indices_train[i])
#         spec_indices_train = spec_indices_train_after_removal
#         print('indices after removal of doubles =', len(spec_indices_train_after_removal))

    

#     #get the spectograms
#     train_specs = []
#     test_specs = []

#     for i in spec_indices_train:
#         train_specs.append(spec[i])
#     for i in spec_indices_test:
#         test_specs.append(spec[i])

#     #get the features for the train and test set
#     features_train = []
#     features_test = []
#     for i in spec_indices_train:
#         features_train.append(features[i])
#     for i in spec_indices_test:
#         features_test.append(features[i])

#     features_train = np.array(features_train)
#     features_test = np.array(features_test)

#     #check if there are as many features as spectograms in the train and test sets
#     assert len(train_specs) == len(features_train), "Number of train spectograms and labels do not match"
#     assert len(test_specs) == len(features_test), "Number of test spectograms and labels do not match"

#     #pad the spectograms according to our training procedure
#     padding_length = 160
#     train_padded_specs = create_padded_spectograms.pad_spectograms_sorted(train_specs, padding_length)
#     test_padded_specs = create_padded_spectograms.pad_spectograms_sorted(test_specs, padding_length)

#     #delete unnecesarry big files in RAM
#     del spec, train_specs, test_specs

#     #normalize according to our training
#     if parameters['max_value_per_spec'] == 'true':
#         max_value_per_spec = True
#         print(max_value_per_spec)
#     train_padded_specs = normalize_padded_spectograms.normalize_specs(train_padded_specs, max_value_per_spec)
#     test_padded_specs = normalize_padded_spectograms.normalize_specs(test_padded_specs, max_value_per_spec)

#     #convert to torch tensors
#     latent_train_loader = torch.utils.data.DataLoader(train_padded_specs, batch_size=1, shuffle=False)
#     latent_test_loader = torch.utils.data.DataLoader(test_padded_specs, batch_size=1, shuffle=False)

#     #delete the padded spectograms from memory
#     del train_padded_specs, test_padded_specs

#     #initialize model
#     latent_space_size = parameters['latent_space_size']
#     slope_leaky = parameters['slope_leaky']
#     learning_rate = parameters['learning_rate']
#     precision_model = parameters['precision_model']
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = autoencoder_functions.VAE_KL_leaky(z_dim = latent_space_size, device_name='cuda',slope=slope_leaky,lr = learning_rate, model_precision =precision_model)
#     model = model.to(device)

#     #load the model
#     model_file_name = path_to_model + '/model.pt'
#     model.load_state_dict(torch.load(model_file_name))

#     train_latent = model.get_latent(latent_train_loader)
#     test_latent = model.get_latent(latent_test_loader)

#     print('train latent shape:',train_latent.shape)
#     print('test latent shape:',test_latent.shape) 
  
#     #get the nearest neighbours of the test set in the train set
#     neigh = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree')
#     neigh.fit(train_latent)
#     distances, indices = neigh.kneighbors(test_latent)
#     print("Indices shape: ", indices.shape)
#     print("Distances shape: ", distances.shape)

#     #get the features of the nearest neighbours
#     features_nearest_neighbours = features_train[indices]
#     print("Features nearest neighbours shape: ", features_nearest_neighbours.shape)

#     #determine the predicted features based on the nearest neighbours
#     predicted_features = np.zeros((test_latent.shape[0], features.shape[1]))
#     for i in range(test_latent.shape[0]):
#         #get the features of the nearest neighbours
#         features_nearest_neighbours = features_train[indices[i]]
#         #calculate the mean of the features of the nearest neighbours
#         predicted_features[i] = np.mean(features_nearest_neighbours, axis=0)

#     #assert if predicted features has the same shape as features test
#     assert predicted_features.shape == np.array(features_test).shape, "Number of predicted features and test features do not match"

#     #determine the rmse of the prediction for each feature and the total rmse
#     rmse_per_feature = np.sqrt(np.mean((predicted_features - features_test)**2, axis=0))
#     total_rmse = np.sqrt(np.mean((predicted_features - features_test)**2))

#     print("Total RMSE: ", total_rmse)
#     print("RMSE per feature: ", rmse_per_feature)

#     #now permut the test features n times and calculate the mean rmse and the standard deviation of the rmse for the
#     #overall permuted rmse and the permuted rmse for each feature
#     permuted_rmses = np.zeros(n_permutations)
#     permuted_rmses_per_feature = np.zeros((n_permutations, features.shape[1]))
#     for i in range(n_permutations):
#         np.random.shuffle(features_train)

#         permuted_predicted_features = np.zeros((test_latent.shape[0], features.shape[1]))
#         for j in range(test_latent.shape[0]):
#             #get the features of the nearest neighbours
#             features_nearest_neighbours = features_train[indices[j]]
#             #calculate the mean of the features of the nearest neighbours
#             permuted_predicted_features[j] = np.mean(features_nearest_neighbours, axis=0)
        
#         permuted_rmses[i] = np.sqrt(np.mean((permuted_predicted_features - features_test)**2))
#         permuted_rmses_per_feature[i] = np.sqrt(np.mean((permuted_predicted_features - features_test)**2, axis=0))
#     mean_permuted_rmses_per_feature = np.mean(permuted_rmses_per_feature, axis=0)
#     std_permuted_rmses_per_feature = np.std(permuted_rmses_per_feature, axis=0)

#     mean_permuted_rmse = np.mean(permuted_rmses)
#     std_permuted_rmse = np.std(permuted_rmses)

#     print("Mean permuted RMSE: ", mean_permuted_rmse)
#     print("Std permuted RMSE: ", std_permuted_rmse)
#     print("Mean permuted RMSEs per feature: ", mean_permuted_rmses_per_feature)
#     print("Std permuted RMSEs per feature: ", std_permuted_rmses_per_feature)

#     #determine the variance explained by the model for each feature and total variance explained (variance explained = 1 - (rmse / mean_permuted_rmse))
#     variance_explained_per_feature = 1 - (rmse_per_feature / mean_permuted_rmses_per_feature)
#     total_variance_explained = 1 - (total_rmse / mean_permuted_rmse)

#     #plot the variance explained per feature with error bars for the std of the permuted rmse per feature
#     plt.figure(figsize=(10,5))
#     #show total variance explained as a horizontal dotted line
#     plt.axhline(y=total_variance_explained*100, color='r', linestyle='--', label='Total Variance Explained: {:.2f}%'.format(total_variance_explained*100))
#     plt.bar(range(len(variance_explained_per_feature)), variance_explained_per_feature*100, yerr=std_permuted_rmses_per_feature / mean_permuted_rmses_per_feature, alpha=0.9, label='Variance Explained')
#     plt.xticks(range(len(variance_explained_per_feature)), [hardcoded_feature1, hardcoded_feature2, hardcoded_feature3, hardcoded_feature4, hardcoded_feature5, hardcoded_feature6, hardcoded_feature7, hardcoded_feature8, hardcoded_feature9, hardcoded_feature10, hardcoded_feature11, hardcoded_feature12, hardcoded_feature13, hardcoded_feature14, hardcoded_feature15], rotation=45)
#     plt.title('Variance Explained for each feature across latent space of model',fontweight='bold')
#     plt.xlabel('Features')
#     plt.ylabel('Variance Explained (%)')
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     # plt.text(-2.0,-30.1 ,'Model_name: {}'.format(path_to_model))
#     # plt.text(-2.0,-35.1 ,'Number of neighbours: {}'.format(n_neighbours))
#     plt.ylim(0, 100)
#     #show which model was used in the bottom left corner
#     plt.figtext(0.01, 0.01 ,'Model: {}'.format(path_to_model.split('/')[-1]), fontsize=8)
#     plt.show()