#Load data for the autoencoder
#This pythone file will contain numerous functions to load data for the autoencoder. We want to load the data normally and with
#naive oversampling.
import torch
import joblib.numpy_pickle as joblib
import numpy as np

import torch
import joblib.numpy_pickle as joblib # Assuming joblib is installed and used elsewhere
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset # Import necessary classes

def load_data(input_data, labels_data, indices_after_padding, batch_size, oversample=False, nr_workers=0):
    """
    Load the data from the input and return it as a train and test loader
    that yields ONLY spectrograms, while providing separate label/index arrays for evaluation.
    """

    spectograms = np.array(input_data, dtype=np.float32)
    labels = np.array(labels_data, dtype=int)

    # Check data integrity
    if len(spectograms) != len(labels):
        raise ValueError("Input and labels must be the same size")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must be 0 or 1")
    
    # Make spectograms, labels, and indices tensors (original, full dataset)
    spectograms_tensor = torch.tensor(spectograms, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    indices_padding_tensor = torch.tensor(indices_after_padding, dtype=torch.int64)

    # Split the dataset into train, validation and test sets and save the train and test indexes
    total_size = len(spectograms_tensor)
    train_size = int(0.7 * total_size)
    validation_size = int(0.15 * total_size)
    test_size = total_size - train_size - validation_size
    
    # Use random permutations for splitting to avoid bias from initial ordering
    all_indices = np.arange(total_size)
    np.random.shuffle(all_indices) # Shuffle all indices
    
    train_indices_initial = all_indices[:train_size]
    valiadation_indices_initial = all_indices[train_size:train_size + validation_size]
    test_indices_initial = all_indices[train_size + validation_size:]

    #make sure the sizes are correct by summing them and see if they equal the total size
    assert len(train_indices_initial) + len(valiadation_indices_initial) + len(test_indices_initial) == total_size, "Sizes do not add up correctly"

    # --- Naive Oversampling Logic ---
    final_train_indices = train_indices_initial # Initialize with initial indices
    final_validation_indices = valiadation_indices_initial # Initialize with initial indices
    final_test_indices = test_indices_initial   # Initialize with initial indices

    if oversample:
        # --- Oversample for TRAIN set ---
        # Get labels for the initial train split to find minority class
        train_labels_initial = labels_tensor[train_indices_initial]
        
        num_0_train = torch.sum(train_labels_initial == 0).item()
        num_1_train = torch.sum(train_labels_initial == 1).item()
        
        if num_0_train < num_1_train:
            minority_class_train = 0
            majority_count_train = num_1_train
        else:
            minority_class_train = 1
            majority_count_train = num_0_train
        
        # Get original indices of the minority class within the initial train split
        minority_indices_train_in_original_dataset = train_indices_initial[np.where(train_labels_initial == minority_class_train)[0]]
        num_minority_train = len(minority_indices_train_in_original_dataset)
        
        if num_minority_train > 0 and num_minority_train < majority_count_train: # Only oversample if minority exists and is truly smaller
            num_samples_to_add_train = majority_count_train - num_minority_train
            duplicated_indices_train = np.random.choice(minority_indices_train_in_original_dataset, size=num_samples_to_add_train, replace=True)
            final_train_indices = np.concatenate((train_indices_initial, duplicated_indices_train))
        
        np.random.shuffle(final_train_indices) # Shuffle combined indices

        # # --- Oversample for TEST set (following your original logic) ---
        # test_labels_initial = labels_tensor[test_indices_initial]

        # num_0_test = torch.sum(test_labels_initial == 0).item()
        # num_1_test = torch.sum(test_labels_initial == 1).item()

        # if num_0_test < num_1_test:
        #     minority_class_test = 0
        #     majority_count_test = num_1_test
        # else:
        #     minority_class_test = 1
        #     majority_count_test = num_0_test
        
        # minority_indices_test_in_original_dataset = test_indices_initial[np.where(test_labels_initial == minority_class_test)[0]]
        # num_minority_test = len(minority_indices_test_in_original_dataset)

        # if num_minority_test > 0 and num_minority_test < majority_count_test:
        #     num_samples_to_add_test = majority_count_test - num_minority_test
        #     duplicated_indices_test = np.random.choice(minority_indices_test_in_original_dataset, size=num_samples_to_add_test, replace=True)
        #     final_test_indices = np.concatenate((test_indices_initial, duplicated_indices_test))
        
        # np.random.shuffle(final_test_indices) # Shuffle combined indices

    # --- Create NEW TensorDatasets for DataLoaders, containing ONLY SPECTROGRAMS ---
    # Select the spectrograms using the final (potentially oversampled) indices
    train_spectograms_for_loader = spectograms_tensor[final_train_indices]
    validation_spectograms_for_loader = spectograms_tensor[final_validation_indices]
    test_spectograms_for_loader = spectograms_tensor[final_test_indices]
    
    # Create new TensorDatasets, each containing only the spectrograms
    train_dataset_for_loader = TensorDataset(train_spectograms_for_loader)
    validation_dataset_for_loader = TensorDataset(validation_spectograms_for_loader)
    test_dataset_for_loader = TensorDataset(test_spectograms_for_loader)

    # Create the train and test loaders directly from these new datasets
    # They will now yield ONLY the spectrograms
    TrainLoader = DataLoader(train_dataset_for_loader, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nr_workers)
    ValidationLoader = DataLoader(validation_dataset_for_loader, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nr_workers)
    TestLoader = DataLoader(test_dataset_for_loader, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nr_workers)
    
    # --- Prepare labels and spectrogram indices for RETURN VALUES (for evaluation) ---
    # These will use the final (potentially oversampled) indices
    labels_train_final = labels_tensor[final_train_indices].numpy()
    labels_validation_final = labels_tensor[final_validation_indices].numpy()
    labels_test_final = labels_tensor[final_test_indices].numpy()
    spec_indices_train_final = indices_padding_tensor[final_train_indices].numpy()
    spec_indices_validation_final = indices_padding_tensor[final_validation_indices].numpy()
    spec_indices_test_final = indices_padding_tensor[final_test_indices].numpy()


    # Print shapes (now reflecting the single tensor yielded by the DataLoaders)
    print("TrainLoader will yield batches of (spectrograms,)")
    print(f"Number of batches in TrainLoader: {len(TrainLoader)}")
    print(f"Number of samples in TrainLoader (after oversampling if applied): {len(TrainLoader.dataset)}")

    print("\nValidationLoader will yield batches of (spectrograms,)")
    print(f"Number of batches in ValidationLoader: {len(ValidationLoader)}")
    print(f"Number of samples in ValidationLoader (after oversampling if applied): {len(ValidationLoader.dataset)}")
    
    print("\nTestLoader will yield batches of (spectrograms,)")
    print(f"Number of batches in TestLoader: {len(TestLoader)}")
    print(f"Number of samples in TestLoader (after oversampling if applied): {len(TestLoader.dataset)}")

    print(f"\nTrain indices shape (after oversampling if applied): {final_train_indices.shape}")
    print(f"Validation indices shape (after oversampling if applied): {final_validation_indices.shape}")
    print(f"Test indices shape (after oversampling if applied): {final_test_indices.shape}")
    print(f"Train labels shape (after oversampling if applied): {labels_train_final.shape}")
    print(f"Validation labels shape (after oversampling if applied): {labels_validation_final.shape}")
    print(f"Test labels shape (after oversampling if applied): {labels_test_final.shape}")
    print(f"Train spectrogram indices shape (after oversampling if applied): {spec_indices_train_final.shape}")
    print(f"Validation spectrogram indices shape (after oversampling if applied): {spec_indices_validation_final.shape}")
    print(f"Test spectrogram indices shape (after oversampling if applied): {spec_indices_test_final.shape}")

    return TrainLoader,ValidationLoader, TestLoader, final_train_indices,final_validation_indices, final_test_indices, labels_train_final,labels_validation_final, labels_test_final, spec_indices_train_final, spec_indices_validation_final,spec_indices_test_final

# def load_data(input,labels,indices_after_padding, batch_size, oversample=False):
#     """
#     Load the data from the input file and return it as a train and test loader
#     """
#     spectograms = np.array(spectograms, dtype=np.float32)
#     labels = np.array(labels, dtype=np.int64)
#     indices_after_padding = np.array(indices_after_padding, dtype=np.int64)

#     spectograms = np.array(spectograms, dtype=np.float32)
#     #check if size of input and labels are the same and that the labels are 0 or 1 (0 for WT and 1 for KO)
#     if len(spectograms) != len(labels):
#         raise ValueError("Input and labels must be the same size")
#     if not np.all(np.isin(labels, [0, 1])):
#         raise ValueError("Labels must be 0 or 1")
    
#     #make both the spectograms and labels and indices tensors
#     spectograms = torch.tensor(spectograms, dtype=torch.float32)
#     labels = torch.tensor(labels, dtype=torch.int64)
#     indices_after_padding = torch.tensor(indices_after_padding, dtype=torch.int64)

    # #create a dataset from the spectograms and labels
    # dataset = torch.utils.data.TensorDataset(spectograms, labels, indices_after_padding)

    # #split the dataset into train and test sets and save the train and test indexes
    # train_size = int(0.8 * len(spectograms))
    # test_size = len(spectograms) - train_size
    # train_indices = np.random.choice(len(spectograms), size=train_size, replace=False)
    # test_indices = np.setdiff1d(np.arange(len(spectograms)), train_indices)
    # #shuffle the test indices
    # np.random.shuffle(test_indices)
    # #create the train and test datasets
    # train_dataset = torch.utils.data.Subset(dataset, train_indices)
    # test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # #make the train and test indices into numpy arrays
    # train_indices = np.array(train_indices)
    # test_indices = np.array(test_indices)

    # #if oversample is true, oversample the train and test datasets for the minority class by duplicating the samples
    # if oversample:
    #     #check number of 0 and 1 labels in train_indices
    #     num_0 = torch.sum(labels[train_indices] == 0)
    #     num_1 = torch.sum(labels[train_indices] == 1)
    #     #check the minority class
    #     if num_0 < num_1:
    #         minority_class = 0
    #         majority_class = 1
    #     else:
    #         minority_class = 1
    #         majority_class = 0

    #     #get the indices of the minority class
    #     minority_indices = np.where(labels[train_indices] == minority_class)[0]
    #     #duplicate the samples of the minority class until the number of samples is equal to the majority class
    #     num_minority = len(minority_indices)
    #     num_majority = len(train_indices) - num_minority
    #     num_samples_to_add = num_majority - num_minority
    #     #get the indices of the minority class
    #     minority_indices = train_indices[minority_indices]
    #     #duplicate the samples of the minority class until the number of samples is equal to the majority class
    #     duplicated_indices = np.random.choice(minority_indices, size=num_samples_to_add, replace=True)
    #     #add the duplicated indices to the train indices
    #     train_indices = np.concatenate((train_indices, duplicated_indices))
    #     #shuffle the train indices
    #     np.random.shuffle(train_indices)
    #     #create the train dataset
    #     train_dataset = torch.utils.data.Subset(dataset, train_indices)

    #     #do the same for the test dataset
    #     #check number of 0 and 1 labels in test_indices
    #     num_0 = torch.sum(labels[test_indices] == 0)
    #     num_1 = torch.sum(labels[test_indices] == 1)
    #     #check the minority class
    #     if num_0 < num_1:
    #         minority_class = 0
    #         majority_class = 1
    #     else:
    #         minority_class = 1
    #         majority_class = 0
    #     #get the indices of the minority class
    #     minority_indices = np.where(labels[test_indices] == minority_class)[0]
    #     #duplicate the samples of the minority class until the number of samples is equal to the majority class
    #     num_minority = len(minority_indices)
    #     num_majority = len(test_indices) - num_minority
    #     num_samples_to_add = num_majority - num_minority
    #     #get the indices of the minority class
    #     minority_indices = test_indices[minority_indices]
    #     #duplicate the samples of the minority class until the number of samples is equal to the majority class
    #     duplicated_indices = np.random.choice(minority_indices, size=num_samples_to_add, replace=True)
    #     #add the duplicated indices to the test indices
    #     test_indices = np.concatenate((test_indices, duplicated_indices))
    #     #shuffle the test indices
    #     np.random.shuffle(test_indices)
    #     #create the test dataset
    #     test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # #extract the spectograms from the train and test datasets that are subsets
    # train_spectograms = [train_dataset[i][0] for i in range(len(train_dataset))]
    # test_spectograms = [test_dataset[i][0] for i in range(len(test_dataset))]

    # # #make the train and test spectograms into numpy arrays
    # # train_spectograms = np.array(train_spectograms, dtype=np.float32)
    # # test_spectograms = np.array(test_spectograms, dtype=np.float32)
    
    # #create the train and test loaders
    # TrainLoader = torch.utils.data.DataLoader(train_spectograms, batch_size=batch_size, shuffle=True, pin_memory=True)
    # TestLoader = torch.utils.data.DataLoader(test_spectograms, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # #create a list of labels for each index in the train and test sets
    # labels_train = [labels[i] for i in train_indices]
    # labels_test = [labels[i] for i in test_indices]

    # #make the labels into numpy arrays
    # labels_train = np.array(labels_train, dtype=np.int64)
    # labels_test = np.array(labels_test, dtype=np.int64)

    # #create a list of indices for each spectrogram index in the train and test sets
    # spec_indices_train = [indices_after_padding[i] for i in train_indices]
    # spec_indices_test = [indices_after_padding[i] for i in test_indices]

    # #make the indices into numpy arrays
    # spec_indices_train = np.array(spec_indices_train, dtype=np.int64)
    # spec_indices_test = np.array(spec_indices_test, dtype=np.int64)

    # # #print shapes of the train and test loaders
    # # print("TrainLoader shape: ", train_spectograms.shape)
    # # print("TestLoader shape: ", test_spectograms.shape)
    # # #print shapes of the train and test indices
    # # print("Train indices shape: ", train_indices.shape)
    # # print("Test indices shape: ", test_indices.shape)
    # # #print shapes of the train and test labels
    # # print("Train labels shape: ", labels_train.shape)
    # # print("Test labels shape: ", labels_test.shape)
    # # #print shapes of the train and test spectrogram indices
    # # print("Train spectrogram indices shape: ", spec_indices_train.shape)
    # # print("Test spectrogram indices shape: ", spec_indices_test.shape)

    # #return the train and test loaders and the train and test indices
    # return TrainLoader, TestLoader, train_indices, test_indices, labels_train, labels_test, spec_indices_train, spec_indices_test
