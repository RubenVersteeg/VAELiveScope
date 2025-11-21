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
from collections import Counter


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
    # if not np.all(np.isin(labels, [0, 1])):
    #     raise ValueError("Labels must be 0 or 1")
    
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
        # Get labels for the initial train split
        train_labels_initial = labels_tensor[train_indices_initial]

        # Use Counter to get the count of each unique label
        label_counts = Counter(train_labels_initial.tolist())
        
        # Identify the majority class and its count
        # This works for any number of classes
        majority_class_train = max(label_counts, key=label_counts.get)
        majority_count_train = label_counts[majority_class_train]
        
        final_train_indices = train_indices_initial.tolist()
        
        # Iterate through all labels and oversample minority classes
        for label, count in label_counts.items():
            if count < majority_count_train:
                # Find the indices of the current minority class
                minority_indices = train_indices_initial[np.where(train_labels_initial.cpu().numpy() == label)[0]]
                
                # Calculate the number of samples to add
                num_samples_to_add = majority_count_train - count
                
                # Randomly select and duplicate indices from the minority class
                duplicated_indices = np.random.choice(minority_indices, size=num_samples_to_add, replace=True)
                
                # Add the new indices to the list
                final_train_indices.extend(duplicated_indices)

        # Convert the final list back to a numpy array and shuffle
        final_train_indices = np.array(final_train_indices)
        np.random.shuffle(final_train_indices)

        # # --- Oversample for TEST set ---
        # # As previously mentioned, oversampling the test set is generally not recommended
        # # for a realistic evaluation. However, to match your original request, the multi-class logic is applied here as well.
        # test_labels_initial = labels_tensor[test_indices_initial]
        
        # label_counts_test = Counter(test_labels_initial.tolist())
        # majority_count_test = max(label_counts_test.values())
        
        # final_test_indices = test_indices_initial.tolist()

        # for label, count in label_counts_test.items():
        #     if count < majority_count_test:
        #         minority_indices_test = test_indices_initial[np.where(test_labels_initial.cpu().numpy() == label)[0]]
        #         num_samples_to_add_test = majority_count_test - count
        #         duplicated_indices_test = np.random.choice(minority_indices_test, size=num_samples_to_add_test, replace=True)
        #         final_test_indices.extend(duplicated_indices_test)

        # final_test_indices = np.array(final_test_indices)
        # np.random.shuffle(final_test_indices)

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
    print(f"Test indices shape (after oversampling if applied): {final_test_indices.shape}")
    print(f"Validation indices shape (after oversampling if applied): {final_validation_indices.shape}")
    print(f"Train labels shape (after oversampling if applied): {labels_train_final.shape}")
    print(f"Validation labels shape (after oversampling if applied): {labels_validation_final.shape}")
    print(f"Test labels shape (after oversampling if applied): {labels_test_final.shape}")
    print(f"Train spectrogram indices shape (after oversampling if applied): {spec_indices_train_final.shape}")
    print(f"Validation spectrogram indices shape (after oversampling if applied): {spec_indices_validation_final.shape}")
    print(f"Test spectrogram indices shape (after oversampling if applied): {spec_indices_test_final.shape}")

    return TrainLoader, ValidationLoader, TestLoader, final_train_indices, final_validation_indices, final_test_indices, labels_train_final, labels_validation_final, labels_test_final, spec_indices_train_final, spec_indices_validation_final,spec_indices_test_final