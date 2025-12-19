import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib.numpy_pickle as joblib
import torch
import torch.nn as nn
from pathlib import Path

# Import your custom modules
import seperate_spec_and_features
import make_labels
import create_padded_spectograms
import normalize_padded_spectograms
import load_usv_data_pups
import autoencoder_functions
import save_model

# --- Global Configuration (can be outside the main block) ---
batch_size = 64
model_choice = 'KL_leaky'
oversampling_method = 'normal'
method_is_oversample = (oversampling_method == 'oversample')

max_value_is_per_spec = 'true'
max_value_per_spec = (max_value_is_per_spec.lower() == 'true')

removal_of_mother_and_females_is_done = 'true'
removal_of_mother_and_females = (removal_of_mother_and_females_is_done.lower() == 'true')

learning_rate = 1e-3
precision_model = 10.0

slope_leaky = 0.1
nr_epochs = 1000
amount_of_patience = 25
number_of_models_per_latent_size = 1
latent_sizes = [8]

# --- Main script execution starts here ---
if __name__ == "__main__":
    print(f"CUDA is available: {torch.cuda.is_available()}")

    # --- Load Data ---
    spectograms_file_pups = 'emitter_and_specs.jl'
    usvs_pups = joblib.load(spectograms_file_pups)
    (spec_pups, emitter_pups )= seperate_spec_and_features.seperate_spec_and_emitter_pups(usvs_pups)

    spectograms_file_shank3 = 'shank3USVs_many_properties.jl'
    usvs_shank3 = joblib.load(spectograms_file_shank3)
    (
        spec_shank3,
        emitter_shank3,
        distance_to_nearest_non_emitter,
        duration,
        start_time_in_recording,
        end_time_in_recording,
        intensities,
        intensity,
        location,
        main_freq,
        recording_path,
    ) = seperate_spec_and_features.separate_spec_and_features(usvs_shank3)

    # --- print shapes of data ---
    print(f"Number of pup specs: {len(spec_pups)}")
    print(f"Number of shank3 specs: {len(spec_shank3)}")
    print(f"Total number of specs: {len(spec_pups) + len(spec_shank3)}")
    print(f"Number of pup emitters: {len(emitter_pups)}")
    print(f"Number of shank3 emitters: {len(emitter_shank3)}")
    print(f"Total number of emitters: {len(emitter_pups) + len(emitter_shank3)}")

    # --- Combine Data ---
    spec = spec_pups + spec_shank3
    emitter = emitter_pups + emitter_shank3

    # --- print shapes of combined data ---
    print(f"Combined number of specs: {len(spec)}")
    print(f"Combined number of emitters: {len(emitter)}")


    labels, emitter_library = make_labels.make_labels(emitter,  label_type='emitter')
    padded_spec, original_indices, indices_after_padding, labels_after_padding = create_padded_spectograms.pad_specs_combined(spec, 160, labels, removal_of_mother_and_females=removal_of_mother_and_females)
    normalized_padded_spec = normalize_padded_spectograms.normalize_specs(padded_spec, max_value_per_spec=max_value_per_spec)

    # --- Clean up unnecessary variables for memory ---
    del spec, emitter,  spec_shank3,emitter_shank3,distance_to_nearest_non_emitter,duration,start_time_in_recording,end_time_in_recording,intensities,intensity,location,main_freq,recording_path,
    del labels, padded_spec, original_indices,spec_pups, emitter_pups 

    # --- Main Training and Saving Loop ---
    base_folder = Path('final_models')


    for j in range(number_of_models_per_latent_size):
        for i in latent_sizes:
            # 1. Create the specific sub-folder for this model run
            sub_folder = base_folder / f'shank3_pups_combined_{i}_v{j}'
            sub_folder.mkdir(parents=True, exist_ok=True)
            print(f"Working in folder: {sub_folder.resolve()}")

            # 2. Get the file paths for saving
            labels_file = sub_folder / 'labels.pkl'
            parameters_file = sub_folder / 'parameters.pkl'
            model_file = sub_folder / 'model_files.pkl'
            plot_file = sub_folder / f'Loss_figure_shank3_{i}_v{j}.png'

            latent_space_size = i
            (TrainLoader, ValidationLoader, TestLoader, train_indices, validation_indices, test_indices, labels_train, labels_validation, labels_test, spec_indices_train, spec_indices_validation,spec_indices_test) = load_usv_data_pups.load_data(normalized_padded_spec,labels_after_padding,indices_after_padding,batch_size=batch_size, oversample=method_is_oversample, nr_workers=9)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            VAE_model = autoencoder_functions.VAE_KL_leaky(
                z_dim=latent_space_size, device_name='cuda', slope=slope_leaky,
                lr=learning_rate, model_precision=precision_model
            )
            VAE_model = VAE_model.to(device)

            # Train the model
            (
                train_loss, validation_loss, early_stopping_length,
                reconstruction_loss_train, reconstruction_loss_validation,
                KLD_loss_train, KLD_loss_validation
            ) = VAE_model.train_loop(
                {'train': TrainLoader, 'test': ValidationLoader},
                epochs=nr_epochs, test_freq=1, save_freq=10, vis_freq=1,
                patience=amount_of_patience
            )
            
            # Save model parameters and files
            save_model.save_parameters_combined(spectograms_file_shank3,spectograms_file_pups,labels_file,batch_size,model_choice,sub_folder,latent_space_size,nr_epochs,amount_of_patience,slope_leaky,oversampling_method,learning_rate,precision_model, max_value_is_per_spec,removal_of_mother_and_females_is_done)

            
            parameters = joblib.load(parameters_file)
            
            save_model.save_model_files(
                VAE_model, sub_folder, train_indices, validation_indices, test_indices,
                labels_train, labels_validation, labels_test, train_loss, validation_loss,
                early_stopping_length, reconstruction_loss_train,
                reconstruction_loss_validation, KLD_loss_train, KLD_loss_validation,
                spec_indices_train, spec_indices_validation, spec_indices_test
            )
            
            # Plot validation reconstruction loss and KLD loss
            plt.figure(figsize=(10, 5))
            plt.plot(reconstruction_loss_validation, label='Validation Reconstruction Loss')
            plt.plot(KLD_loss_validation, label='Validation KLD Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Loss for Latent Space {i}, Run {j}')
            plt.legend()
            
            # Save the plot
            plt.savefig(plot_file)
            plt.close()
            
            # Save the emitter library
            joblib.dump(emitter_library, sub_folder / 'emitter_library.pkl')
            
            # Delete .tar files
            for file in Path('.').glob('*.tar'):
                file.unlink()
                print(f'Removed file: {file}')