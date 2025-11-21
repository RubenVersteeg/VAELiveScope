These scripts were used to create and analyse the model of my thesis. To recreate my results shank3USVs_many_properties.jl and emitter_and_specs.jl should be downloaded from the Radboud repository and added to this repository on your local machine.
Here I will give a step-wise explanation on how to recreate my results:

1. Run train_multiple_models_pups.py and train_multiple_models_shank3.py This will train 50 models for each script with each model taking 1 hour to train (Extensive workload)

2. Run Check_test_loss_vs_latent.ipynb, which will show Figure 2 of my thesis in the notebook.

After this a latent dimension of 8 was chosen so in both train_multiple_models_pups.py and train_multiple_models_shank3.py change the following variables:
  number_of_models_per_latent_size = 1
  latent_sizes = [8]
  base_folder = Path('final_models')

Run both these files again. This will train the final models, where analysis will be performed on. This step will approximately 2 hours. After this run:

3. Run from_vocalisation_to_latent_visualisation.ipynb to recreate Figure 3 and extract traditional features from every single spectrogram in both the Shank3 and pups datasets. These traditional features will be saved.

4. Run compare_latent_vs_hardcoded.ipynb to recreate Figure 4, which is created in the last cell of the notebook.

5. Run thesis_figures.ipynb to recreates Figures 5, 6, 7 and 8.

6. Run check_features.ipynb to recreate Figure 9

7. Run test_times.ipynb to test pipeline times. Results in thesis are with specified hardware in thesis.
