import joblib.numpy_pickle as joblib
import os
import torch
import numpy as np
import VAELiveScope.create_padded_spectograms as create_padded_spectograms
import VAELiveScope.normalize_padded_spectograms as normalize_padded_spectograms
#first get training parameters for the model from the notebook and save them in a file
def save_parameters(spectograms_file,labels_file,batch_size,model_choice,name,latent_space_size,nr_epochs,amount_of_patience,slope_leaky,oversampling_method,learning_rate,precision_model,max_value_is_per_spec, removal_of_female):
    #save the parameters to a file in directory with name

    library_path = os.path.join(os.getcwd(), name)
    os.makedirs(library_path, exist_ok=True)
    #save the parameters to a file in directory with name
    parameters = {
        'spectograms_file': spectograms_file,
        'labels_file': labels_file,
        'batch_size': batch_size,
        'model_choice': model_choice,
        'name': name,
        'latent_space_size': latent_space_size,
        'nr_epochs': nr_epochs,
        'amount_of_patience': amount_of_patience,
        'slope_leaky': slope_leaky,
        'oversampling': oversampling_method,
        'learning_rate': learning_rate,
        'precision_model': precision_model,
        'max_value_per_spec': max_value_is_per_spec,
        'removale_of_female': removal_of_female

    }
    joblib.dump(parameters, os.path.join(library_path, "parameters.pkl"))

def save_parameters_pups(spectograms_file,labels_file,batch_size,model_choice,name,latent_space_size,nr_epochs,amount_of_patience,slope_leaky,oversampling_method,learning_rate,precision_model,max_value_is_per_spec, removal_of_mother):
    #save the parameters to a file in directory with name

    library_path = os.path.join(os.getcwd(), name)
    os.makedirs(library_path, exist_ok=True)
    #save the parameters to a file in directory with name
    parameters = {
        'spectograms_file': spectograms_file,
        'labels_file': labels_file,
        'batch_size': batch_size,
        'model_choice': model_choice,
        'name': name,
        'latent_space_size': latent_space_size,
        'nr_epochs': nr_epochs,
        'amount_of_patience': amount_of_patience,
        'slope_leaky': slope_leaky,
        'oversampling': oversampling_method,
        'learning_rate': learning_rate,
        'precision_model': precision_model,
        'max_value_per_spec': max_value_is_per_spec,
        'removale_of_mother': removal_of_mother

    }
    joblib.dump(parameters, os.path.join(library_path, "parameters.pkl"))



def save_model_files(model,name,train_indices,validation_indices,test_indices,labels_train,labels_validation,labels_test,train_loss,validation_loss,early_stopping_length,reconstruction_loss_train,reconstruction_loss_validation,KLD_loss_train,KLD_loss_validation,spec_indices_train, spec_indices_validation,spec_indices_test):
    library_path = os.path.join(os.getcwd(), name)
    os.makedirs(library_path, exist_ok=True)
    model_file_name = os.path.join(library_path, 'model.pt')
    torch.save(model.state_dict(), model_file_name)
    torch.save(model.state_dict(), model_file_name)
    torch.save(train_loss, os.path.join(library_path, 'train_loss'))
    torch.save(validation_loss, os.path.join(library_path, 'test_loss'))
    torch.save(reconstruction_loss_train, os.path.join(library_path, 'reconstruction_loss_train'))
    torch.save(reconstruction_loss_validation, os.path.join(library_path, 'reconstruction_loss_test'))
    torch.save(KLD_loss_train, os.path.join(library_path, 'KLD_loss_train'))
    torch.save(KLD_loss_validation, os.path.join(library_path, 'KLD_loss_test'))
    torch.save(train_indices, os.path.join(library_path, 'train_indices'))
    torch.save(validation_indices, os.path.join(library_path, 'validation_indices'))
    torch.save(test_indices, os.path.join(library_path, 'test_indices'))
    torch.save(labels_train, os.path.join(library_path, 'labels_train'))
    torch.save(labels_validation, os.path.join(library_path, 'labels_validation'))
    torch.save(labels_test, os.path.join(library_path, 'labels_test'))
    torch.save(early_stopping_length, os.path.join(library_path, 'early_stopping_length'))
    torch.save(spec_indices_train, os.path.join(library_path, 'spec_indices_train'))
    torch.save(spec_indices_validation, os.path.join(library_path, 'spec_indices_validation'))
    torch.save(spec_indices_test, os.path.join(library_path, 'spec_indices_test'))
    #get the encoder state of the model
    encoder_state = model.get_state_encoder()
    encoder_state.keys()
    #save the encoder state
    encoder_file_name = os.path.join(library_path, 'encoder_state.pt')
    torch.save(encoder_state, encoder_file_name)

def visualize_test_data(model,spectograms_file, labels, test_indices,name, max_value_per_spec=True):
    #visualize 10 samples of test data. Labels need to be provided to let all the function that pad work correctly.
    #max_value_per_spec is a boolean that indicates if the spectograms should be normalized by the maximum value per
    #spectogram and needs to be set to the same value as in the training.
    padded_spectograms = []
    while len(padded_spectograms) != 10:
        indices = np.random.choice(test_indices, 10, replace=False).astype(int)
            #load the spectograms for these indices
        all_spectograms = joblib.load(spectograms_file)
        labels = labels
        specs = [all_spectograms[i] for i in indices]
        for i in range(len(specs)):
            specs[i] = specs[i]['spec']

        padded_specs, _, _, _= create_padded_spectograms.pad_specs(specs, 160, labels, removal_of_female=False)
        padded_spectograms = normalize_padded_spectograms.normalize_specs(padded_specs, max_value_per_spec=max_value_per_spec)
        #check if there are still 10 padded spectograms, if not rerun the code
        if len(padded_spectograms) < 10:
            print(f'Not enough padded spectograms, found {len(padded_spectograms)}, rerunning code...')
            continue

    print(f'Found {len(padded_spectograms)} padded test spectograms, using them for visualization.')


    spectograms = padded_spectograms
    spectograms = torch.tensor(spectograms, dtype=torch.float32).to('cuda')
    #visualize the spectograms
    model_file_name = name + '/model.pt'
    model.load_state_dict(torch.load(model_file_name))
    with torch.no_grad():
        m, _, _ = model.encode(spectograms)
        _, _, rec = model.forward(spectograms, return_latent_rec=True)

    n = np.size(m,1) / 8
    n = int(n)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(10, 3, figsize=(10,20))
    for i in range(10):
        axs[i,0].imshow(spectograms[i].cpu().numpy(), cmap='viridis',origin='lower')
        axs[i,1].imshow(m[i].cpu().numpy().reshape(n,8), cmap='viridis',origin='lower')
        axs[i,2].imshow(rec[i], cmap='viridis',origin='lower')
        plt.savefig(name + '/test_data_visualization.png')
    # #choose 10 random indices from test_indices
    # indices = np.random.choice(test_indices, 10, replace=False).astype(int)
    # #load the spectograms for these indices
    # spectograms = joblib.load(spectograms_file)
    # specs = [spectograms[i] for i in indices]
    
    
    

    # spectograms = spectograms[indices]
    # spectograms = torch.tensor(spectograms, dtype=torch.float32).to('cuda')
    # #visualize the spectograms
    # model_file_name = name + '/model.pt'
    # model.load_state_dict(torch.load(model_file_name))
    # with torch.no_grad():
    #     m, _, _ = model.encode(spectograms)
    #     _, _, rec = model.forward(spectograms, return_latent_rec=True)

    # n = np.size(m,1) / 8
    # n = int(n)
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(10, 3, figsize=(10,20))
    # for i in range(10):
    #     axs[i,0].imshow(spectograms[i].cpu().numpy(), cmap='viridis',origin='lower')
    #     axs[i,1].imshow(m[i].cpu().numpy().reshape(n,8), cmap='viridis',origin='lower')
    #     axs[i,2].imshow(rec[i], cmap='viridis',origin='lower')

def visualize_test_data_pups(model,spectograms_file, labels, test_indices,name, max_value_per_spec=True):
    #visualize 10 samples of test data. Labels need to be provided to let all the function that pad work correctly.
    #max_value_per_spec is a boolean that indicates if the spectograms should be normalized by the maximum value per
    #spectogram and needs to be set to the same value as in the training.
    padded_spectograms = []
    while len(padded_spectograms) != 10:
        indices = np.random.choice(test_indices, 10, replace=False).astype(int)
            #load the spectograms for these indices
        all_spectograms = joblib.load(spectograms_file)
        labels = labels
        specs = [all_spectograms[i] for i in indices]
        for i in range(len(specs)):
            specs[i] = specs[i][1]

        padded_specs, _, _ = create_padded_spectograms.pad_specs(specs, 160, labels, removal_of_female=False)
        padded_spectograms = normalize_padded_spectograms.normalize_specs(padded_specs, max_value_per_spec=max_value_per_spec)
        #check if there are still 10 padded spectograms, if not rerun the code
        if len(padded_spectograms) < 10:
            print(f'Not enough padded spectograms, found {len(padded_spectograms)}, rerunning code...')
            continue

    print(f'Found {len(padded_spectograms)} padded test spectograms, using them for visualization.')


    spectograms = padded_spectograms
    spectograms = torch.tensor(spectograms, dtype=torch.float32).to('cuda')
    #visualize the spectograms
    model_file_name = name + '/model.pt'
    model.load_state_dict(torch.load(model_file_name))
    with torch.no_grad():
        m, _, _ = model.encode(spectograms)
        _, _, rec = model.forward(spectograms, return_latent_rec=True)

    n = np.size(m,1) / 8
    n = int(n)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(10, 3, figsize=(10,20))
    for i in range(10):
        axs[i,0].imshow(spectograms[i].cpu().numpy(), cmap='viridis',origin='lower')
        axs[i,1].imshow(m[i].cpu().numpy().reshape(n,8), cmap='viridis',origin='lower')
        axs[i,2].imshow(rec[i], cmap='viridis',origin='lower')
        plt.savefig(name + '/test_data_visualization.png')