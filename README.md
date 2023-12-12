# mphys-moon: Modeling the visibility of the new crescent moon
This folder contains the code and data relation to an MPhys project about using machine learning techniques to model the visibility of the new crescent moon.

# Layout:
## Code folder
The Code folder contains a set of visibility techniques. Yallop's criterion is used in "moon_visibility_Yallop.py" to calculate Yallop's 'q' value from lon-lat coordinates, the file also contains functions to plot maps of the visibility from the 'q' value as contour plots across the globe in various formats. "moon_visibility_ML.ipynb" trains a selection of machine learning models from the scikit library on our dataset, and compares metrics from the models. The "moon_visibility_NN.ipynb" file uses a pytorch artifical neural network to predict moon visibilty, and produces metrics for the model.

The two files "moon_visibility_ML_WANDB.py" and "moon_visibility_NN_WANDB.py" are adaptions of the neural network and xgboost code to integrate with 'weights and biases'. 'Weights and biases' is a site that allows hyperparameter tuning by adjusting the values between runs and saving the results.

The "moon_sightings_visualisation.ipynb" file visualises the properties and trends of our combined and separate datasets across a number of plots.
