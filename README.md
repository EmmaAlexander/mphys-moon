# mphys-moon: Modeling the visibility of the new crescent moon
This folder contains the code and data relation to an MPhys project about using machine learning techniques to model the visibility of the new crescent moon.

# Layout:
## Code folder
The Code folder contains a set of visibility techniques. Yallop's criterion is used in "moon_visibility_Yallop.py" to calculate Yallop's 'q' value from lon-lat coordinates, the file also contains functions to plot maps of the visibility from the 'q' value as contour plots across the globe in various formats. "moon_visibility_ML.ipynb" trains a selection of machine learning models from the scikit library on our dataset, and compares metrics from the models. The "moon_visibility_NN.ipynb" file uses a pytorch artifical neural network to predict moon visibilty, and produces metrics for the model.

The two files "moon_visibility_ML_WANDB.py" and "moon_visibility_NN_WANDB.py" are adaptions of the neural network and xgboost code to integrate with 'weights and biases'. 'Weights and biases' is a site that allows hyperparameter tuning by adjusting the values between runs and saving the results.

The "moon_sightings_visualisation.ipynb" file visualises the properties and trends of our combined and separate datasets across a number of plots.

The "moon_visibility_calculations.ipynb" file has a few uses. This file can generate a set of moon parameters over a 24 hour period for a given day, and save this as a CSV file. This data can then be displayed in various formats. Data can also be read in from a file and displayed. It can also convert sighting data in various formats into a standardised file for use in other notebooks.

### Data Pre-Processing
This folder contains code to scrape data from online. 'cloudcoverbatch.py'takes cloud cover data from weatheronline.com and generates the valid sightings to a file. Both of the ICOP data scrape files take sighting data from the ICOP website.

## Data
The data folder contains files of each data set used, and sets of data for generated longitude and latitude. Most importantly, this folder contains the combined data set and cloud data sets.

## Initial Resources
This folder contains material initialy given by our MPhys supervisor when starting the project. Essentially good background information.