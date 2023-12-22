# mphys-moon: Using Machine Learning to Predict the First Visibility of the New Crescent Moon
This folder contains the code and data relating to an MPhys project that uses machine learning techniques to model the visibility of the new crescent Moon.

This work was performed by MPhys students Ezzy Cross and Neil Power, with supervision from Dr. Emma Alexander, at the University of Manchester.

# Layout:
## Code folder
The "moon_visibility_calculations.ipynb" file is a general file for calculating the properties of the Moon from sets of dates, latitudes and longitudes. This file can generate a set of moon parameters over a 24 hour period for a given day, and save this as a CSV file. This data can then be displayed in various formats. Data can also be read in from a file and displayed. It can also convert sighting data in various formats into a standardised file for use in other notebooks. The visibility using Yallop's criterion can also be plotted, either as a contour plot, globe plot or 3D globe plot.

The "moon_visibility_ML.ipynb" trains a selection of machine learning models from the scikit library on our dataset, and compares metrics from the models. The "moon_visibility_NN.ipynb" file uses an artificial neural network from the pytorch library to predict moon visibility, and also produces metrics for the model.

The two files "moon_visibility_ML_WANDB.py" and "moon_visibility_NN_WANDB.py" are adaptions of the neural network and ML notebooks to integrate with the Weights and Biases framework (wandb.ai). This allows hyperparameter tuning by adjusting the values between runs and saving the results.

The "moon_sightings_visualisation.ipynb" file visualises the properties and trends of our combined and separate datasets across a number of plots.

## Data
The data folder contains files of each data set used, and sets of data for generated longitude and latitude. Most importantly, this folder contains the combined data set and cloud data sets.

### Data Pre-Processing
This folder contains code to scrape data from online. 'cloudcoverbatch.py' takes cloud cover data from weatheronline.com and generates the valid sightings to a file. Both of the ICOP data scrape files take sighting data from the ICOP website.

## Initial Resources
This folder contains a workplan and starting material provided by our MPhys supervisor when starting the project.