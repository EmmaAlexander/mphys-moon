import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# MACHINE LEARNING IMPORTS
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

#WANDB IMPORTS
import wandb
from wandb.xgboost import WandbCallback

# %% [markdown]
# ## Options and setup

# %%
#METHOD = False # replace seen column with method seen column
MULTI_OUTPUT_METHOD = False #Replace naked eye seen column with array of methods
MULTI_LABEL_METHOD = False #Replace naked eye seen column with either seen, visual aid or not seen
XGBOOST = True #Use xgboost forest or random forest
RANDOM = False # replace data with random arrays
CLOUDCUT = False # cut all complete cloud cover data points
REPEAT_ACCURACY = False #Run 20 times and calculate average accuracy
LINUX = True #Use linux file paths
USE_GPU = True #Use a GPU
WANDB = True # Log to weights and biases.

TITLE = f"{'XGBoost' if XGBOOST else 'Random Forest'} {'Eye' if not MULTI_LABEL_METHOD and not MULTI_OUTPUT_METHOD else ''}{'Multi-label' if MULTI_LABEL_METHOD else ''}{'Multi-output' if MULTI_OUTPUT_METHOD else ''} visibility"

# %% [markdown]
# ## Hyperparameters

# %%
NOTSWEEPING = False
if NOTSWEEPING:
    if XGBOOST:
        if MULTI_OUTPUT_METHOD:
            PARAMS = {'learning_rate': 0.1733, 'max_depth': 6, 'n_estimators': 50}
        elif MULTI_LABEL_METHOD:
            PARAMS = {'learning_rate': 0.4183, 'max_depth': 2, 'n_estimators': 100}
        else:
            PARAMS = {'learning_rate': 0.5, 'max_depth': 2, 'n_estimators': 50}
            
    else: #Random forest
        if MULTI_OUTPUT_METHOD:
            PARAMS = {}
        elif MULTI_LABEL_METHOD:
            PARAMS = {'max_depth': 10, 'max_features': 'log2', 'n_estimators': 50}
        else:
            PARAMS = {'max_depth': 12, 'n_estimators': 150}

    wandb.init(
        # set the wandb project where this run will be logged
        project="moon_visibility_xgboost",
        config=PARAMS)
else:
    wandb.init(
        # set the wandb project where this run will be logged
        project="moon_visibility_xgboost")

# %%
icouk_data_file = '..\\Data\\icouk_sighting_data_with_params.csv'
icop_data_file = '..\\Data\\icop_ahmed_2020_sighting_data_with_params.csv'
alrefay_data_file = '..\\Data\\alrefay_2018_sighting_data_with_params.csv'
allawi_data_file = '..\\Data\\schaefer_odeh_allawi_2022_sighting_data_with_params.csv'

if LINUX:
    icouk_data_file = 'mphys-moon/Data/icouk_sighting_data_with_params.csv'
    icop_data_file = 'mphys-moon/Data/icop_ahmed_2020_sighting_data_with_params.csv'
    alrefay_data_file = 'mphys-moon/Data/alrefay_2018_sighting_data_with_params.csv'
    allawi_data_file = 'mphys-moon/Data/schaefer_odeh_allawi_2022_sighting_data_with_params.csv'

icouk_data = pd.read_csv(icouk_data_file)
icop_data = pd.read_csv(icop_data_file)
alrefay_data = pd.read_csv(alrefay_data_file)
allawi_data = pd.read_csv(allawi_data_file)

data = pd.concat([icouk_data,icop_data,alrefay_data,allawi_data])

#Drop index, dependent parameters (q value etc) and visibility scale
data = data.drop(["Index","q","W","q'","W'","Visibility"], axis = 1)

if MULTI_OUTPUT_METHOD:
    data = data.drop(["Seen", "Method"], axis = 1) # replaced by methods column

    ptype = ["Seen_eye", "Seen_binoculars", "Seen_telescope", "Seen_ccd","Not_seen"]

elif MULTI_LABEL_METHOD:
    data = data.drop(["Seen", "Methods"], axis = 1) # replaced by method column
    data["Method"] = data["Method"].replace("Seen_binoculars", "Seen_with_aid")
    data["Method"] = data["Method"].replace("Seen_telescope", "Seen_with_aid")
    data["Method"] = data["Method"].replace("Seen_ccd", "Seen_with_aid")

    ptype = ["Seen_eye", "Seen_with_aid", "Not_seen"]

else:
    data = data[data["Method"] !="Seen_binoculars"] #DROP BINOCULARS
    data = data[data["Method"] !="Seen_ccd"] #DROP CCD
    data = data[data["Method"] !="Seen_telescope"] #DROP TELESCOPE
    
    data=data.drop(['Method','Methods'], axis = 1) #Only use seen

    ptype = ["Seen", "Not_seen"]

if RANDOM:
    data.insert(1,"Random1",np.random.rand(data.shape[0],1))
    data.insert(2,"Random2",np.random.rand(data.shape[0],1))
    data = data[['Seen', "Random1","Random2"]]

if CLOUDCUT:
    data = data[data["Cloud Level"] <= 0.5]
    #data = data[data["Cloud Level"] == 0]

# %%
# List of features without label feature
variable_list =  data.columns.tolist()
features = variable_list


if MULTI_OUTPUT_METHOD:
    orig_y = np.array(data['Methods'].str.split(";"))
    mlb = MultiLabelBinarizer(classes=ptype)
    y = mlb.fit_transform(orig_y)
    features.remove('Methods')

elif MULTI_LABEL_METHOD:
    data["Method"] = data["Method"].replace("Seen_eye",2) #XGboost needs 1 and 0
    data["Method"] = data["Method"].replace("Seen_with_aid",1)
    data["Method"] = data["Method"].replace("Not_seen",0)
    y = np.array(data['Method'])
    features.remove('Method')
    
else:
    if XGBOOST:
        data["Seen"] = data["Seen"].replace("Seen",1) #XGboost needs 1 and 0
        data["Seen"] = data["Seen"].replace("Not_seen", 0)
    y = np.array(data['Seen'])
    features.remove('Seen')
    pos = 'Seen' # for ROC curve +ve result

X = data[features]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80/20 training/test split

# %%
# Produce randomforest classifier model and fit to training data
def select_model():
    if XGBOOST:
        if MULTI_OUTPUT_METHOD:
            if USE_GPU:
                method = "gpu_hist"
            else:
                method = "hist"
            model = XGBClassifier(tree_method=method, n_jobs=-1)
        else:
            model = XGBClassifier(n_jobs=-1, callbacks=[WandbCallback()])
    else:
        if MULTI_OUTPUT_METHOD:
            model = MultiOutputClassifier(RandomForestClassifier(n_jobs=-1))
        else:
            model = RandomForestClassifier(n_jobs=-1)

    model = model.set_params(**wandb.config)
    return model
model = select_model()
# Fitting takes the input and "truth" data for classification purposes
model.fit(x_train, y_train)

# %%
def get_easiest_method_array(methods):
    easiest_methods = np.zeros(methods.shape)
    easiest_methods[np.arange(0,methods.shape[0],1),np.argmax(methods,axis=1)] = 1
    return easiest_methods

def get_easiest_method_names(methods):
    easiest_methods = get_easiest_method_array(methods)
    return mlb.inverse_transform(easiest_methods.astype(int))

# Produce predictions for the classification of your training dataset using your model:
y_pred = model.predict(x_train)

# plot the accuracies of said predictions
print("Accuracy on training dataset:",accuracy_score(y_train, y_pred))
rf_acc_train = accuracy_score(y_train, y_pred)
y_pred = model.predict(x_test)

print("Accuracy on testing dataset:", accuracy_score(y_test, y_pred))
rf_acc_test = accuracy_score(y_test, y_pred)

wandb.log({"Accuracy": rf_acc_test})

if MULTI_OUTPUT_METHOD:
    print("Accuracy on testing dataset (easiest method only):", accuracy_score(get_easiest_method_names(y_test), get_easiest_method_names(y_pred)))


# %% [markdown]
# ## ROC curve

# %%
# Get predicted class probabilities for the test set 
if MULTI_OUTPUT_METHOD:
    #y_pred_prob = rf.predict_proba(x_test)
    #roc_auc = roc_auc_score(y_test, y_pred_prob)
    print("Not currently working")
elif MULTI_LABEL_METHOD:
    y_pred_prob = model.predict_proba(x_test)
    roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class="ovr")
    print(f"ROC curve {roc_auc}")
    wandb.log({"ROC curve" :roc_auc})
else:
    y_pred_prob = model.predict_proba(x_test)[:, 1] 
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC curve {roc_auc}")
    wandb.log({"ROC curve" :roc_auc})

# %%
#wandb.log({'learning_rate': wandb.config["learning_rate"], 'max_depth': wandb.config["max_depth"], 'n_estimators': wandb.config["n_estimators"]})
wandb.finish()