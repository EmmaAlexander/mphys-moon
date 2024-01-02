import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #Issue with xgboost & new pandas version

import pandas as pd
import numpy as np

# MACHINE LEARNING IMPORTS
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

#WANDB IMPORTS
import wandb
wandb.login()

sweep_config = {
    "method": "bayes", # try grid or random
    "metric": {
      "name": "Accuracy",
      "goal": "maximize"   
    },
    "parameters": {
        "max_depth": {
            "distribution": "int_uniform",
            "max": 10,
            "min": 1
        },
        "learning_rate": {
            "distribution": "uniform",
            "max": 0.5,
            "min": 0.01
        },
        "n_estimators": {
            "distribution": "int_uniform",
            "max": 200,
            "min": 10
        }
    }
}

"""
        "max_bin": {
            "distribution": "int_uniform",
            "max": 500,
            "min": 100
        }
        "min_split_loss": {
            "distribution": "uniform",
            "max": 5,
            "min": 0
        },
        "subsample": {
            "distribution": "uniform",
            "max": 1,
            "min": 0.1
        },
                "max_leaves": {
            "distribution": "int_uniform",
            "max": 5,
            "min": 0
        },
                "reg_alpha": {
            "distribution": "uniform",
            "max": 5,
            "min": 0
        },
                "reg_lambda": {
            "distribution": "uniform",
            "max": 5,
            "min": 0
        },

"""

#METHOD = False # replace seen column with method seen column
MULTI_OUTPUT_METHOD = False #Replace naked eye seen column with array of methods
MULTI_LABEL_METHOD = False #Replace naked eye seen column with either seen, visual aid or not seen
XGBOOST = True #Use xgboost forest or random forest
RANDOM = False # replace data with random arrays
CLOUDCUT = False # cut all complete cloud cover data points
LINUX = False #Use linux file paths
USE_GPU = True #Use a GPU
WANDB = True # Log to weights and biases.

TITLE = f"{'XGBoost' if XGBOOST else 'Random Forest'} {'Eye' if not MULTI_LABEL_METHOD and not MULTI_OUTPUT_METHOD else ''}{'Multi-label' if MULTI_LABEL_METHOD else ''}{'Multi-output' if MULTI_OUTPUT_METHOD else ''} visibility"

PARAMS = {'learning_rate': 0.2, 'max_depth': 6, 'n_estimators': 50}

data_file = 'Data/moon_sighting_data.csv'

data = pd.read_csv(data_file)

#Drop index, dependent parameters (q value etc) and visibility scale
data = data.drop(["Index","q","W","q'","Source"], axis = 1)

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

if CLOUDCUT:
    data = data[data["Cloud Level"] <= 0.5]
    #data = data[data["Cloud Level"] == 0]

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

def select_model():
    if XGBOOST:
        if MULTI_OUTPUT_METHOD:
            if USE_GPU:
                method = "gpu_hist"
            else:
                method = "hist"
            model = XGBClassifier(tree_method=method, n_jobs=-1)
        else:
            model = XGBClassifier(n_jobs=-1)
    else:
        if MULTI_OUTPUT_METHOD:
            model = MultiOutputClassifier(RandomForestClassifier(n_jobs=-1))
        else:
            model = RandomForestClassifier(n_jobs=-1)
    return model


def train():
    config_defaults = PARAMS
    wandb.init(config=config_defaults)  # defaults are over-ridden during the sweep
    config = wandb.config

    def traintestml(X,y, rf):
        accuracy_arr = []
        roc_arr = []
        for i in range(10):
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            # Fitting takes the input and "truth" data for classification purposes
            rf.fit(x_train, y_train)
            # Produce predictions for the classification of your training dataset using your model:
            y_pred = rf.predict(x_test)
            #print("Accuracy on testing dataset:",accuracy_score(y_test, y_pred))

            accuracy_arr.append(accuracy_score(y_test, y_pred))
            if MULTI_LABEL_METHOD:
                y_pred_prob = model.predict_proba(x_test)
                roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class="ovr")
            else:
                y_pred_prob = model.predict_proba(x_test)[:, 1] 
                roc_auc = roc_auc_score(y_test, y_pred_prob)
            roc_arr.append(roc_auc)

        accuracy_avg = np.mean(accuracy_arr)
        accuracy_std = np.std(accuracy_arr)
        roc_avg = np.mean(roc_arr)
        roc_std = np.std(roc_arr)
        return accuracy_avg,accuracy_std,roc_avg,roc_std

    model = select_model()
    model = model.set_params(**config)
    accuracy_avg,accuracy_std,roc_avg,roc_std=traintestml(X,y, model)
    wandb.log({"Accuracy": accuracy_avg,"Accuracy std": accuracy_std,"ROC":roc_avg,"ROC std": roc_std})

sweep_id = wandb.sweep(sweep_config, project="moon_visibility_xgboost")
wandb.agent(sweep_id, train, count=200)
wandb.finish()