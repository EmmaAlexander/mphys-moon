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
REPEAT_ACCURACY = False #Run 20 times and calculate average accuracy
LINUX = True #Use linux file paths
USE_GPU = True #Use a GPU
WANDB = True # Log to weights and biases.

TITLE = f"{'XGBoost' if XGBOOST else 'Random Forest'} {'Eye' if not MULTI_LABEL_METHOD and not MULTI_OUTPUT_METHOD else ''}{'Multi-label' if MULTI_LABEL_METHOD else ''}{'Multi-output' if MULTI_OUTPUT_METHOD else ''} visibility"

PARAMS = {'learning_rate': 0.5, 'max_depth': 2, 'n_estimators': 50}

icouk_data_file = '..\\Data\\icouk_sighting_data_with_params.csv'
icop_data_file = '..\\Data\\icop_ahmed_2020_sighting_data_with_params.csv'
alrefay_data_file = '..\\Data\\alrefay_2018_sighting_data_with_params.csv'
allawi_data_file = '..\\Data\\schaefer_odeh_allawi_2022_sighting_data_with_params.csv'
yallop_data_file = '..\\Data\\Data/yallop_sighting_data_with_params.csv'

if LINUX:
    icouk_data_file = '../Data/icouk_sighting_data_with_params.csv'
    icop_data_file = '../Data/icop_ahmed_2020_sighting_data_with_params.csv'
    alrefay_data_file = '../Data/alrefay_2018_sighting_data_with_params.csv'
    allawi_data_file = '../Data/schaefer_odeh_allawi_2022_sighting_data_with_params.csv'
    yallop_data_file = '../Data/yallop_sighting_data_with_params.csv'

icouk_data = pd.read_csv(icouk_data_file)
icop_data = pd.read_csv(icop_data_file)
alrefay_data = pd.read_csv(alrefay_data_file)
allawi_data = pd.read_csv(allawi_data_file)
yallop_data = pd.read_csv(yallop_data_file)


data = pd.concat([icouk_data,icop_data,alrefay_data,yallop_data])

#Drop index, dependent parameters (q value etc) and visibility scale
data = data.drop(["Index","q","W","q'","W'","Visibility","Source"], axis = 1)

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

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80/20 training/test split

    model = select_model()
    model = model.set_params(**config)
    # Fitting takes the input and "truth" data for classification purposes
    model.fit(x_train, y_train)

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

sweep_id = wandb.sweep(sweep_config, project="moon_visibility_xgboost")
wandb.agent(sweep_id, train, count=200)
wandb.finish()