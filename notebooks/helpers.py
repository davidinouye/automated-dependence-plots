import sys
import os
import urllib
import hashlib
import pickle
import numpy as np
import scipy.stats
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer 

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F    
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Create and train models
def create_and_train_models(model_names, dtypes, X_train, y_train, X_test=None, y_test=None, cache_folder=None):
    if cache_folder is None:
        cache_folder = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_dicts = np.array([
        create_model(model_name, dtypes, random_state=0)
        for model_name in model_names
    ])
    for d in model_dicts:
        ########### Handle pretrained model cache ############
        # Create hash bytestring from parameters of model
        #  and the input dataset
        string_to_hash = b',\n'.join([
            b'model_name=', bytes(d['model_name'], 'ascii'), 
            b'dtypes=', bytes(str(dtypes), 'ascii'),
            b'random_staet=', b'0',
            b'X_train=', X_train.tostring(),
            b'y_train=', y_train.tostring()])
        # Create a trained model hash string and filename
        model_hash = hashlib.sha1(string_to_hash).hexdigest()
        cache_filename = 'cached_model_' + model_hash + '.pkl'
        full_cache_filename = os.path.join(cache_folder, cache_filename)
        if os.path.isfile(full_cache_filename):
            # Load model from cache
            print('Loading fitted model from cache via file %s' % cache_filename)
            with open(full_cache_filename, 'rb') as f:
                fitted_estimator = pickle.load(f)
            d['estimator'] = fitted_estimator
        else:
            print('Training estimator because fitted model could not be load from the cache via file %s' % cache_filename)
            d['estimator'].fit(X_train, y_train)
            # Save model into cache
            def ensure_dir(file_path):
                directory = os.path.dirname(file_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
            ensure_dir(full_cache_filename)
            with open(full_cache_filename, 'wb+') as f:
                pickle.dump(d['estimator'], f)
        ########### End of handling pretrained model cache ############
        d['train_score'] = accuracy_score(d['estimator'].predict(X_train), y_train)
        if X_test is not None and y_test is not None:
            d['test_score'] = accuracy_score(d['estimator'].predict(X_test), y_test)
        d['model'] = d['get_model'](d['estimator'])
    return model_dicts

def create_model(model_name, dtypes, random_state=0):
    if model_name == 'RBFSVM':
        base_estimator = SVC(random_state=random_state)
        param_grid = {
            'svc__C': np.logspace(-1, 3, 10),
            'svc__gamma': np.logspace(-4, 1, 10),
        }
        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state)
        def get_model(estimator):
            def model(X):
                func = estimator.decision_function
                if np.asarray(X).ndim == 1:
                    return func(X.reshape(1, -1))[0]
                return func(X)
            return model
    elif model_name == 'RBFSVM-2':
        base_estimator = SVC(random_state=random_state)
        param_grid = {
            'svc__C': np.logspace(-1, 1, 10),
            'svc__gamma': np.logspace(-3, 1, 10),
        }
        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state)
        def get_model(estimator):
            def model(X):
                func = estimator.decision_function
                if np.asarray(X).ndim == 1:
                    return func(X.reshape(1, -1))[0]
                return func(X)
            return model
    elif model_name == 'LogisticRegression':
        base_estimator = LogisticRegression()
        param_grid = {
            'logisticregression__C': np.logspace(-4, -1, 100),
        }
        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state)
        def get_model(estimator):
            def model(X):
                func = estimator.decision_function
                if np.asarray(X).ndim == 1:
                    return func(X.reshape(1, -1))[0]
                return func(X)
            return model
    elif model_name == 'DNN':
        base_estimator = _DNNEstimator(max_epoch=1000, lr=1e-4, batch=1000, random_state=random_state)
        param_grid = {
            '_dnnestimator__batch': [100, 200, 400],
        }
        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state)
        def get_model(estimator):
            def dnn_model(X):
                func = estimator.decision_function
                if np.asarray(X).ndim == 1:
                    p = func(X.reshape(1, -1))[0]
                else:
                    p = func(X)
                return _logit(p)
            return dnn_model
    elif model_name == 'DecisionTree':
        base_estimator = DecisionTreeClassifier(random_state=random_state)
        param_grid = {
            'decisiontreeclassifier__max_leaf_nodes': [5, 10, 20, 40],
            'decisiontreeclassifier__max_depth': np.arange(10) + 1,
        }
        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state)
        def get_model(estimator):
            def decision_tree_model(X):
                func = estimator.predict_proba
                if np.asarray(X).ndim == 1:
                    p = func(X.reshape(1, -1))[0][1]
                else:
                    p = func(X)[:, 1]
                return _logit(p)
            return decision_tree_model
    elif model_name == 'GradientBoost':
        base_estimator = GradientBoostingClassifier(random_state=random_state)
        param_grid = {
            'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
            #'gradientboostingclassifier__max_depth': [1, 2, 3, 4, 5],
            'gradientboostingclassifier__n_estimators': [25, 50, 100, 200, 500],#[1, 2, 3, 4, 5],
        }
        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state)
        def get_model(estimator):
            def gradient_boost_model(X):
                func = estimator.decision_function
                if np.asarray(X).ndim == 1:
                    return func(X.reshape(1, -1))[0]
                return func(X)
            return gradient_boost_model
    elif model_name == 'RandomForest':
        base_estimator = RandomForestClassifier(max_features=None, random_state=random_state)

        param_grid = {
            'randomforestclassifier__max_depth': [4, 5, 6],
            'randomforestclassifier__n_estimators': [100, 200, 400],#[1, 2, 3, 4, 5],
        }

        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state)
        def get_model(estimator):
            def random_forest_model(X):
                func = estimator.predict_proba
                if np.asarray(X).ndim == 1:
                    p = func(X.reshape(1, -1))[0][1]
                else:
                    p = func(X)[:, 1]
                return _logit(p)
            return random_forest_model
    elif model_name == 'RandomForest-2':
        base_estimator = RandomForestClassifier(max_features=None, random_state=2)
        param_grid = {
            'randomforestclassifier__max_depth': [4, 5, 6],
            'randomforestclassifier__n_estimators': [100, 200, 400],#[1, 2, 3, 4, 5],
        }

        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state)
        def get_model(estimator):
            def random_forest_model(X):
                func = estimator.predict_proba
                if np.asarray(X).ndim == 1:
                    p = func(X.reshape(1, -1))[0][1]
                else:
                    p = func(X)[:, 1]
                return p
            return random_forest_model
    else:  ##############
        raise ValueError('Could not recognize "%s" model name.' % model_name)

    return dict(estimator=estimator, get_model=get_model, model_name=model_name)


def fetch_1980_data():
    # Download data if needed
    filename = os.path.join(os.path.dirname(__file__), '..', 'data', 'Data_1980.csv')
    #Original data: https://www.icpsr.umich.edu/icpsrweb/NACJD/studies/8987/version/1/datadocumentation
    if not os.path.isfile(filename):
        print('Downloading data to %s' % filename)
        urllib.request.urlretrieve(
            ('https://raw.githubusercontent.com/marcotcr/anchor-experiments'
             '/master/datasets/recidivism/Data_1980.csv'),
            filename)
        
    # Load data and setup dtypes
    col_names_category = [
        'WHITE', 'ALCHY', 'JUNKY', 'SUPER', 'MARRIED',
        'FELON', 'WORKREL', 'PROPTY', 'PERSON',
        'MALE', 
        'RECID', 'FILE']

    col_names_numerical = ['SCHOOL', 'PRIORS', 'RULE',
        'AGE', 'TSERVD', 'FOLLOW','TIME'
        ]

    df = pd.read_csv(filename)

    for col in df:
        if col in col_names_category:
            df[col] = df[col].astype('category')
        else:
            df[col] = df[col].astype(float)

    original_dataframe = df.copy()

    # Preprocessing
    #filter out data with file number = 3; remove the TIME column so that the last column is prediction
    df = df.drop('TIME', axis = 1)
    df = df[df.FILE != 3]
    df.AGE = df.AGE/12  # Convert to years
    df = df[df.PRIORS != -9]
    df = df.drop('FILE', axis = 1)

    def get_numpy(df):
        new_df = df.copy()
        cat_columns = new_df.select_dtypes(['category']).columns
        new_df[cat_columns] = new_df[cat_columns].apply(lambda x: x.cat.codes)
        return new_df.values
    dtypes = df.dtypes[:-1]
    Xy = get_numpy(df)
    X = Xy[:,:-1]
    y = Xy[:,-1]
    feature_labels = df.columns.values[:-1]  # Last is prediction

    feature_labels[np.where(feature_labels =='AGE')[0][0]] = 'AGE (Year)'
    feature_labels[np.where(feature_labels =='SUPER')[0][0]] = 'PAROLE'
    feature_labels[np.where(feature_labels =='FELON')[0][0]] = 'FELONY'

    return dict(X=X, y=y, dtypes=dtypes, feature_labels=feature_labels,
                dataframe=df, original_dataframe=original_dataframe)


def fetch_german_data():
    # Download data if needed
    filename = os.path.join(os.path.dirname(__file__), '..', 'data', 'german.data')
    if not os.path.isfile(filename):
        print('Downloading data to %s' % filename)
        urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                                   filename)
        
    # Load data and setup dtypes
    col_names = [
        'checkin_acc', 'duration', 'credit_history', 'purpose', 'amount',
        'saving_acc', 'present_emp_since', 'inst_rate', 'personal_status',
        'other_debtors', 'residing_since', 'property', 'age',
        'inst_plans', 'housing', 'num_credits',
        'job', 'dependents', 'telephone', 'foreign_worker', 'status']
    df = pd.read_csv(filename, delimiter=' ', header=None, names=col_names)
    for k, v in _german_loan_attribute_map.items():
        df.replace(k, v, inplace=True)
    for col in df:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        else:
            df[col] = df[col].astype(float)

    def get_numpy(df):
        new_df = df.copy()
        cat_columns = new_df.select_dtypes(['category']).columns
        new_df[cat_columns] = new_df[cat_columns].apply(lambda x: x.cat.codes)
        return new_df.values
    Xy = get_numpy(df)
    X = Xy[:,:-1]
    y = Xy[:,-1]
    # Make 1 (good customer) and 0 (bad customer)
    # (Originally 2 is bad customer and 1 is good customer)
    sel_bad = y == 2  
    y[sel_bad] = 0
    y[~sel_bad] = 1
    feature_labels = df.columns.values[:-1]  # Last is prediction
    dtypes = df.dtypes[:-1]
    return dict(X=X, y=y, dtypes=dtypes, feature_labels=feature_labels, dataframe=df)


def create_constant_model(x0, model):
    # Comparison to constant model
    def constant_model(X): # Constant model
        if np.asarray(X).ndim == 1:
            return model(x0)
        return np.ones(X.shape[0]) * model(x0)
    return constant_model


###############
# Private helper functions
###############

def _logit(p):
    p = np.minimum(1-1e-7, np.maximum(p, 1e-7))
    assert np.all(p < 1) and np.all(p > 0)
    return np.log(p/(1-p))


def _is_categorical(dtypes):
    # Copied from adp.funcs module for convenience
    def check_dtype(dtype):
        if pd.api.types.is_categorical_dtype(dtype):
            return True
        try:
            dtype.categories
        except AttributeError:
            try:
                dtype['categories']
            except TypeError:
                return False     
            except KeyError:
                return False
            else:
                return True
        else:
            return True
    return np.array([check_dtype(dtype) for dtype in dtypes])


def _create_pipe(estimator, dtypes):
    categories = [np.arange(len(dtype.categories)) for dtype in dtypes[_is_categorical(dtypes)]]
    one_hot = OneHotEncoder(sparse=False, categories=categories)
    return make_pipeline(
        ColumnTransformer([("One_Hot_Encoder", one_hot, _is_categorical(dtypes))], remainder="passthrough"),
        StandardScaler(),
        estimator,
    )


def _create_cv_pipe(estimator, param_grid, dtypes, random_state=0):
    pipe = _create_pipe(estimator, dtypes)
    cv = StratifiedKFold(5)
    return GridSearchCV(pipe, param_grid, scoring='accuracy', cv=cv, refit=True)


#######################
# Deep neural network
#######################
class _NN(nn.Module):
    def __init__(self, input_dim, num_class):
        super(_NN, self).__init__()
        self.input_dim = input_dim
        self.num_class = num_class
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_class)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def _nn_train(model, dataset, max_epoch=5000, lr=0.01, batch=64, seed=None):
    # create your optimizer
    if seed is not None:
        torch.manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for n_epochs in range(max_epoch):
        loss_vals = 0.0
        for i_batch, sample_batched in enumerate(dataloader):
            optimizer.zero_grad()   # zero the gradient buffers
            input, target = sample_batched
            output = model(input.float())
            one_hot_targets = torch.from_numpy(np.eye(2)[target]).float()
            loss = criterion(output, one_hot_targets)
            loss.backward()
            optimizer.step()
            loss_vals += loss.item()
        if n_epochs % 100 == 0:
            print('ep%d\tloss:%f'%(n_epochs, loss_vals))
            
    #torch.save(model.state_dict(), 'dnn.pt')
    

class _CreateDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return (torch.from_numpy(self.x[idx,:]), torch.from_numpy(np.array(self.y[idx], dtype=np.int)))
    

class _DNNEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, max_epoch=5000, lr=0.01, batch=64, random_state=None):
        self.max_epoch = max_epoch
        self.lr = lr
        self.batch = batch
        self.random_state = random_state
    
    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)
        torch.manual_seed(rng.randint(2**32-1))
        
        self.classes_ = np.unique(y)
        model = _NN(n_features, len(self.classes_))
        dataset = _CreateDataset(X, y)
        _nn_train(model, dataset, max_epoch=self.max_epoch, 
                  lr=self.lr, batch=self.batch)
        self.model_ = model
        return self
    
    def predict(self, X):
        return self.classes_[np.argmax(self.model_(torch.as_tensor(X).float()).detach().numpy(), axis=1)]
    
    def decision_function(self, X):
        return self.model_(torch.as_tensor(X).float()).detach().numpy().astype(np.double)[:, 1]

# Linux command line to create dictionary for attributes
#cat temp.txt | grep ":" | grep -v Attribute | grep -v Attibute | sed "s/ : /='/g" | sed "s/ $/',/g" | sed 's/ <= ... < /-/g' | sed 's/\.\.\. //g' | sed 's/\.\. //g'
_german_loan_attribute_map = dict(
    A11='< 0 DM',
    A12='0-200 DM',
    A13='>= 200 DM',
    A14='no checking',
    A30='no credits',
    A31='all credits paid back',
    A32='existing credits paid back',
    A33='delayed past payments',
    A34='critical account',
    A40='car (new)',
    A41='car (used)',
    A42='furniture/equipment',
    A43='radio/television',
    A44='domestic appliances',
    A45='repairs',
    A46='education',
    A47='(vacation?)',
    A48='retraining',
    A49='business',
    A410='others',
    A61='< 100 DM',
    A62='100-500 DM',
    A63='500-1000 DM',
    A64='>= 1000 DM',
    A65='unknown/no sav acct',
    A71='unemployed',
    A72='< 1 year',
    A73='1-4 years',
    A74='4-7 years',
    A75='>= 7 years',
    A91='male & divorced',
    A92='female & divorced/married',
    A93='male & single',
    A94='male & married',
    A95='female & single',
    A101='none',
    A102='co-applicant',
    A103='guarantor',
    A121='real estate',
    A122='life insurance',
    A123='car or other',
    A124='unknown/no property',
    A141='bank',
    A142='stores',
    A143='none',
    A151='rent',
    A152='own',
    A153='for free',
    A171='unskilled & non-resident',
    A172='unskilled & resident',
    A173='skilled employee',
    A174='management/self-employed',
    A191='no telephone',
    A192='has telephone',
    A201='foreigner',
    A202='non-foreigner',
)


def fetch_mnist_data():
    from torchvision import datasets
    data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    datasets.MNIST(data_root, train=True, download=True)


if __name__ == '__main__':
    # Download example data
    fetch_1980_data()
    fetch_german_data()
    fetch_mnist_data()
