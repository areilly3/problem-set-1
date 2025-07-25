'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

class DecisionTree:
    def decision_tree_model():
        """
        Creates/runs a decision tree model
        Args: none
        Returns: none
        """
        
        # Reads in training and testing CSV's
        df_arrests_test = pd.read_csv('data/df_arrests_test.csv')
        df_arrests_train = pd.read_csv('data/df_arrests_train.csv')

        features = ['num_fel_arrests_last_year', 'current_charge_felony']

        param_grid_dt = {'max_depth': [3, 5, 7]}

        # Initializes models
        dt_model = DTC()
        gs_cv_dt = GridSearchCV(estimator= dt_model, param_grid= param_grid_dt, cv= 5)
        
        print(df_arrests_train.head())

        # Runs model
        features_train = df_arrests_train[features]
        y_train = df_arrests_train['y']
        gs_cv_dt.fit(features_train, y_train)

        print(f'What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? {gs_cv_dt.best_params_}')

        # Creates new column 'pred_dt'
        features_test = df_arrests_test[features]
        pred_dt = gs_cv_dt.predict(features_test)

        df_arrests_test['pred_dt'] = pred_dt


