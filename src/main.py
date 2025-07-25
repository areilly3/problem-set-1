'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
from part1_etl import ETL
from part2_preprocessing import PreProcessing
from part3_logistic_regression import LogisticRegression
from part4_decision_tree import DecisionTree

# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    etl = ETL.etl()

    # PART 2: Call functions/instanciate objects from preprocessing
    y_column = PreProcessing.y_column()
    predictive_feature = PreProcessing.predictive_feature()
    num_fel_arrest_last_year_column = PreProcessing.num_fel_arrests_last_year_column()
    cleanup = PreProcessing.cleanup()

    # PART 3: Call functions/instanciate objects from logistic_regression
    logistic_regression = LogisticRegression.logistic_regression_model()

    # PART 4: Call functions/instanciate objects from decision_tree
    decision_tree = DecisionTree.decision_tree_model()

    # PART 5: Call functions/instanciate objects from calibration_plot


if __name__ == "__main__":
    main()