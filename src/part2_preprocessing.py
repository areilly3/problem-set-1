'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd
import os
import numpy as np

# Load csv's into df's
pred_universe = pd.read_csv('data/pred_universe_raw.csv')
arrest_events = pd.read_csv('data/arrest_events_raw.csv')

# Perform outer join to create new df
df_arrests = pd.merge(pred_universe, arrest_events, how= 'outer', on= 'person_id')

class PreProcessing:

    # Create new column 'y' 
    def y_column():
        """
        Creates a new column 'y' that gets its value depending on if an arrested individual was arrested for a felony within a year of their inital arrest. 
        Args: none
        Returns: none
        """
        
        # Ensures that arrest_date_event is in the proper format
        df_arrests['arrest_date_event'] = pd.to_datetime(df_arrests['arrest_date_event'])
        
        # Creates 'y' column in df_arrests
        df_arrests['y'] = 0
        felony_arrest = df_arrests[df_arrests['charge_degree'] == 'felony'].copy()

        # Checks if an individual was rearrested for a felony within 1 year of inital arrest. 
        for i, row in df_arrests.iterrows():
            person_id = row['person_id']
            arrest_date = row['arrest_date_event']
            
            future_arrest = felony_arrest[
                (felony_arrest['person_id'] == person_id) &
                (felony_arrest['arrest_date_event'] > arrest_date)&
                (felony_arrest['arrest_date_event'] <= arrest_date + pd.Timedelta(days= 365))
            ]   
            
            # Assigns new value to 'y' if rearrest within 1 year was true. 
            if len(future_arrest) > 0:
                df_arrests.at[i, 'y'] = 1

        # Gets proportion of rearrests within a year
        rearrest_count = (df_arrests['y'] == 1).sum()
        total_rearrest = len(df_arrests['y'])
        rearrest_proportion = (rearrest_count / total_rearrest)
        print(f'What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year? {rearrest_proportion}')

    def predictive_feature():
        """
        Predictive feature to check if current arrest is for a felony
        Args: none
        Returns: none
        """
        df_arrests['current_charge_felony'] = np.where(df_arrests['charge_degree'] == 'felony', 1, 0)
        current_felony_arrest = (df_arrests['current_charge_felony'] == 1).sum()
        total_current_charge_felony = len(df_arrests['current_charge_felony'])
        current_felony_arrest_proportion = (current_felony_arrest / total_current_charge_felony)
        print(f'What share of current charges are felonies? {current_felony_arrest_proportion}')

    def num_fel_arrests_last_year_column():
        """
        Creates a new column 'num_fel_arrests_last_year' that checks how many felony arrests an individual had the year prior to their current arrest.
        Args: none
        Returns: none 
        """
        
        # Ensures 'arrest_date_event' is in the proper format
        df_arrests['arrest_date_event'] = pd.to_datetime(df_arrests['arrest_date_event'])
        
        # Sets value of 'num_fel_arrests_last_year' column
        df_arrests['num_fel_arrests_last_year'] = 0 
        felony_arrest = df_arrests[df_arrests['charge_degree'] == 'felony'].copy()

        # Checks for total felony arrests within the last year of the current arrest. 
        for i, row in df_arrests.iterrows():
            person_id = row['person_id']
            arrest_date = row['arrest_date_event']
            
            prior_felony_total = felony_arrest[
                (felony_arrest['person_id'] == person_id) &
                (felony_arrest['arrest_date_event'] < arrest_date)&
                (felony_arrest['arrest_date_event'] >= arrest_date - pd.Timedelta(days= 365))
            ]   
            
            df_arrests.at[i, 'num_fel_arrests_last_year'] = len(prior_felony_total)
        
        # Equation to determine avg number of felony arrests in the last year
        total_felony_arrest = df_arrests['num_fel_arrests_last_year'].sum()
        total_people_arrested = len(df_arrests['num_fel_arrests_last_year'])
        avg_fel_arrest_last_year = (total_felony_arrest / total_people_arrested)
        print(f'What is the average number of felony arrests in the last year? {avg_fel_arrest_last_year}')

    def cleanup():
        pred_universe['num_fel_arrests_last_year'] = df_arrests['num_fel_arrests_last_year']
        print(f'Mean num_fel_arrests_last_year: {pred_universe['num_fel_arrests_last_year'].mean()}')
        print(pred_universe.head())

        df_arrests.to_csv('data/df_arrests.csv')