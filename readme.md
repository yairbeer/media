### Main features

1. Reading and arranging the data to a pandas Dataframe class.
    1. Subsampling for experimenting
2. Preprocessing:
    1. I ended up not parsing the date to get week and month because the train and test data
       are in following week. I only improved the hour by adding the minutes
    2. Create dummy variables for user tags
    3. Split IP into 3 numbers (split by '.')
    4. Change User_ID, Domain, Ad_slot_ID into frequency variables
    5. Factorizing User-Agent, Creative_ID, Key_Page_URL
    6. Count user tags
3. Stratified cross validation
    1. Optimize hyper-parameters (partly because of the time constraints)
    2. For each paramets finding best number of rounds
    3. Create Full prediction trainset
    4. Using n Monte Carlo experiments to find the standard deviation in the metric function
       and if neccesary measuring a small improvement.
    5. I use scale_pos_weight != 1. It gives a higher AUC but the probabilities
       are affected. In real life I wouldn't use it, only for this test.
4. Train the model on all the train data
5. Predict test results
6. Write results file
