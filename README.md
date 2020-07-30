# 1C-Company-Sales-Prediction
A challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company, 
is used to predict total sales for every product and store in the next month.

Daily historical sales data is provided. The task is to forecast the total amount of products sold in every shop for the test set. Note that the list of shops and products slightly changes every month. Creating a robust model that can handle such situations is part of the challenge.

File descriptions :

    sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
    test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
    sample_submission.csv - a sample submission file in the correct format.
    items.csv - supplemental information about the items/products.
    item_categories.csv  - supplemental information about the items categories.
    shops.csv- supplemental information about the shops.
    
Data fields :

    ID - an Id that represents a (Shop, Item) tuple within the test set
    shop_id - unique identifier of a shop
    item_id - unique identifier of a product
    item_category_id - unique identifier of item category
    item_cnt_day - number of products sold. You are predicting a monthly amount of this measure
    item_price - current price of an item
    date - date in format dd/mm/yyyy
    date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
    item_name - name of item
    shop_name - name of shop
    item_category_name - name of item category
    
Submissions are evaluated by root mean squared error (RMSE). True target values are clipped into [0,20] range.

This dataset has approximately 3E6 rows and several features from which several other useful features are necessary to achieve a good RMSE score. For that reason, a typical laptop may not suffice for local runs. This analysis was debugged locally and refined on a VM instance with 16cpus/64GB RAM on GoogleCloud.

The key elements of this analysis are the feature interactions, features based on trends over time, encoded features, and lag features. XGBoost was used in the validation and prediction. The hyperparameters are roughly optimal but could be further tuned with RandomizedSearchCV and GridsearchCV.

The final RMSE score produced is approx. 0.92.

Link to Kaggle compettition:
https://www.kaggle.com/c/competitive-data-science-final-project/overview

