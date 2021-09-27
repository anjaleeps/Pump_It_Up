# Pump It Up Challenge Submission
This repository provides a machine learning-based solution to the Pump It Up: Data Mining the Water Table challenge on the Driven Data platform. The solution carries out an initial analysis of the dataset, identifies important aspects that could impact a predictive model, and implements a machine learning classification model to predict the operational status of waterpoints spanned across Tanzania. 

## Problem Overview
The purpose of the data mining challenge run by the Driven Data organization is to identify which water pumps are functional, nonfunctional, or need repairing based on a 39-feature dataset, made available by Taarifa and Tanzanian Ministry of Water. 
The feature set of the dataset records the basic and vital information about each waterpoint, including its location, management, and inception. The functional status, provided by the label, “status_group”, is divided into three categories: functional, nonfunctional, and functional needs repair. In the training dataset, the data points corresponding to each status group are as follows. 
<ul>
  <li>Functional: 32259</li>
  <li>Nonfunctional: 22824</li>
  <li>Functional needs repair: 4317</li>
</ul>
As evident here, the training dataset contains imbalanced classes with the “functional needs repair” class having a significantly lower number of data points. 

## Training Data Analysis
Among the 39 available features in the dataset, we can identify interesting patterns and similarities between features and their impact on the status group outcome. Columns such as amount_tsh, construction_year, and gps_height contain many data points marked as 0 instead of being marked as missing in the original dataset. Values of features such as waterpoint type, quantity, payment, district_code, region, and basin have a clear impact on the functional status of a water pump. For example, when considering waterpoint type, communal standpipes are more likely to be functional than the communal standpipe multiple types. Similarly, waterpoints with enough water quantity are more likely to be functional than those that are in the dry range. 

When analyzing separate features in the dataset, some features that are more or less similar to each other can be seen in several instances as listed below.
<ul>
  <li>Waterpoint_type_group/waterpoint_type</li>
  <li>Quantity/quantity_group</li>
  <li>Payment/payment_type</li>
  <li>Extraction type/extraction_type_group/extracction_type_group</li>
  <li>Management/management_group</li>
  <li>Scheme/scheme_name</li>
  <li>Water_quality/quality_group</li>
  <li>Source/source_type/source_class</li>
  <li>Region/region_code</li>
</ul>

These grouped features present in the dataset often represent more or less similar values and value counts as can be seen in the above plots. Therefore, picking only one feature to represent each group would be sufficient for building the model. It allows the elimination of a total of 10 features from the dataset, bringing down the number of features to 29.

The dataset also uses several classification methods to record the area each waterpoint is located in: region, region_code, district_code, lga, ward, subvillage. Since preserving all these records would be redundant when training the model, excessive features region_code and ward were chosen to be removed from the training and test datasets. 
The number of features used for model training can be further reduced by removing some features that don’t have an impact on the final outcome of each datapoint. They are wpt_name (name of the waterpoint) and recorded_by (since all data were recorded by one party). 

## Cleaning Data

Categorical data values in the dataset, such as funder and installer, contain refers to the same entity but with lowercase/uppercase differences in the value. Therefore, such columns are converted to lowercase to clean the dataset. 

## Handle Missing Data

In the training dataset, the missing data counts are as follows. 
funder: 3635
installer: 3635
subvillage: 371
public_meeting: 3334
scheme_management: 3877
permit: 3056

These missing data were handled by adding an additional value “not known” to indicate the records that had missing data points. 
In addition, several other fields had used 0 instead of meaningful numerical values. These data should also be considered as missing data in this exercise. As shown in the above plots, these fields are: Construction_year, Amount_tsh, Gps_height

The zero values in these fields were first replaced by “NaN” to allow for treating them as missing data. For filling missing values in the construction year, first, the median of data points grouped in the order of district_code -> region -> scheme_management were taken. To fill the points that didn’t receive a numerical value with this mechanism the median of data grouped by district_code -> region were taken. Still existing missing data were filled by the median of data grouped by district_code. Finally, the median of the entire construction_year column was used to fill the still missing values. 

Similarly, missing values in other columns were filled with a similar grouping technique with the following orders. 
Amount_tsh: district_code -> region -> subvillage 
Gps_height: district_code -> region -> subvillage

## Preprocessing

A new field named operated_years were extracted by taking the difference between the year in the date_recorded and construction_year. 
operated_years = year(date_recorded) - construction_year

## Encode Categorical Data

All categorical data fields in the dataset were label encoded using the pandas factorize function. 

## Normalize Data

Amount_tsh, gps_height, and population fields were normalized into the range of 1-30 using the min-max scaler. Other numerical values in the dataset fall into a range closer to the one defined here, therefore required no normalization. 

## Model Selection

The implementation used four classifier models and trained and scored each model against the dataset. Here are the results mean of the scores acquired through cross-validation (f1_macro was used as the evaluation metric)
K-nearest neighbors: 0.446 
Random forest: 0.693
XG boost: 0.686
Light GBM: 0.637

Based on these results Random forest classifier was chosen for further improvement. 

## Feature Selection

Feature selection was carried out using the random forest classifier. The three features with the lowest importance, permit, public_meeting, water_quality were removed from the dataset for the final model training. 

## Hyperparameter tuning

For the selected random forest classifier, hyperparameter tuning was carried out using GridSearchCV for parameters n_estimators (100, 400, 700, 1000) and max_depth (20, 50, 70, 100, 100). The best parameters retrieved from this training were:
N_estimators = 1000
Max_depth = 50

## Building the final model

Using the tuned parameters, the final random forest classifier was built. Then, the trained model was used to make predictions on the test dataset. The model managed to score 0.6453 on the Driven Data platform. 




