# conflict_forecast_colab

## Input 
In the model, there is 3 parts in the input : 
1.	The prio grid data (static and yearly), extract from the grid corresponding to the localization (lat, long) 
2.	Spatial part : The last week and last month other localities sum of events 
3.	Autoregressive part : the last 30 days (event/no event) 

## Model 
The model is a Random Forrest Classifier (Conflict event : yes or no). The model was built reducing the train dataset with no events output, with higher number of forecasted events. Each country is trained over the last 5 years before the start of the conflict period.  

## Dataset output 
1.	The forecast model results, called “conflicts.csv”. 
2.	The observed events values, called “Observed_events.csv”.
3.	The observed fatalities values, called “Observed_fatalities.csv”.

## Cases/countries
- Mali 2012
- Central African Republic 2013
- Burundi 2015
- South Sudan 2013

## Limits/Idea 

- Now, the downsizing of 'non-events' (when there is no conflict) is the same ratio for every cases. It was optimized reducing the global error but might not be ideal. I tried to use a validation set to create one ratio individually, but results were not good. Maybe we can think of other ways to adress this issue.. 
- Also, only the localities encoded as 'conflict_zone' are forecasted (before, I made a mistake and also included the camps). Maybe we can use all the information from Acled ? Using the distance from towns maybe ? 

