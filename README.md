# Data_Science_Project_Expedition_Delay_Prediction


Machine Learning
Use Cases
TD-01
Expedition delay Prediction

Aymen BOUGUERRA A5-DIA-2


Part One: Answering the relevant questions:

 ![image](https://user-images.githubusercontent.com/97101162/195266855-b77542ca-ef23-43c1-8098-47c174510412.png)

Received Data from the client

1.	What is the action associated with the delay prediction 
Predicting a delay is not an action, what would have also be interesting in a real life scenario would to have the label as a feature and the service provider as a label, so what a client would normal want is the chose the delay and then the model returning the best service provider that would have more chances to expediate in that delay, however that is not possible with this dataset due to the disproportionality of the data regarding service providers.
 
 ![image](https://user-images.githubusercontent.com/97101162/195266892-8f81cd72-d576-449f-aa4e-f9b02d5e358c.png)

2.	Do we have enough data?
It is true that the amount of raw data that we have for this application is large (more than 500k entries), but at first glance we have a very limited number of features. After further examining the data, and with feature engineering we were able to extract multiple and very important additional features of the initial ones.

3.	What is the scope of this project?
Even if it not an action, and to respect the objective of the TD we will focus on the delay prediction with no error margin (as it was not agreed upon in the session); the prediction is true only and only if the rounded prediction in days is identical to the ground truth.

4.	BIAS identification 
We have identified some heavy and important biases in the dataset:

 ![image](https://user-images.githubusercontent.com/97101162/195266922-f1acadad-652f-404a-a225-c0cf2340e485.png)

The first bias is the dates, we only have data for 45 days from November first to December 17th (we will see the consequences of this bias in the accuracy difference of our two approaches)

 ![image](https://user-images.githubusercontent.com/97101162/195266942-ed83ab29-a8c9-4175-bd80-b5de1c66dc09.png)

The second bias we have found is the disproportionality of the labels; having too many of one ‘class’ in the labels could tilt the prediction towards it, we have encountered while training our models that sometimes the model doesn’t learn and instead jut predicts 2 all the time and be correct for almost half of the times. This was however a rare case where while splitting the dataset into training and testing most of the 2s were in the training while the other classes were mostly in the test set.

 ![image](https://user-images.githubusercontent.com/97101162/195266961-806f2745-f1f6-449e-8b5b-8c7bae1c5741.png)

Another bias is the lack of data regarding other service providers.
We assume that we cannot have additional data regarding the other dates, other delays or other service providers to reduce the bias.

5.	Is it fully automatic or half?
Not relevant in this application.

6.	Have we adjusted the classification threshold?
We chose a regression model and not a classification one.

7.	What would be a successful model in this application?
Since we are writing this report after finishing the project, we realize that the value of our model depends directly to the expectation of the client; in our case, in our second approach (testing the model on days that were not in the training data), the accuracy was significantly lower than expected, and if we had agreed with the client on an error margin of 1-day, then the results would have been more than sufficient.

8.	What is the life expectancy of the model and how and how much to maintain it?
We assume (feel like) as time goes on, that the model will keep fine tuning itself with new and relevant data, also reducing the biases that it had, further increasing the accuracy that it has. We also feel like that the maintenance cost would be low if the model would have been built to continue feeding and learning from new data. 

9.	Is there any data or features that would be relevant but unobtainable?
Initially, we would say no, but reflecting on the work we think that economic data such as inflation, buying power, holidays; etc. would help the model predicting the delays as this information are correlated to the number of orders received.

10.	Will new biases arise from implementation of our model? (Maintenance bias)
For this specific model we assume that the chances are low, the only bias that could accrue is an increasing expedition delay due to various reasons such as fires, strikes etc.
And it goes both ways, because as our model would learn from this new data it will artificially increase the delays that won’t be relevant anymore as the aggravating factors would no longer be in play.























Note: Since we are writing this report after finishing the project, we realize that the value of our model depends directy to the expectation of the client; in our case, in our second approach (testing the model on days that were not in the training data), the accuracy was significantly lower than expected, and if we had agreed with the client on an error margin of 1-day, then the results would have been more than sufficient.

Part Two: Feature Engineering:
The client’s data have 3 variables: precise order date and time, who is the service provider and the expedition date.
This variables alone are far from being enough, and even though people think that machine and even more deep learning is a magical black box that will just “find the solution“, our job as data scientists is to prepare the data in a relevant way to be fed to the algorithms.

1.	Debit of this day 


 ![image](https://user-images.githubusercontent.com/97101162/195266990-d1e83603-0093-4ef3-903d-85242503fa59.png)

We want to know how many orders will be processed today, some would argue that would should use this variable as normally we need the day to end to count how many orders we processed, we argue however that the expedition centers know how many workers they employ for that day in advance and therefore the debit of that day, and even so, we need this variable to create the next one, that no one would argue the utility:
2.	Debit of yesterday 


 ![image](https://user-images.githubusercontent.com/97101162/195267007-de7b229f-020c-4520-9f79-088df39980c8.png)

By simply shifting the debit column down one position we have now the debit of yesterday and which represents the amount of orders processed the day before (we however needed to cut one day because of NaN)
3.	Separation of time and date of emission 

 ![image](https://user-images.githubusercontent.com/97101162/195267021-e0ed93ab-415e-4812-8069-9e68b8a29b88.png)

We need to separate the two information to work on other features.
4.	We convert the order time from our human forma HH:MM:SS to a more friendly format to be fed to the mdoels. 
We simply convert the time of order to the number of minutes that have passed from midnight until the order. We think that this information is crucial as it allows the model to know if this command was made early in the day and be part of Day+0 or made too late and be treated is Day+1.

 ![image](https://user-images.githubusercontent.com/97101162/195267055-06e66126-6e20-4b78-93fc-92d82a509e3c.png)


5.	Delay in days
 ![image](https://user-images.githubusercontent.com/97101162/195267067-03bea25d-9c0c-47bd-b994-56060e3e61c4.png)

Our label which is the difference between the day of expedition and day of order.
6.	Orders received yesterday 
 ![image](https://user-images.githubusercontent.com/97101162/195267084-fe0156f7-cf52-4b8a-ba31-c5b8fe12daab.png)

We want to know how many new orders we received; this variable is very important as it will also allow us to calculate the most useful one.
7.	Number of orders in queue  

![image](https://user-images.githubusercontent.com/97101162/195267111-36a1fd15-9c78-418a-b2e8-751e4538144f.png)






---
![image](https://user-images.githubusercontent.com/97101162/195267137-e2327e46-a800-4b52-a426-aaa6516daaf2.png)






===
![image](https://user-images.githubusercontent.com/97101162/195267156-33808f64-3db0-4e99-91fc-12fb339aae7c.png)






By subtracting the cumulative summation of the debit up to yesterday from the cumulative summation of the order up to yesterday we can conclude the amount of orders that are waiting to be processed from the previous days 
Note: we also had to slice more rows from the last table and that’s why it start from 14k, also the two tables above don’t start from the same date and is the reason for the inconcinnity.
8.	Removal of noise, outdated data and wrong data:
 ![image](https://user-images.githubusercontent.com/97101162/195267177-694161dc-a27e-48d9-84a9-337f086d8dbd.png)

We removed data linked to 1980 that we assume is outdated.
We removed date from 26/07/2019 as we only had one entry of said date.
 ![image](https://user-images.githubusercontent.com/97101162/195267195-9c855b9f-28ac-4eee-8a47-9a913a4c4a04.png)

We removed some wrong entries as we found that some delays were negative.
9.	Final dataframe:
 ![image](https://user-images.githubusercontent.com/97101162/195267209-cdf2f349-94ae-4ce4-b530-96b3d25ec8ff.png)



We used two models representing two approaches:
The only difference between said approaches was their evaluation data; for the first approach we split the train and test sets from the main dataframe randomly in a classical way, while in the second we followed the recommendation of our teacher that proposed to remove dates from the original dataframe and do the evaluation on them.

The difference being that the first model will be evaluated on data that he never saw but that is similar to data that he trained on, while the second will be evaluated on data that was never in the training set and that represent dates and features that are completely different from the training set.

Theoretically the second approach have a better evaluation method that it simulates a real-life application.
Both approaches use regression and not classification
Absolute accuracy in the main metric used.
Predictions are rounded to the day.
A prediction is true only and only if the rounded prediction is identical to the ground truth.
The results were as follow 
Approach One; random evaluation set: 87%(+-5%)
Approach Two; exclusive dates set: 53%(+-17%)
•	The accuracy in this second approach is significantly lower than the first one.

•	which indicates that our either our model have low generalization ability or that the dataframe and features that we composed aren’t so good

•	we tried to investigate where this accuracy dropout came from with no success

•	53% accuracy can either be good or bad depending on the error margin agreed upon with the client

•	if and error margin of 1day is acceptable then the accuracy would skyrocket

•	53% base accuracy however could be achieved without any AI of any sort and by just giving random lucky guesses.

•	we also have noticed that the standard deviation of the accuracy score of the second approach is height (+-17%) meaning its highly volatile.

•	while it is not the case for the first approach (+-5%)



