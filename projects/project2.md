# Can you predict the total number of incorrect calls an umpire will make within a singular game?  

## Problem Definition:
The target variable is total incorrect calls. This is a regression problem. MLB teams might benefit from this model, especially in an era of the Automated Ball-Strike (ABS) Challenge System. Additionally, the MLB Baseball Operations division might benefit as well, since they handle the hiring of umpires and could use a model like this to assess inadequate performances. The momentum of a baseball game can be changed drastically based on the accuracy or inaccuracy of calling balls and strikes. Even with ABS, the amount of times you can challenge a pitch is limited to two. If a team runs out of challenges, full control goes back into the umpire as it was before the ABS era. As Brian Mills states, “ball–strike calls may leave each player's performance in an at bat subject to bias by these officials.” These umpires suddenly can become sloppier in their judgement, still allowing missed calls that should have been avoided.  

## Background and Context:
While the pitcher is the most important player on the field, the umpire holds an excessive amount of discretion in the trajectory of every at-bat. My domain knowledge for this question and my approach comes from years of watching baseball, and even working in baseball. I have direct experience with pitching data, determining if pitches were called incorrectly, and seeing if the result of the at-bat should have been different if the pitches were called correctly. While this is not one of the variables that was included in the dataset I have access to, I found it interesting that temperature could be a major variable in predicting the accuracy of incorrect calls. This is introduced in an article written by Eric Fesselmeyer, where he explores the impact of temperature and heat index on umpiring. Though, there are variables in the dataset that can be used to represent home team advantage, which was explored in an article by Mike Hsu. The article focuses on accuracy, similar to this project, and points out that “umpires may have their own perceptions of strike zones and call the pitches accordingly.” Umpires are not absolute predictable beings, but it is possible to find supported patterns that can help teams and league officials know how their assigned umpire may affect the game.  

## Data Description:
The dataset comes from [MLB Baseball Umpire Scorecards (2015 - 2022)](https://www.kaggle.com/datasets/mattop/mlb-baseball-umpire-scorecards-2015-2022) by user mattop on Kaggle. Each row in the dataset represents a different, individual MLB game that was umpired between the years 2015 and 2022. The data set features 19 columns and 18213 rows. While there are a lot of featured variables in this dataset, the target variable that this project focuses on is total incorrect calls. Potential features available to help focus on this target variable include the amount of pitches called, the expected amount of incorrect calls, and how many runs either team scored. The data comes from the years 2015-2022, which was umpiring without ABS. Since umpires are unique human beings, there is an unknown in their routines and attitudes for the games they were umpiring. Additionally, factors such as weather and ballpark are not measured in this dataset, and it was previously explored that these are variables that could further help predict umpire performance.  

## Data Understanding and Exploration:

## Data Preparation and Feature Selection:
```
cols = ['home_team_runs', 'away_team_runs', 'incorrect_calls', 'expected_incorrect_calls']
df_clean = df[cols].copy()

df_clean = df_clean.replace('ND', np.nan)
df_clean = df_clean.apply(pd.to_numeric)

df_clean = df_clean.dropna()
```  
While the dataset claims to have been cleaned for data analysis, I had come across problems of missing values and had to further clean the data for my own analysis.

## Baseline and Model Development:
To establish a baseline for this project, I started by creating an OLS model to see values such as co-efficients and R^2 values. It is appropriate to use a baseline in machine learning to see how the complexity shifts the model. OLS is one of the most simplistic forms of machine learning, which makes seeing this complexity easier. To answer my research question, I used linear regression and ridge regression. These models were appropriate because these forms of regression models handle multiple variables very well. Additionally, this research question allows for the chance of multicollinearity, which regression models handle better than some other models. To my knowledge, I do not believe that I tuned any model settings or hyperparameters based on the sklearn documentations I used for this project. To assure that the models were compared fairly, I pulled the same result metrics, such as the R^2 value, for both models to see if there was any differentiation between the two.

## Model Evaluation and Selection:
To evaluate the models, I compared the two model’s final results to each other and the baseline OLS model. Specifically, the R^2 value, mean squared error, variable coefficients, and intercept are the values I found capable of being compared across all of the forms of machine learning.

## Model Interpretation and Insights:

## Limitations, Ethics, and Reflection:
This project and dataset is revolved around MLB and umpiring, which means that this is human influenced data. Therefore, there will never be a 100% certainty in recreating nor predicting the absolute outcomes of data. This can create a gap or bias between what the model thinks should occur and what the human influencing the data is actually experiencing. Similar to who would be affected by the model in general, MLB teams and league offices would be impacted by incorrect predictions. Incorrect predictions can end up translating into poor performances and invalid game preparations. In result, this could mean the difference between wins and losses. False positives and false negatives translate to strikes called balls and balls called strikes. With this dataset, there is a category for expected incorrect calls, but there is no further specification on that metric. With more variables and fine tuning, I could see this model being appropriate for real-world decision-making. Even with ABS challenges available, it is helpful to teams in advance to know the consistency of the umpire behind them. While working on this project, the Atlanta Braves vs Los Angeles Dodgers NLCS Game 1 was on. The umpire had an exaggerated zone that I believe a model like this could have prepared the teams on knowing how many calls this umpire may miss and how often they may expect to exercise their ABS challenges. I would love to find more variables to expand this model with, such as how weather and stadium layouts affect umpire performance. While I believe that the two forms of regression that I chose for this project are the best forms of modeling, I think trying other forms of machine learning would have been interesting to see the evaluation impact. Users should understand that this model is not currently absolute. Therefore, there is still a lot of room for error in the model.  

## Code and Transparency:
Jupyter Notebook: Project2.html  
Dataset: [MLB Baseball Umpire Scorecards (2015 - 2022)](https://www.kaggle.com/datasets/mattop/mlb-baseball-umpire-scorecards-2015-2022)  

## Key Academic References:
Fesselmeyer, E. (2021). The impact of temperature on labor quality: Umpire accuracy in Major League Baseball. Southern Economic Journal, 88(2), 545-567.  
Hsu, M. (2024). Umpire home bias in major league baseball. Journal of Sports Economics, 25(4), 423-442.  
Mills, B. M. (2014). Social pressure at the plate: Inequality aversion, status, and mere exposure. Managerial and Decision Economics, 35(6), 387-403.
