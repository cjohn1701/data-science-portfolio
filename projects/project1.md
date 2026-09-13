# In 2025, was batting average or home run total more likely to predict RBI total?

## Research Question and Dataset: 
This research question specifically applies to Major League Baseball, a sport that has intrigued me for all of my life. These three metrics are at the top for batters and are commonly referenced when determining what a good or bad player is. A batting triple crown is awarded if a player leads their respective league in batting average, home runs, and Runs Batted In. When coming up with this question, I had players like Luis Arraez in mind. Arraez has been on triple crown watch multiple times throughout the years, and he is famously known for typically having high batting averages. There are other players across the league similar to Arraez, and I was curious if a player like Arraez was more likely to drive in players with their power over distance over a player like Kyle Schwarber that swings for the fences with every at bat. The power of players is commonly taken into consideration when constructing gameday lineups, trying to arrange players to determine where the average hitters should go and the long ball hitters should go in order to maximize team RBI potential. Understanding the relationship between batting average, home runs, and RBIs can be beneficial to managers and players alike to help their teams make the most out of their rosters.  
The source of the dataset I am using to explore this research question is API [MLB-StatsAPI](https://pypi.org/project/MLB-StatsAPI/) by toddrob99 on GitHub. The unit of analysis for this project includes Major League Baseball players and their respective relationships with batting averages, home runs, and Runs Batted In. The features I am focusing on include identifying a player’s name and their respective statistics for average, home runs, and RBIs. While the original data being provided by the API is very large, I was able to shrink the size of the dataset according to what directly correlated with my research question. There were no missing values to worry about with this dataset since all baseball statistics involved default to zero at the beginning of a season, which means there will always be a corresponding number regardless of performance.

## Conceptualized/Operationalized Variables and Important Context:
The key variables I am using for this project are batting average, home run total, and runs batted in. Batting average is calculated by Hits/At-Bats. Hits include singles, doubles, triples, and home runs. An at-bat is a plate appearance that results in either a hit or an out. The home run total focuses only on how many home runs a player hit in a season. A home run is classified as a hit that is launched out of the park and is awarded four bases. Runs batted in (RBI) is a statistic that measures how many times a player’s at bat results in a runner on base scoring, or also including the batter himself scoring if the result is a home run. This research question focuses on studying if RBI is more predictable through batting average or home run totals. All three of these variables go into winning MLB’s Triple Crown for batters, and this question can also predict who may end up winning a triple crown. The dataset is being provided through the API [MLB-StatsAPI](https://pypi.org/project/MLB-StatsAPI/) by toddrob99 on GitHub. Each row in my dataset represents a player that was either top 10 in batting average, home runs, and/or RBI during the 2026 season. Each row lists the player’s name, batting average, home run total, and RBI total. These variables are listed across all players regardless of where they ranked in all categories. I have the list sorted by last name for visual convenience. The original dataset I pulled from with MLB-StatsAPI was initially very large and included more statistics than I really needed for this research question. Originally, I ran into a problem that `statsapi.player_stats` was not paying attention to the season input that I was giving. I solved this by using `statsapi.player_stat_data` instead, and used data from that function instead to pull the variables that I needed for my research question.

## Data Cleaning and Preparation:
```
players = {
    'Jo Adell': 666176,
    'Pete Alonso': 624413,
    'Bo Bichette': 666182,
    'Junior Caminero': 691406,
    'Rafael Devers': 646240,
    'Yandy Díaz': 650490,
    'Freddie Freeman': 518692,
    'Riley Greene': 682985,
    'Nico Hoerner': 663538,
    'Aaron Judge': 592450,
    'Nick Kurtz': 701762,
    'Shohei Ohtani': 660271,
    'Vinnie Pasquantino': 686469,
    'Jeremy Peña': 665161,
    'Cal Raleigh': 663728,
    'Kyle Schwarber': 656941,
    'Juan Soto': 665742,
    'George Springer': 543807,
    'Eugenio Suárez': 553993,
    'Trea Turner': 607208,
    'Taylor Ward': 621493,
    'Jacob Wilson': 805779,
    'Bobby Witt Jr.': 677951
}

data = []

for name, player_id in players.items():
    stats = statsapi.player_stat_data(
        personId=player_id,
        group="hitting",
        type="season",
        season=2025
    )

    stat = stats['stats'][0]['stats']

    data.append({
        'Player': name,
        'AVG': float(stat['avg']),
        'Home Runs': stat['homeRuns'],
        'RBI': stat['rbi']
    })

df = pd.DataFrame(data)

print(df)
```
While there are a lot of interesting variables being transmitted when running the `statsapi.player_stat_data` code, this research question only involves three variables. Therefore, I created a DataFrame using pandas that only picked out the player’s name, batting average, home run total, and RBI total. This puts the information into a more presentable and readable format that specifically highlights only the variables that are most important to the question. Thankfully, there were no missing values in the dataset, so I did not have to deal with cleaning up nonexistent values. The main way I was able to filter data was by locating what a player’s ID was for the specific list of players I was looking for and pulling their average, home runs, and RBIs. These steps made all of the data I was being presented with less overwhelming, allowing me to focus on only what was relevant to the research question.

## Visualizations:
<img width="851" height="546" alt="Image" src="https://github.com/user-attachments/assets/7a3b1b01-b7cb-468e-b9eb-8faef2e72b62" />  

**Batting Average vs RBI**  
This is a scatter plot that shows the relationship between batting average and Runs Batted In. Each blue dot on the visualization represents a unique player that was pulled from the DataFrame. The red line represents the regression line, while the red shaded area represents the 95% confidence interval associated with the best-fit line. According to the graph, there is a negative correlation between the two variables. This suggests that having a higher batting average does not directly translate toward having a higher RBI total.

<img width="851" height="546" alt="Image" src="https://github.com/user-attachments/assets/558dd2ae-8524-44ca-ba52-ec4171eb30b1" />  

**Home Runs vs RBI**  
This is a scatter plot that shows the relationship between home runs and Runs Batted In. Each blue dot on the visualization represents a unique player that was pulled from the DataFrame. The red line represents the regression line, while the red shaded area represents the 95% confidence interval associated with the best-fit line. According to the graph, there is a positive correlation between the two variables. This suggests that having a higher home run total does directly translate toward having a higher RBI total.

<img width="646" height="526" alt="Image" src="https://github.com/user-attachments/assets/bb3142fb-328d-414e-9d37-9633c7163e4f" />  

**Correlation Between AVG, HR, RBI**  
This is a heat map that shows the correlations between batting average, home runs, and Runs Batted In. The more correlated two variables are, the square relating the variables will appear more red. The less correlated two variables, the square relating the variables will appear more blue. This visualization supports the trends being shown by the two scatter plots. The square that shows the numerical correlation between batting average and RBIs is blue and prints a negative number, implying negative correlation. Meanwhile, the square that shows the numerical correlation between home runs and RBIs is red and prints a positive number, implying positive correlation.

<img width="668" height="546" alt="Image" src="https://github.com/user-attachments/assets/cb1630dc-0951-4274-812f-a03751e05c7c" />  

**Batting Average vs Home Runs**  
This is a hexbin plot that shows player distribution based off of batting average and home run totals. The graph is colorized by player density, implying that a bin with a darker square holds more player value than the lighter colored bins. This graph has the darkest bin in a region of high batting average and low home runs. Therefore, this suggests that having a higher batting average does not mean that a player is going to hit more home runs, which ties back into the earlier visualizations by suggesting that players in this category are less likely to have high RBI totals. 

Based on the visualizations, the graphs show that home runs were statistically more likely to predict RBI count in 2025 as opposed to batting average. Considering sports are not always able to be soundly predicted, there will naturally be a few outliers because of the human performance aspect. The graph that I notice the most anomalies in is the batting average vs RBI scatter plot. There is a player towards the upper right corner, which can be identified as Aaron Judge of the New York Yankees. Despite this graph showing a negative line of best fit, the location of the player is very far from the confidence interval and even has the potential to be within a confidence interval if the trend was positive. This opens the possibility of individual cases of how human performance reflects results, and questions what may be behind why some players are so far from their counterparts.

## Storytelling and Narrative:
To relate these graphics back to the original research question, there is evidence to draw a conclusion that there is more correlation between home runs and RBIs than batting average and RBIs. With the graphs producing a conclusion that home runs have a more positive correlation with RBIs than batting average does, this relates back to our research question drawing a conclusion that home runs had more influence on RBI count than batting average in the year 2025. The data is attempting to tell the story that home runs have a stronger correlation towards RBI than batting average does towards RBI production. Though, it would be incorrect to draw a conclusion that home run total is the only variable that contributes to RBI count. This research question takes into account only batting average and home runs, and assuming that there are no other possible variables that can play a role would be an incorrect assumption. Therefore, the results of the research can only say that home runs have a positive correlation with earning RBIs.


## Ethics and Limitations:
One of the main details this dataset fails to capture is the typical spot in the order for each batter involved in this research question. This is an aspect of the game of baseball that cannot be whittled down to a singular number like batting average can. Lineup orders can greatly affect all of the variables related to this research question depending on who is batting ahead of the specific players. I believe that this dataset naturally holds a slight bias based on the selection of players themselves. Out of the 6 divisional leaders in 2025, 5 of those teams are represented by players chosen for the dataset. Naturally, teams that lead the league in wins are more likely to have higher amounts of runs and RBIs across their players because that translates to outscoring the opponent. It is possible that the teams have a larger influence on player performance than the numbers are able to reveal. If I had more time, I would love to add more variables to the research question such as slugging and see if there are further relationships behind how to predict what causes players to have high RBI counts at the end of a season. Additionally, if there was a way to add non tangibles into the equation, such as batting order, I would have loved to incorporate that. For example, most of these players typically bat in the top half of the order, and I would like to see if there is a specific spot, such as cleanup or 4th, that typically drives in more RBI than other spots in the order.

## Code and AI Transparency:
Jupyter Notebook: [Project1.html](Project1.html)  
Dataset/API: [MLB-StatsAPI](https://pypi.org/project/MLB-StatsAPI/)  
No AI was used on this project. Any Python code or GitHub code was tweaked through either the [MLB-StatsAPI Wiki GitHub page](https://github.com/toddrob99/MLB-StatsAPI/wiki), YouTube tutorials, or various documentation websites to help with syntax and visual customization.  
To identify what players to use for this research question, I used Baseball-Reference.com's list of [2025 Major League Baseball Batting Leaders](https://www.baseball-reference.com/leagues/majors/2025-batting-leaders.shtml).

## Key Academic References:

