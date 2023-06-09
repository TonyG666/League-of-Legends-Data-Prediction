# League-of-Legends-Data-Prediction

By Ricky Zhu(r2zhu@ucsd.edu) and Tony Guo(xig003@ucsd.edu)

---

# Framing the Problem

## Prediction Problem and Type

We are interested in the following prediction: 
**Predict which role (top-lane, jungle, support, etc.) a player played given their post-game data.**

Prediction Type: **multiclass classification**

## Response Variable

The player's **position**(i.e., Top laners(**top**), Jungle(**jng**), Mid laners(**mid**), Bottom laners(**bot**), or Support(**sup**))

We choose position as our response variable because we are interested in seeing any possible correlation between a player's performance data and their position in the team.

## Metric

We choose **Accuracy Score** as our metric. Accuracy Score is a simple and intuitive metric that measures the overall correctness of predictions. It calculates the ratio of correctly classified samples to the total number of samples. While Accuracy Score is sensitive to class imbalances, our five player position classes are balanced and have the same number of samples, so we don't need to worry about it.

## Time of Prediction

We are using post-game data to predict the position of player, while the position of player is determined at the start of the game. As we do the predictions after the games are finished, we can use all the data in the dataset.

## Relevant Columns

1. **damagetakenperminute**: Damage Taken Per Minute, indicating the amount of damage a player take from the attacks of others per minute.

2. **wpm**: **Ward Per Minute**, indicating the amount of wards a player place on the ground per minute. *Ward is a deployable unit that removes the fog of war in a certain area of the map.*

3. **position**: The player's **position** in a single eSport match.
* *Note that in a League of Legends map, there are three lanes total and also some jungle areas. Top laner covers top lane, Mid laner covers mid lane, Jungle covers jungle areas, Support and Bottom laner cover the bottom lane.*

4. **kills**: The **number of kills** a player have in a single eSport match.

5. **assists**: The **number of assists** a player have in a single eSport match.

6. **death**: The **number of deaths** a player have in a single eSport match.

7. **earned gpm**: earned **Gold Per Minute** by player in a single game. 
* *Gold in League of Legends is used to buy items in order to make your character stronger.*

8. **cspm**: **Creep Score Per Minute**, indicating the number of **minions** killed by a player per minute.

9. **vspm**: **Vision Score Per Minute**, indicating how much of your team's vision has been influenced by a player's play. 

10. **dpm**: **Damage Per Minute**, indicating the amount of damage dealt by a player per minute.

11. **result**: The result of the game, i.e., win or lose. **Win is denoted by 1 while death is 0**.

12. **gamelengh**: The length of the game, marked by a number in **seconds**.


# Baseline Model

## Model Description

* The pipeline consists of **two main steps**: **data preprocessing** and **classification**.
* The data **preprocessing** step is performed by a **ColumnTransformer**, which applies specific transformations to different subsets of columns in the dataset.
* The preprocessing step includes standard scaling of the "wpm" feature using StandardScaler and one-hot encoding of the "result" feature using OneHotEncoder.
* The remainder parameter is set to **'passthrough'**, indicating that any remaining columns not explicitly transformed will be passed through without any changes.
* The classification step utilizes a **DecisionTreeClassifier**, which is responsible for making predictions based on the preprocessed data.

## Features in the Model

* The features used for training and prediction are all columns in the cleaned dataframe except for the "position" column.
* We focus on the following features to do transformations: 
  * **Quantitative** Feature: "wpm"
  * **Nominal** Feature: "result" (after one-hot encoding)
  
## Encoding

* The "wpm" feature is **standardized** using **StandardScaler**, which scales the values to have zero mean and unit variance.
* The "result" feature is one-hot encoded using **OneHotEncoder**, creating **binary columns** for each unique category of the "result" feature.

## Performance

For 10 times, we **randomly** split whole dataset into a train/test split of **7:3** and fit the pipeline model. Then we calculate the **accuracy scores** on the test set. 
*We calculate the score for ten times because the result may vary and we want to see if the accuracy scores are consistent.*

After experiments, we notice that the accuracy scores are around **0.69**. Thus we can conclude that the model's performance is **not very good**, which indicates that we can't accurately predict player positions solely with the information about ward per minute and result of the game.

Also, there may be **other limitations with the DecisionTreeClassifier** we are using, since decision trees have a tendency to **overfit the training data**. 
* *Overfitting occurs when the model captures noise and irrelevant patterns in the training data, leading to poor generalization on unseen data. This is more likely to happen with multiclass classification because the decision boundaries can become more intricate, and the model might memorize the training examples instead of learning meaningful patterns.*


# Final Model

## Performance

...

# Fairness Analysis

In order to test the fairness of our Final Model, we try to answer the question “does my model perform worse for games of **X** length than it does for games of **Y** length?”, for an interesting choice of X and Y.

Choice of X & Y: **long & short**

Evaluation Meric: **accuracy score**

We do it with **permutation** testings.

**Null Hypothesis**: Our model is fair. The classifier's accuracy is the same for both long games(longer than 30 minutes, 
or to say 1800 seconds) and short games, and any differences are due to chance.
**Alternative Hypothesis**: Our model is unfair. The classifier's accuracy is higher for short games.

Choice of **Test Statistic**: Difference between accuracy score of Long and Short games(Short - Long).

**Significane Level**: 0.05

<iframe src="assets/hist.html" width=800 height=600 frameBorder=0></iframe>

**P-value**: Our p-value vary from 0.7 to almost 1.0 in 100 permutation testings, which are all greater than 0.05.

**Conclusion**: we **fail to reject the null hypothesis**, which suggests that our model **may be fair**, and the classifier's accuracy may be the same for both long games and short games. This fail of rejection is based on our statistical analysis, specifically the calculation of the p-value, which is a measure of the likelihood of obtaining a result as extreme as, or more extreme than, the one observed if the null hypothesis is true. In our case, the obtained p-value is bigger than our commonly used significance level of 0.05 (p-value > 0.05).

However, it is important to consider the **limitations** of our study, such as the specific context, population size, and potential confounding variables, when interpreting these results. Further research and analysis may be warranted to gain a deeper understanding of whether we can predict a player's position based on their post-game data.