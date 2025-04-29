

## Readme

The goal of this project is to simulate the environment of using a neural network within a SaaS (Software as a Service) startup. In this case, the startup’s main product is a software for which customers (up to ~10,000 paying customers at any one time) pay a monthly subscription to use. Unfortunately, sometimes customers may abruptly stop using the service and / or cancel their subscription. The percentage of customers that do this per unit time (in this case monthly) is referred to as the **churn rate**.

In order to predict the churn rate for each successive month, we would like to develop a neural network that analyses the **features** of each new customer and uses these features (essentially integers and floats) to predict the **probability** of the customer terminating their subscription.

Files:

- **`Readme.md`**
    
    **This file.**
    
- **`data_creation.py`**
    
    A separate python file for running to create the initial data (which is split in a train-test ratio of 80:20)
    
- **`neural_network.py`**
    
    The main python code for the neural network, including hardcoded hyperparameters.
    
- **`expected_sample_data.txt`**
    
    A text file containing a result of running data_creation.py, for reference.
    

Note: I would recommend pasting the contents of both **`data_creation.py`** and **`neural_network.py`** in either file and then running that file, as opposed to running them separately.


## Data Creation

First, run data_creation.py. Before we create the neural network and train it on customer features, we first need a past history of customer stats (features) to exist. This is the job of the code in **data_creation.py**

---

## Understanding the Significance of the Data

Here are the previously mentioned 6 features, per customer:

| **Feature** | **Description** |
| --- | --- |
| Number of logins in last month | Activity |
| Number of support tickets filed | Problems |
| Number of new features used | Engagement |
| Days since last login | Inactivity |
| Account age in months | Loyalty |
| Total amount spent | Revenue contribution |

These are the 6 attributes of a customer’s subscription that will be used by the neural network to decide if they are likely to stay or leave in a given timeframe.

Taking into consideration the datatypes of each feature:

| Feature | Description | Raw Type | Final Tensor Type |
| --- | --- | --- | --- |
| num_logins | Number of logins in last month | int | float32 |
| support_tickets | Number of support tickets filed for a given customer | int | float32 |
| new_features_used | Number of new features used by the customer | int | float32 |
| days_since_last_login | Days since last login | int | float32 |
| account_age_months | Account age | int | float32 |
| total_spent | Total amount spent in dollars (currency doesn’t matter that much here) | float | float32 |

Note: I chose a float datatype for all 6 features for consistency. The weights and gradients are floats as well. Across the data the output will be a probability, a float between 0 and 1.

---

## How The Neural Network is Trained

1. Forward Pass to get predictions
2. Compute Loss of those predictions
3. Backpropagage to compute gradients 
    
    (Find out how much the error changes for a given change in gradient
    
4. Recompute weights 
    
    (New weight = Old weight – Learning rate × Gradient)
    
5. Repeat for all batches, if any

---

## Best Practices while Creating The Neural Network (From other GitHub Projects)

1. Using a Modular Architecture by making use of layers, modular functions, feed-forward and feed-backward functions, loss-functions, and optimizers. This is good for readability and makes things reusable as well.
2. Defining and implementing ways to handle missing values, normalizing features (if needed), and taking note of outliers are all recommended practices in the GitHub Community. Data cleaning and preprocessing prior to use for training will make sure that the neural network is properly trained while avoiding or reducing the chance for anomalous results.
3. Using regularization (minimizing needless complexity and exposing the network to more diverse data) will help prevent overfitting while improving generalization, which is desired in most neural networks. Note: There are multiple methods for regularization.
4. Optional: Tuning Hyperparameters like epochs and learning rate specific to strategies such as grid search or Beayesian optimization, can enhance model performance.
5. Taking note of the Bias and Variance of a given model is important as it enables the tuning of hyperparameters to match different scenarios (Essentially finding a point between Linear Regression and overfitting).
6. Obviously, a well-written README file (hello there) and good directory structure are useful to have for any given project.
