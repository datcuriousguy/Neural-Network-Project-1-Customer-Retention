import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 10000

"""
Notes:

We use a poisson distribution in order to simulate users 
logging in not quite every day - 
2/3rds of the month with an 
average of 20.

We use arguments for np.random.poisson():

a) lam - This is the rough mean in the poisson distribution. Gives the function an idea of y-axis center for the 
         poisson distribution.
b) size - This is the total number of points (customers) in the poisson distribution
"""

# Feature 1: Number of logins last month (0 to 100)
num_logins = np.random.poisson(lam=20, size=n_samples)

# Feature 2: Support tickets filed (0 to 10)
support_tickets = np.random.poisson(lam=1.5, size=n_samples)

# Feature 3: New features used (0 to 5)
new_features_used = np.random.randint(0, 6, size=n_samples)

# Feature 4: Days since last login (0 to 60)
days_since_last_login = np.random.randint(0, 61, size=n_samples)

# Feature 5: Account age in months (1 to 36)
account_age_months = np.random.randint(1, 37, size=n_samples)

# Feature 6: Total spent in dollars ($10 to $2000), (say)
total_spent = np.random.gamma(shape=2.0, scale=200.0, size=n_samples)

# Label: Churn (0 indicates stay, 1 indicates stopping the subscription service) - simulated with slight correlation
# The intuition: Customers with fewer logins, more tickets, and fewer new features are more likely to churn

"""
Note:
churn_prob represents the probability that a given customer will stop paying the subscription at will.
It is the average of the:

- number of logins. 5 is arbitrary and low, a number below which, would indicate disinterest.
- support tickets filed (indicating a less-than-pleasant customer experience). 3 is the average.
- number of new features used (perhaps linked to customer interest in making full use of the service)
- the lower the days since last login, the more interested a customer is. 30 is the average.

Notice that the all 4 features are being converted into 1 or 0 and the average is taken, resulting in a scenario
such that the majority (more 0's than 1's or more 1's than 0's) wins. In addition, 1100 would indicate a 0.5

"""

churn_prob = (
    (num_logins < 5).astype(int) +
    (support_tickets > 3).astype(int) +
    (new_features_used == 0).astype(int) +
    (days_since_last_login > 30).astype(int)
) / 4

"""
Below, churn_label is simply the final 0/1 (Analogically True/False) indicating if the customer stayed (0) or stopped (1) 
the subscription. Computed into a float for consistency across the features, which are also floats.
Also note 0.1 is simply deliberately added noise (for reality)
"""

churn_labels = (churn_prob + 0.1 * np.random.rand(n_samples)) > 0.5
churn_labels = churn_labels.astype(np.float32)

# Combine all into a DataFrame (it would be helpful for inspection so why not)
df = pd.DataFrame({
    'num_logins': num_logins,
    'support_tickets': support_tickets,
    'new_features_used': new_features_used,
    'days_since_last_login': days_since_last_login,
    'account_age_months': account_age_months,
    'total_spent': total_spent,
    'churn': churn_labels
})

"""
Because Torch expects a 'Tensor' object, we use torch.tensor() to convert our dataframe into 
a proper Tensor. We do this by:
 
 - extracting the features (df.drop(columns='churn').values)
 
 - ensure that values within the tensor are of the datatype 'float32' since
   this is the primary datatype used in the training process (hence ensuring compatibility)
   
 - turn any 1-Dimensional arrays into two-dimensional arrays tensor([1., 0., 1., 0., 1.]) become
    tensor([[1.],
        [0.],
        [1.],
        [0.],
        [1.]])
        
        i.e. 2-dimensional which suits our task of classifying our predictions into two values:
        - stay (0), or
        - end subscription (1)

According to torch.tensor() docs, this function:

"preserves autograd history and avoids copies where possible. :func:`torch.from_numpy` 
creates a tensor that shares storage with a NumPy array".
"""

features = torch.tensor(df.drop(columns='churn').values, dtype=torch.float32)
labels = torch.tensor(df['churn'].values, dtype=torch.float32).unsqueeze(1)

"""
The dataset creation and the dataloader object allow for what appear to be standard procedures in
training neural networks:

 - Batching: splits the data into batches of 64 samples during training (faster & memory-efficient.)

 - Shuffling: randomizes order each epoch (prevents the model from learning data order hence defeating the point.)
"""

dataset = TensorDataset(features, labels)

# used for seeing the object:  print('dataset TensorDataset: \n ',dataset)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

"""
Since DataFrames don't work with Torch Tensors, we need to convert
the features Tensor into a numpy array using the .numpy() function.
We need to show it as a DataFrame.
"""

features_np = features.numpy()

"""
Though not necessary, we use .flatten() as a good practice
within showing a DataFrame as a Numpy Array.
"""

labels_np = labels.numpy().flatten()
full_data = pd.DataFrame(features_np, columns=df.drop(columns='churn').columns)

print("Sample data:\n")

"""
Note: you can use this line to print the entire set of predictions and their 6 labels.

    # print(full_data.head(len(features)).to_string(index=False))

"""
print(full_data)
