
"""

This is the complete code, consisting of three main functions:

1. create_data()
2. prepare_data()
3. define_and_run_nn()

Current final (last few lines) output of running it on 50 epochs:

--- Evaluation on Test Data ---
Test accuracy: 93.65%
Example predictions:
tensor([0., 0., 0., 0., 1., 1., 0., 0., 0., 1.]) vs Ground truth:
tensor([0., 0., 0., 0., 1., 1., 0., 0., 0., 1.])

"""

def create_data():

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn.functional as F
    # Set random seed for reproducibility
    np.random.seed(42)
    n_samples = 10000

    """
    Note: This code consists of three functions: 

    1. create
    """

    # overarching global declarations to make creation, cleaning and defining/running functions compatible:

    global train_loader, y_test_tensor

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
    global df
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

    # Uncomment this line to see the full sample data created:  print(full_data.head(len(features)).to_string(index=False))

    """ /// ... PREPARING THE DATA ... /// """

    global train_loader, y_test_tensor, y_test_tensor

create_data()


def prepare_data():

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn.functional as F

    # we use the existing df with the churn column

    """
    prepare_data() is meant to prepare the (now created) data for being used to train the neural network in
    neural_network.py

    note: X_test_tensor and Y_test_tensor are only going to be used for the testing process in the neural network,
    so not yet.

    features_np is the dataframe using all datatypes as float32.

    labels_np is reshaped as a vertical matrix of the

    [[0.]
    [0.]
    [0.]
    ...
    [0.]
    [0.]
    [0.]]

    here, the significance of the -1 is that it allows the appropriate number of rows as necessary (dynamically).

    Again, the training data is wrapped into a DataLoader to enable batch training with shuffling (good practice).
    """

    features_np = df.drop(columns='churn').values.astype(np.float32)

    labels_np = df['churn'].values.astype(np.float32).reshape(-1, 1)

    # Train-test split ratio of 80-20. 42 is arbitrarily chosen just like seed()
    X_train, X_test, y_train, y_test = train_test_split(features_np, labels_np, test_size=0.2, random_state=42)

    """
    global declarations to make the three functions compatible. Note that, though the
    y_train_tensor, x_train_tensor are unused as of now, these are still declared globally
    for safety.
    """

    global y_test_tensor, x_test_tensor, y_train_tensor, x_train_tensor

    # Convert to tensors
    x_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    x_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    # Wrap in DataLoader
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    global train_loader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    """
    for batch_features, batch_labels in train_loader:
    # Forward pass, loss, backward pass, optimizer step happen in this function in the neural_network.py file.
    """
    return train_dataset, train_loader


prepare_data()

"""
Here I defined the Network class and inside it, the function to start
forward propagation. Note there are:

- One input_features layer
- two hidden layers 
- one output layer (it is named as out_features by convention)

"""

def define_and_run_nn():

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn.functional as F


    class SimpleNet(nn.Module):
        def __init__(self, in_features, h1=16, h2=8, out_features=1):
            super().__init__()
            self.fc1 = nn.Linear(in_features, h1)
            self.fc2 = nn.Linear(h1, h2)
            self.out = nn.Linear(h2, out_features)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.sigmoid(self.out(x))
            return x

    model = SimpleNet(in_features=6)
    print("Model structure:\n", model)

    """
    criterion defines the loss (Binary Cross Entropy). It seems like a popular choice for
    binary classification, which is indeed the case here.

    the optimizer uses the popular Adam for optimizing the model's parameters
    such as its gradient, before backpropagation. Heren the .parameters() function
    is what the torch.optim.Adam() function uses to 'know' which specific parameters
    need to be adjusted. Obviously, lr stands for learning rate and starts at 0.001/pass.

    """
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ---------- Step 6: Train the Model ----------
    epochs = 50
    for epoch in range(epochs):
        global train_loader
        print(f"\nEpoch {epoch + 1}/{epochs}")
        for i, (batch_features, batch_labels) in enumerate(train_loader):
            # Forward pass:
            outputs = model(batch_features)
            # loss or error between predictions and actual labels
            loss = criterion(outputs, batch_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Batch {i}: Loss = {loss.item():.4f}")

    """
    To run and evaluate the model, we need to set it to evaluation mode, hence the
    model.eval().


    """

    model.eval()  # Set to evaluation mode
    with torch.no_grad():  # I found that turning off gradient tracking improves memory efficiency& performance
        global y_test_tensor, x_test_tensor
        test_outputs = model(x_test_tensor.float())  # actually getting predictions
        predicted = (test_outputs > 0.5).float()  # converting those outputs >0.5 (i.e., 0 or 1) to binary floats.
        correct = (
                    predicted == y_test_tensor).float().sum()  # these two lines compare predictions to expected (correct) outcomes
        accuracy = correct / y_test_tensor.shape[0]
        print(f'acc print: {accuracy}')  # interestingly, its about 90% accurate :)

        print("\n--- Evaluation on Test Data ---")
        print(f"Test accuracy: {accuracy.item() * 100:.2f}%")
        print(
            f"Example predictions:\n{predicted[:10].squeeze()} vs Ground truth:\n{y_test_tensor[:10].squeeze()}")  # that 90%
        # number becomes clear here . see below docstring of output:

    """
    for 5 epochs:

    --- Evaluation on Test Data ---
    Test accuracy: 82.05%
    Example predictions:
    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]) vs Ground truth:
    tensor([0., 0., 0., 1., 0., 1., 0., 0., 1., 0.])

    for 50 epochs:

    --- Evaluation on Test Data ---
    Test accuracy: 93.05%
    Example predictions:
    tensor([0., 0., 0., 0., 0., 1., 0., 0., 1., 0.]) vs Ground truth:
    tensor([0., 0., 0., 1., 0., 1., 0., 0., 1., 0.])

    You can see how more epochs by intuition makes the model better at predictions.
    """

define_and_run_nn()


