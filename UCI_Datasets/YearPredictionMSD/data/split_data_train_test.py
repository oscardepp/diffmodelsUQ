
import numpy as np

# We set the random seed

np.random.seed(1)

# We load the data

data = np.loadtxt('data.txt', delimiter=',')
n = data.shape[ 0 ]

# We generate the training test splits

n_splits = 1
for i in range(n_splits):

    permutation = np.random.choice(range(n), n, replace = False)

    index_train = np.arange(463715)
    index_test = np.arange(463715, n)

    np.savetxt("index_train_{}.txt".format(i), index_train, fmt = '%d')
    np.savetxt("index_test_{}.txt".format(i), index_test, fmt = '%d')

    print(i)

np.savetxt("n_splits.txt", np.array([ n_splits ]), fmt = '%d')
np.savetxt('data.txt', data)

# We store the index to the features and to the target

index_features = np.array(range(1, data.shape[1]), dtype = int)
index_target = np.array([0])

np.savetxt("index_features.txt", index_features, fmt = '%d')
np.savetxt("index_target.txt", index_target, fmt = '%d')

# We store the number of hidden neurons to use

np.savetxt("n_hidden.txt", np.array([ 50 ]), fmt = '%d')

# We store the number of epochs to use

np.savetxt("n_epochs.txt", np.array([ 40 ]), fmt = '%d')
