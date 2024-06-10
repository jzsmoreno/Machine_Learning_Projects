from moviesDataset_new import ToTensor, movieDataset

data_trainPath = "./Data/ratings_new_train.db"
data_valPath = "./Data/ratings_new_val.db"

data_train = movieDataset(data_trainPath, transform=ToTensor())
data_val = movieDataset(data_valPath, transform=ToTensor())

dataPath = "./Data/"

# Lets save ratings data
shelve_data = shelve.open(dataPath + "ratings.db")
try:
    ratings = shelve_data["ratings"]
finally:
    shelve_data.close()

indexes_train = []
indexes_val = []

uniqueUsers = ratings["User"].unique()
for user in uniqueUsers:
    indexes = ratings[ratings["User"] == user].index.values
    i_train = np.random.choice(indexes, size=int(len(indexes) * 0.8), replace=False)
    i_val = np.setdiff1d(indexes, i_train, assume_unique=True)
    indexes_train.append(i_train)
    indexes_val.append(i_val)

ratings_train_indexes = np.concatenate(indexes_train)
ratings_val_indexes = np.concatenate(indexes_val)

ratings_train = ratings.iloc[ratings_train_indexes, :]
ratings_val = ratings.iloc[ratings_val_indexes, :]


# Lets save ratings data
shelve_data = shelve.open(dataPath + "ratings_new_train.db")
try:
    shelve_data["ratings"] = ratings_train
finally:
    shelve_data.close()


# Lets save ratings data
shelve_data = shelve.open(dataPath + "ratings_new_val.db")
try:
    shelve_data["ratings"] = ratings_val
finally:
    shelve_data.close()


from torch.optim import Adam


class AdamCustom(Adam):
    def __init__(self, params, weight_decay2=0.1, **kwargs):
        super(AdamCustom, self).__init__(params, **kwargs)
        self.weight_decay2 = weight_decay2

    def step(self, epoch, switch, closure=None):
        if epoch < switch:
            super().step(closure=closure)
        else:
            for group in self.param_groups:
                group["weight_decay"] = self.weight_decay2
            super().step(closure=closure)


weight_decay = 0.0001
output_range = (0, 0)
scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.05,
    steps_per_epoch=math.ceil(len(data_train) / batch_size),
    epochs=num_epochs,
)


weight_decay = 0.0001
output_range = (0, 5.5)

scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.05,
    steps_per_epoch=math.ceil(len(data_train) / batch_size),
    epochs=num_epochs,
)
