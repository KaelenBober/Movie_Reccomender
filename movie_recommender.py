import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch and format data
data = fetch_movielens(min_rating = 4.0)

#print training and testing data
print(repr(data["train"]))
print(repr(data["test"]))

#create model
'''
warp = Wighted Approximate-Rank Pairwise
Helps create reccomendations by looking at existing using rating pairs and creating rankings for each.
'''
model = LightFM(loss = "warp")


#train model
model.fit(data['train'], epoch=30, num_threads=2)

def sample_reccomendation(model, data, user_ids):

    #number of users and movies in training data
    n_users, n_items = data['train'].shape

    #generate recommendations for each user we input
    for user_id in user_ids:

        #movies that are known to be liked
        known_pos = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies the model predicts will be liked
        scores = model.predict(user_id, np.arage(n_items))
        #ranks in order most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        #print the reseults 
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_pos[:3]:
            print("     %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("             %s" % x)

sample_reccomendation(model, data, [3, 25, 450])