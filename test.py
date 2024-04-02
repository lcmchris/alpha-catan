import pickle

with open("catan_model.pickle", "rb") as file:
    turn_list = pickle.load(file)

print(turn_list)