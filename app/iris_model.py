import pickle
import gzip
with gzip.open('app/model/iris_model.pgz', 'r') as f:
    Model = pickle.load(f)

def predict(input):
    pred = Model.predict(input)[0]
    print(pred)
    return pred
