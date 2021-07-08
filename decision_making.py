import mldata, multilayerperceptron
from pickle import load

mlp = load(open('mlp.pkl', 'rb'))
mlp.predict()
