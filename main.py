# 0. Import Dat
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('Churn.csv')

X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
y_train.head()


# 1. Import Dependencies
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# 2. Build and Compile Model
model = Sequential()  # instantiating sequential class

# adding bunches of layers

# add Dense-connected layers
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns))) #32 neurons inside the dense layer
# activation function act as a modifier in our neural network
# get the raw output from the neuron
# input_dim is the length of the X_train data frame: same num of dimensions taht are available inside of our feature data frame


model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid')) # bool value to return whether the client is churn or not
# Final Layer Shape
# the num of units in your last layer dictates what your output will look like.
# One unit will mean that we wonly get one num back, sophisticated models for other tasks often have multiple units in the final layer



model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy') # tell tff how to train, what metrics, what optimizer
# loss: bias of our estimation
# optimizer: how to search
# metrics: evaluate how well our model is performing



# 3. Fit, Predict and Evalute
model.fit(X_train, y_train, epochs=50, batch_size=32)

y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]


# 4. Saving and Reloading
model.save('tfmodel')
# del model
#model = load_model('tfmodel')
