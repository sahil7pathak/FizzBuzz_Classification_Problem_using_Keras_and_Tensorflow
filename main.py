
#Logic Based FizzBuzz Function [Software 1.0]

import pandas as pd

def fizzbuzz(n):

    # Logic Explanation
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return 'Other'

#Creating Training and Testing Datafiles

#Create Training and Testing Datasets in CSV Format
def createInputCSV(start,end,filename):

    # Why list in Python?
    ''' Ans :- Lists are ordered and can be sorted, also we can add or remove elements from the lists. They also support
               duplicate elements. In our outputData, which is a list, we have many duplicate member such as 'Fizz', 'Buzz',
               'FizzBuzz' and 'Other'. Data can be retreived from a particular location easily '''
    inputData   = []
    outputData  = []

    # Why do we need training Data?
    ''' Ans :- Training means creating/ learning the model. We need training data because it is labeled data
               that we use to train our machine learning model. As we show the labeled examples to the model, it
               enables the model to gradually learn the relationships between features and labels '''
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))

    # Why Dataframe?
    ''' Ans :-  Dataframes are two-dimensional labeled data structures with columns of potentially different types.
                Manipulating the data is easier, also selecting or replacing columns and indices
                to reshaping your data can be done by Dataframes. Since we are dealing with csv files i.e training.csv
                and testing.csv, also our output is getting stored in output.csv file, managing all these files, reading their
                values, updating them becomes easier through dataframes '''
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData

    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)

    print(filename, "Created!")

#Processing Input and Label Data
def processData(dataset):

    # Why do we have to process?
    ''' Ans :- As mentioned eariler, we know that a model gradually learn the relationships between features and labels.
               We have our input as an integer number (single value) and we want to feed it to ur neural network. Having
               just single value as a feature for a neural network is not an effective way of training the model. We need
               some more features to be fed to the neural network, thus, we convert that integer number into a
               binary number. As soon as we do that, we can interpret each digit in that binary sequence as a feature and
               then that feature can be given as an input to each neuron in the neural network.
               Thus, we need to process the integer valued data to its binary representation '''
    data   = dataset['input'].values
    labels = dataset['label'].values

    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)

    return processedData, processedLabel

def encodeData(data):

    processedData = []

    for dataInstance in data:

        # Why do we have number 10?
        ''' Ans :- Since our training data ends at a integer value 1000, at the most 10 digits are needed
                   in order to represent 1000 as a binary number '''
        processedData.append([dataInstance >> d & 1 for d in range(10)])

    return np.array(processedData)

from keras.utils import np_utils

def encodeLabel(labels):

    processedLabel = []

    for labelInstance in labels:
        if(labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel),4)

#Model Definition
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras import optimizers
import numpy as np

input_size = 10
drop_out = 0.2
first_dense_layer_nodes  = 256
second_dense_layer_nodes = 4

def get_model():

    # Why do we need a model?
    ''' Ans :- A labeled example contains both feature(s) and the label. Model defines the relationship
               between feature(s) and label. We use labeled examples to train the model. Once we have trained our model
               with labeled examples, we use that model to predict the label on unlabeled examples '''

    # Why use sequential model with layers?
    ''' Ans :- Sequential model is a linear stack of layers. So, each layer has unique input and output, and those
               inputs and outputs have unique input shape and output shape. As we have unique input values to
               each neuron as binary digits, we use sequential model '''

     # Why use Dense layer and then activation?
    ''' Ans :- Dense layer is simply a layer where each unit or neuron is connected to each neuron in the next layer.
               Without non-linearity our network is just a linear classifier and not able to acquire nonlinear relationships.
               To model a nonlinear problem, we can directly introduce a nonlinearity through an activation function.
               Also, the following statement can be written as model.add(Dense(first_dense_layer_nodes, activation='relu', input_dim=input_size)) '''

    model = Sequential()

    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))

    # Why dropout?
    ''' Ans :- Dropout is a regularization technique, which aims to reduce the complexity of the model with the
               goal to prevent overfitting. The Dropout method in keras.layers module takes in a float value between 0 and 1,
               which is the fraction of the neurons to drop. Here we have set drop_out=0.2 '''
    model.add(Dropout(drop_out))

    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    # Why Softmax?
    ''' Ans :- Softmax activation function is used to calculate probability of each target label in
               neural network for classification. It calculates probability of label for the
               input data, and answers one with the highest probability. Since we want to classify the values in the test
               data as per 'Fizz', 'Buzz', 'FizzBuzz' or 'Other', the softmax function is used, so that we can interpret
               the final output as a probability vector. It classifies test examples to each class and is often used
               in the final layer of neural network based classifier '''
    model.summary()

    # Why use categorical_crossentropy?
    ''' Ans :- The loss function to be minimized on softmax output layer equipped neural nets is the
               cross-entropy loss function. Also as per the keras documentation, "When using the categorical_crossentropy loss,
               our targets should be in categorical format (e.g. if we have 10 classes, the target for each sample should
               be a 10-dimensional vector that is all-zeros except for a 1 at the index corresponding to the class of the sample)",
               and indeed we want to classify the values as 'Fizz', 'Buzz', 'FizzBuzz' or 'Other' which are categorical in nature'''

    #sgd = optimizers.SGD(lr=0.001)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')

#Creating Model
model = get_model()

#Run Model
validation_data_split = 0.2
num_epochs = 10000
model_batch_size = 128
tb_batch_size = 32
early_patience = 100

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

# Read Dataset
dataset = pd.read_csv('training.csv')

# Process Dataset
processedData, processedLabel = processData(dataset)
history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )

#Training and Validation Graphs
import matplotlib.pyplot as plt
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))
plt.show();

#Testing Accuracy [Software 2.0]
def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"

wrong   = 0
right   = 0

testData = pd.read_csv('testing.csv')

processedTestData  = encodeData(testData['input'].values)
processedTestLabel = encodeLabel(testData['label'].values)
predictedTestLabel = []

for i,j in zip(processedTestData,processedTestLabel):
    y = model.predict(np.array(i).reshape(-1,10))
    predictedTestLabel.append(decodeLabel(y.argmax()))

    if j.argmax() == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))

# Please input your UBID and personNumber
testDataInput = testData['input'].tolist()
testDataLabel = testData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "sahilsuh")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50289739")

predictedTestLabel.insert(0, "")
predictedTestLabel.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabel

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')
