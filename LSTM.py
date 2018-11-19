'''
Kaihua Li

'''

from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Dropout

def dataProcess(filename):
    
    dropnan = True
    
    data = read_csv(filename,header=0, index_col=0)
    data = data.values
    
    encoder = LabelEncoder()
    data[:,4] = encoder.fit_transform(data[:,4])
    data = data.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    data = scaler.fit_transform(data)
    #print(data)
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(1, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, 1):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    finalData = concat(cols, axis=1)
    finalData.columns = names
    if dropnan:
        finalData.dropna(inplace=True)
        
    finalData.drop(finalData.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
    print(finalData)
    return finalData, scaler

def trainAndTest(data, trainHours):
    keep = 365 * 24
    data = data.values
    data = data[:keep,:]
    train = data[:trainHours, :]
    test = data[trainHours:, :]
    xTrain, yTrain = train[:, :-1], train[:, -1]
    xTest, yTest = test[:, :-1], test[:, -1]
    xTrain = xTrain.reshape((xTrain.shape[0], 1, xTrain.shape[1]))
    xTest= xTest.reshape((xTest.shape[0], 1, xTest.shape[1]))
    
    return xTrain, yTrain, xTest, yTest

def buildModel(trainData):
    
    model = Sequential()
    model.add(LSTM(27, input_shape=(trainData.shape[1], trainData.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(90, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(47))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model 

def plotResults(predicted_data, true_data, filename):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Original Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.savefig(filename+'.png')

def typeTrans(data):
    data2 = []
    for item in data:
        item = float(item)
        data2.append(item)
    data2 = pandas.Series(data2)
    return(data2)   

def accuracyMeasure(originalData, forecastData):
    x = originalData
    y = forecastData
    a = x - y
    a = abs(a)
    #print(a)
    meanAbsoluteErrorresult = a.mean()
    b = a*a
    rmseResule = (b.mean())**(1/2)
    x2 = []
    for m in x:
        #print(m)
        if m == 0:
            m = x.mean()
        x2.append(m)
    #print(m) 

    p = abs(100*a/x2)
    #print(p)
    mapeResult = p.mean()
    result = [meanAbsoluteErrorresult, rmseResule, mapeResult]
    #print ('MAE is', meanAbsoluteErrorresult,', RMSE is', rmseResule, ', MAPE is', mapeResult)
    return result

data, scaler = dataProcess('pollution.csv')
trainHours = 9 * 31 * 24 + 15 * 24
xTrain, yTrain, xTest, yTest = trainAndTest(data, trainHours)
model = buildModel(xTrain)
y = model.fit(xTrain, yTrain, epochs=50, batch_size=72, validation_data=(xTest, yTest), verbose=2, shuffle=False)
plt.plot(y.history['loss'], label='train')
plt.legend()
plt.savefig('loss.png')
lstmResult = model.predict(xTest)
xTest = xTest.reshape((xTest.shape[0], xTest.shape[2]))
lstmResult = concatenate((lstmResult, xTest[:, 1:]), axis=1)
lstmResult = scaler.inverse_transform(lstmResult)
lstmResult = lstmResult[:,0]
yTest = yTest.reshape((len(yTest), 1))
yTest = concatenate((yTest, xTest[:, 1:]), axis=1)
yTest = scaler.inverse_transform(yTest)
yTest = yTest[:,0]
plotResults(lstmResult,yTest,'LSTM Predictions')
aMeasure = accuracyMeasure(yTest,lstmResult)
print('Accuracy measure for LSTM is: MAE:', aMeasure[0], ', RMSE:',aMeasure[1],", MAPE:", aMeasure[2] )
