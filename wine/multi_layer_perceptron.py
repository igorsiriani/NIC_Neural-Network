import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


# Zscore par normalizar os valores
def zscore(X):

    # recupera o formato da matrix
    [nX, mX] = X.shape

    # calcula a média de cada coluna de X
    XMean = np.mean(X, axis=0)

    # calcula o desvio padrão de cada coluna
    XStd = np.std(X, axis=0, ddof=1)

    # subtrai a média de cada coluna
    zX = X - np.kron(np.ones((nX, 1)), XMean)  # Z = [X - mX]

    # divide pelo desvio padrão
    Zscore = np.divide(zX, XStd)

    return Zscore


def train_test_perceptron(epoch):
    # importa o dataset
    wines = pd.read_csv('./../dataset/wine.csv', names=[
        'class',
        'alcohol',
        'malic_acid',
        'ash',
        'ash_alcalinity',
        'magnesium',
        'total_phenols',
        'flavanoids',
        'nonflavanoids',
        'proanthocyanins',
        'color',
        'hue',
        'dilution',
        'profile'])

    # extrai a coluna de classes
    labels = wines.pop('class')
    # converte a coluna de labels em one-hot encoding
    labels = pd.get_dummies(labels)

    X = wines.values
    X = zscore(X)
    y = labels.values

    # divide 15% para testes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=45)

    # divide 15% para validação
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765, random_state=45)

    print('X_train.shape: ', X_train.shape)
    print('y_train.shape: ', y_train.shape)

    # cria o modelo
    model = keras.Sequential()
    model.add(keras.layers.Dense(12, activation='relu', input_shape=(13,)))
    model.add(keras.layers.Dense(9, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))
    print(model.summary())

    # compila o modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # treina o modelo
    history = model.fit(X_train, y_train,
                        batch_size=12,
                        epochs=epoch,
                        validation_data=(X_val, y_val))

    # # plota a evolução da accuracy por época
    # plt.plot(history.history['accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    # plt.show()

    # Evaluate the model on the test data
    _, accuracy = model.evaluate(X_test, y_test)
    return accuracy


def main():
    result_list = []

    for i in range(0, 200):
        result = train_test_perceptron(10)
        result_list.append(result)

    print('Solução máxima: ', max(result_list))
    print('Solução mínima: ', min(result_list))
    print('Solução média: ', np.mean(result_list))
    print('Solução padrão: ', np.std(result_list))


if __name__ == '__main__':
    main()
