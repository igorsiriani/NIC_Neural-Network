import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from tensorflow import keras


def train_test_perceptron(epoch):
    # importa o dataset
    df = pd.read_csv('./../dataset/iris.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    # extrai a coluna de labels
    labels = df.pop('species')
    # converte a coluna de labels em one-hot encoding
    labels = pd.get_dummies(labels)

    # converte o dataframe em numpy arrays
    X = df.values
    y = labels.values

    # divide 15% para testes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # divide 15% para validação
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765)

    # plota os tamanhos de sepala e petala do conjunto de treinamento
    # plt.subplot(2, 2, 1)
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm.viridis)
    # plt.xlabel('sepal length (cm)')
    # plt.ylabel('sepal width (cm)')
    #
    # plt.subplot(2, 2, 2)
    # plt.scatter(X_train[:, 2], X_train[:, 3], c=y_train, cmap=cm.viridis)
    # plt.xlabel('petal length (cm)')
    # plt.ylabel('petal width (cm)')

    # cria o modelo
    model = keras.Sequential()
    model.add(keras.layers.Dense(16, activation='relu', input_shape=(4,)))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.summary()

    # compila o modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # treina o modelo
    history = model.fit(X_train, y_train,
              batch_size=12,
              epochs=epoch,
              validation_data=(X_val, y_val))

    # plota a evolução da accuracy por época
    # plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

    # Evaluate the model on the test data
    _, accuracy = model.evaluate(X_test, y_test)
    return accuracy


def main():
    result_list = []

    for i in range(0, 1):
        result = train_test_perceptron(200)
        result_list.append(result)

    print('Solução máxima: ', max(result_list))
    print('Solução mínima: ', min(result_list))
    print('Solução média: ', np.mean(result_list))
    print('Solução padrão: ', np.std(result_list))


if __name__ == '__main__':
    main()
