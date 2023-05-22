from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


def train_test_perceptron(epoch):
    # importa o dataset
    fashion_mnist = keras.datasets.fashion_mnist
    # divide 15% para testes
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # divide 15% para validação
    train_images, val_images, train_labels, val_label = train_test_split(train_images, train_labels, test_size=0.175, random_state=45)

    #devine labels
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # downscale das imagens
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    val_images = val_images / 255.0

    # cria o modelo
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # compila o modelo
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # treina o modelo
    history = model.fit(train_images, train_labels, batch_size=64, epochs=epoch, validation_data=(val_images, val_label))

    # plota a evolução da accuracy por época
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

    # Evaluate the model on the test data
    _, accuracy = model.evaluate(test_images, test_labels)
    return accuracy


def main():
    result_list = []

    for i in range(0, 1):
        result = train_test_perceptron(150)
        result_list.append(result)

    print('Solução máxima: ', max(result_list))
    print('Solução mínima: ', min(result_list))
    print('Solução média: ', np.mean(result_list))
    print('Solução padrão: ', np.std(result_list))


if __name__ == '__main__':
    main()
