import tensorflow as tf
import export_image as ei
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
#print(y_test[0])
#x_train, x_test = x_train / 255.0, x_test / 255.0

f = open("x_train0.txt", "w")
for i in range(0,28):
    for j in range(0,28):
        if (x_test[0][i][j] == 0):
            f.write('...')
        else:
            f.write(("000"+str(x_test[0][i][j]))[-4:-1])
    f.write('\n')

f.close()

ei.img_out(x_train[1],"img/image1.png")

"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
"""