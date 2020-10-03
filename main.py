import tensorflow as tf
import export_image as ei
import matplotlib.pyplot  as plt
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
L=[1,2,3]
print(L[:1])
ei.img_out(x_train[2],"img/image2.png")


plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show()



x_train, x_test = x_train / 255.0, x_test / 255.0




model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(10000, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
