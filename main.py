import tensorflow as tf
import export_image as ei
import matplotlib.pyplot  as plt
import numpy as np
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

ei.img_out(x_train[2],"img/image2.png")

#petit test git avec ASMAOU

"""
plt.figure()
plt.imshow(x_test[0])
plt.colorbar()
plt.grid(False)
plt.show()
"""
print(tf.__version__)

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.load_model("./Sauvegarde/")

"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)

model.save("./Sauvegarde/")
"""


img = x_test[0]
img = np.expand_dims(img,0)
predictionssignle = model.predict(img)
print(predictionssignle)

#model.evaluate(x_test, y_test)
