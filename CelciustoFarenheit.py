import tensorflow as tf
import numpy as np

Celcius = np.linspace(-100, 500, num=1000)
Farenheit = Celcius*9/5+32
#Farenheit[250] = 1000

# capa = tf.keras.layers.Dense(units=1, input_shape=[1])
# modelo = tf.keras.Sequential([capa])
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
oculta3 = tf.keras.layers.Dense(units=3)
oculta4 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, oculta3, oculta4, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print("Awanta vara en lo que esta el bisne...")
historial = modelo.fit(Celcius, Farenheit, epochs=700, verbose=False)
print("Camara ya quedo papito!")

import matplotlib.pyplot as plt

plt.xlabel("# EPOCA")
plt.ylabel("Perdida")
plt.plot(historial.history["loss"])

prediccion = float(input("Ingresa los grados Celcius que quieras ingresar: "))
resultado = modelo.predict([prediccion])
print(f"El resultado es {str(resultado)} farenheit!")
