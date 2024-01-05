import numpy as np
import os

# TensorFlow
import tensorflow as tf

print(tf.__version__)

# 1. Generar dataset para X con 450 valores
X = np.linspace(-10.0, 10.0, 450)

# 2. usamos "Y" y = -632x  + 44 
y = -632 * X + 44 + np.random.normal(0, 5, len(X))

# 3. Entrenar para el número de epochs asignados que son 200
tf.keras.backend.clear_session()
linear_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], name='Single')
])

linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.mean_squared_error)
print(linear_model.summary())
#aqui añadimos los epochs
linear_model.fit(X, y, epochs=200)

# 4. Probar el modelo con predict para 16 valores asignados
test_values = np.linspace(-10.0, 10.0, 16).reshape((-1, 1))
predictions = linear_model.predict(test_values).flatten()
print("Predictions:", predictions)
# 5. Exportar el modelo con el nombre asignado en modelname
export_path = './ModelEc8/1/'
tf.saved_model.save(linear_model, os.path.join('./', export_path))

# 6. Extraer los pesos para W y b e imprimirlos
weights, biases = linear_model.layers[0].get_weights()
print(f"Weights (W): {weights.flatten()[0]}")
print(f"Biases (b): {biases[0]}")

print ("test_values", test_values)
