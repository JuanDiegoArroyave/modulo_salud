#####librerias
import numpy as np

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

##### para afinamiento #########
import keras_tuner as kt

# Para cargar salidas
import joblib
import os

# Separamiento entre entrenamiento y testeo
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix

import seaborn as sns


# Poner la ruta de cada maquina en el proyecto
os.chdir(r'C:\Users\amqj1\OneDrive\Escritorio\Codigos JD\Analitica III\Salud\mod_Salud')

######## cargar datos #####
# Cambiar rutas segun la maquina
x = joblib.load('salidas\\x.joblib')
y = joblib.load('salidas\\y.joblib')

# Separar datos entre entrenamiento y testeo
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


x_train.shape
x_train[1]
x_train.max()
x_train.min()

y_train.shape


plt.imshow(x_train[1],cmap='gray')
y_train[1]


# Normalizar datos

x_trains = x_train / 255
x_tests = x_test / 255

x_train[1]
x_train.max()

f=x_train.shape[1]
c=x_train.shape[2]
fxc= f*c

#########################################################
######### red convolucional #############################
#########################################################


# Inicializamos el modelo secuencial
model = Sequential()

# Primera capa convolucional
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(f, c, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Segunda capa convolucional
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Tercera capa convolucional
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Cuarta capa convolucional
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Aplanado
model.add(Flatten())

# Capa densa completamente conectada
model.add(Dense(units=256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Capa de salida
model.add(Dense(units=1, activation='sigmoid'))  # Sigmoid para clasificación binaria

# Compilación del modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()


# Entrenamos el modelo
history = model.fit(
    x_trains,          # Datos de entrenamiento
    y_train,          # Etiquetas de entrenamiento
    validation_data=(x_tests, y_test),  # Datos y etiquetas de validación
    epochs=10,        # Número de épocas, puedes ajustar según el desempeño
    batch_size=30,    # Tamaño del lote, ajustar según la memoria disponible
    verbose=1         # Nivel de salida en pantalla (1 muestra más detalles, 0 solo muestra el progreso)
)

# Exportar modelo
joblib.dump(history, 'salidas\\history.joblib')

# Cargamos el modelo guardado
history = joblib.load('salidas\\history.joblib')

# Evaluamos el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(x_tests, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")



# Gráfica de precisión
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Gráfica de pérdida
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


predicciones = model.predict(x_tests)

np.max(predicciones)
np.min(predicciones)
np.mean(predicciones)
np.median(predicciones)
np.std(predicciones)


############################################################
#########  Afinamiento de hiperparámetros ##################
###########################################################



def build_model(hp):
    model = Sequential()
    
    # Primera capa convolucional
    model.add(Conv2D(
        filters=hp.Choice('filters_1', values=[32, 64, 128]),
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(f, c, 1)
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Choice('dropout_1', values=[0.25, 0.3, 0.5])))

    # Segunda capa convolucional
    model.add(Conv2D(
        filters=hp.Choice('filters_2', values=[64, 128, 256]),
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Choice('dropout_2', values=[0.25, 0.3, 0.5])))

    # Tercera capa convolucional opcional
    if hp.Boolean('extra_conv_layer'):
        model.add(Conv2D(
            filters=hp.Choice('filters_3', values=[128, 256]),
            kernel_size=(3, 3),
            activation='relu'
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(hp.Choice('dropout_3', values=[0.25, 0.3, 0.5])))

    # Aplanado
    model.add(Flatten())

    # Capa densa completamente conectada
    model.add(Dense(units=hp.Choice('units', values=[128, 256]), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Choice('dropout_dense', values=[0.5, 0.6])))

    # Capa de salida
    model.add(Dense(1, activation='sigmoid'))

    # Compilación del modelo
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 5e-5])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Configuramos la búsqueda de hiperparámetros
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Número de configuraciones de hiperparámetros que probará
    executions_per_trial=1,  # Ejecuta una vez por cada combinación
    directory='my_dir',
    project_name='mod_Salud'
)

# Ejecutamos el tuning
tuner.search(x_trains, y_train, epochs=10, validation_data=(x_tests, y_test))

# Guardar el modelo ganador
# Obtener el mejor modelo
best_model = tuner.get_best_models(num_models=1)[0]

# Guardar el modelo en un archivo
best_model.save('best_model.h5')

# Cargar el modelo guardado en el futuro
loaded_model = load_model('best_model.h5')

# Ahora puedes hacer predicciones con el modelo cargado
predictions = loaded_model.predict(x_tests)  # x_test son tus datos de entrada para las predicciones

# guardar predicciones con joblib
joblib.dump(predictions, 'salidas\\predictions.joblib')

# Cargar predicciones con joblib
predictions = joblib.load('salidas\\predictions.joblib')

# Relizar una matriz de confusion para el modelo generado
y_pred = loaded_model.predict(x_tests)
y_pred = np.round(y_pred).astype(int)
cm = confusion_matrix(y_test, y_pred)
print(cm)
# graficar la matriz de confusion


sns.heatmap(cm, annot=True, fmt='d')
plt.title('Matriz de confusion')
plt.xlabel('Etiquetas predichas')
plt.ylabel('Etiquetas reales')

max(predictions)
min(predictions)
np.mean(predictions)
np.std(predictions)


# Ver accuracy de entrenamiento y testeo de best_model
train_loss, train_accuracy = loaded_model.evaluate(x_trains, y_train)
test_loss, test_accuracy = loaded_model.evaluate(x_tests, y_test)

print(f"Train Loss: {train_loss}")
print(f"Train Accuracy: {train_accuracy}")

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


#loaded_model y x_tests 
#y son las etiquetas reales
y_pred = loaded_model.predict(x_tests)

#inicializar listas para almacenar predicciones y valores reales segmentados
seg_1_pred, seg_2_pred, seg_3_pred = [], [], []
seg_1_true, seg_2_true, seg_3_true = [], [], []

#iterar sobre las predicciones continuas y valores reales para clasificar en segmentos
for pred, true in zip(y_pred, y_test):
    if 0 <= pred <= 0.25:
        seg_1_pred.append(np.round(pred).astype(int))  # Redondear después de segmentar
        seg_1_true.append(true)
    elif 0.25 < pred <= 0.75:
        seg_2_pred.append(np.round(pred).astype(int))
        seg_2_true.append(true)
    elif 0.75 < pred <= 1:
        seg_3_pred.append(np.round(pred).astype(int))
        seg_3_true.append(true)

#convertir listas a numpy arrays
seg_1_pred = np.array(seg_1_pred)
seg_2_pred = np.array(seg_2_pred)
seg_3_pred = np.array(seg_3_pred)
seg_1_true = np.array(seg_1_true)
seg_2_true = np.array(seg_2_true)
seg_3_true = np.array(seg_3_true)

#generar y graficar la matriz de confusión para cada segmento
for i, (segment_true, segment_pred, label) in enumerate(
    [(seg_1_true, seg_1_pred, 'Segmento 0-0.25'), 
     (seg_2_true, seg_2_pred, 'Segmento 0.25-0.75'), 
     (seg_3_true, seg_3_pred, 'Segmento 0.75-1')]
):
    if segment_true.size > 0 and segment_pred.size > 0:
        # Calcular la matriz de confusión
        cm = confusion_matrix(segment_true, segment_pred)
        
        # Graficar la matriz de confusión
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matriz de Confusión - {label}')
        plt.xlabel('Etiquetas Predichas')
        plt.ylabel('Etiquetas Reales')
        plt.show()
    else:
        print(f"No hay datos para el {label}")
