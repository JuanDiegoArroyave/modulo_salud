#####librerias
# Tratamiento
import numpy as np
import pandas as pd

# Graficos
import matplotlib.pyplot as plt


# Para tratamiento de imagenes
from PIL import Image

# Exportacion de archivos
import joblib as joblib

# Importar metadata de csv a dataframe
metadata = pd.read_csv('metadata.csv') # Obtiene metadatos de imagenes

images_names = metadata['image'].tolist() # Se agregan nombres a lista

# ruta_images = 'images'

images_paths = ['images\\'+str(i) for i in images_names] # Ruta de las imagenes

sizes = (256, 256) # Tamano de redimension para las imagenes

images = [np.array(Image.open(img_paht).convert("L").resize(sizes)) for img_paht in images_paths]

x = np.stack(images)

estado = metadata['class'].tolist()

# Sera 1 cuando sea tumor y 0 cuando sea normal
y = np.array([1 if item == 'tumor' else 0 for item in estado])

# Prueba tumor
y[0]
plt.imshow(x[0],cmap='gray')

# Prueba no tumor
y[4000]
plt.imshow(x[4000],cmap='gray')

# Exportar x e y a la carpeta de salidas
joblib.dump(x, 'salidas\\x.joblib')

joblib.dump(y, 'salidas\\y.joblib')