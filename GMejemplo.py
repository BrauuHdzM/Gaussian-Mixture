from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from numpy import random 
from pandas import DataFrame

#make_blobs genera conjuntos de datos artificiales para pruebas y demostraciones
random.seed(234)
x, _ = make_blobs(n_samples=2000, centers=4, cluster_std=1)

#Se muestra un grafico de dispersion de dos variables, creando una figura con un tamaño especificado en pixels (800 pixels de ancho y 600 pixels de alto)
plt.figure(figsize=(8, 6))

#Los argumentos de la función scatter son dos columnas de la matriz x.  
#El operador : se utiliza para seleccionar todas las filas de la matriz, 
#El índice 0 o 1 se utiliza para seleccionar la primera o la segunda columna, respectivamente.
#Indexado por columnas": El primer índice se refiere a las filas del arreglo, y el segundo índice se refiere a las columnas. 
#Si se omite el primer índice,se seleccionan todas las filas.
plt.scatter(x[:,0], x[:,1])

#Show muestra la figura en pantalla
plt.show()

#Se crea un objeto de la clase GaussianMixture de scikit-learn y se llama al método 'fit' para entrenar el modelo en el conjunto de datos 'x'
gm = GaussianMixture(n_components=4).fit(x)
#Se obtiene los centros de cada componente gaussiano a partir del atributo 'means_' del objeto gm
centers = gm.means_
print(centers)

#El método 'predict' del objeto 'gm' predice a qué grupo pertenece cada punto de datos en 'x'.
pred = gm.predict(x)

#Se muestran en pantalla los puntos junto con los centros de cada grupo
plt.figure(figsize=(8, 6))
plt.scatter(x[:,0], x[:,1], label="data")
plt.scatter(centers[:,0], centers[:,1], c='r', label="centers")
plt.legend()
plt.show()

#Se crea un DataFrame a partir de 'x', 'y', y 'label'. Cada una de estas entradas es una columna en el DataFrame
df = DataFrame({'x':x[:,0], 'y':x[:,1], 'label':pred})
#Se usa el método groupby del objeto DataFrame para dividir los datos en grupos basados en el valor de la columna 'label'.
groups = df.groupby('label')

#Creamos la figura y los ejes de coordenadas
ig, ax = plt.subplots(figsize=(8, 6))
#Se itera sobre cada uno de los grupos de datos. 
for name, group in groups:
#scatter dibuja un punto en el gráfico para cada fila del grupo de datos actual y se etiquetan con el nombre del grupo, que se obtiene de la variable name
    ax.scatter(group.x, group.y, label=name)
ax.legend()
plt.show()


#Generación del conjunto de datos para pruebas
random.seed(234)
xpred, _  = make_blobs(n_samples=150, centers=4, cluster_std=1)

#se muestra en pantalla el conjutno de datos creado
plt.figure(figsize=(8, 6))
plt.scatter(xpred[:,0], xpred[:,1])
plt.show()

#Se predice a que grupo pertenece cada elemento del conjunto de pruebas
pred2= gm.predict(xpred)

#Se muestran ambos conjuntos de datos: de entrenamiento y pruebas, etiquetados al grupo al que pertenecen
df = DataFrame({'x':x[:,0], 'y':x[:,1], 'label':pred})
groups = df.groupby('label')
df1 = DataFrame({'x':xpred[:,0], 'y':xpred[:,1], 'label':pred2})
groups1 = df1.groupby('label')

fig, ax = plt.subplots(figsize=(8, 6))
for name, group in groups:
    ax.scatter(group.x, group.y, label=name)
for name, group in groups1:
    ax.scatter(group.x, group.y, label=name, marker="^")
ax.legend()
plt.show()

#Se imprime la verosimilitud logarítmica promedio (average log-likelihood)
print("log-likehood: ", gm.score(x))
print("log-likehood prueba: ", gm.score(xpred))

