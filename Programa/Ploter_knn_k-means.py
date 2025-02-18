import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib.colors import ListedColormap
from matplotlib import colors
import matplotlib.pyplot as plt
import mysql.connector



# Declaración de datos y variables
X = []
y = []
grafico_3d = []
conteo=0
valor_KNN=4     #valor de KNN
valor_KMEANS=6  #valor de K-Means (rango de colores aplicado de 3 a 6)
valor_GMM=3    #valor de GMM
N=20000        #cantidad de datos a utilizar

#-----Extracción de datos de la Base de Datos-----

# Credenciales de conexión a la base de datos
config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'X',
    'database': 'datos_sensor'
}

# Conexión a la base de datos
cnx = mysql.connector.connect(**config)
cursor = cnx.cursor()

# Consulta para extraer los datos necesarios de la tabla
query = "SELECT id, co2, temp, hum, presion FROM sensor1"
cursor.execute(query)

# Recorrer los resultados de la consulta
for (id, co2, temp, hum, presion) in cursor:

    if presion != 0:
        #-----Datos extraidos-----
        Tactual = float(temp)           #°C - Temperatura extraida por el sensor en °C
        Tmedida= Tactual + 273.15       #°K - Tmedida = temperatura absoluta actual, °C + 273,15
        ppmC02medido = float(co2)       #ppm - Valor de ppm C02 medido por el sensor
        Pmedida = float(presion)        #hPa - pmedida = Presión actual, en las mismas unidades que la presión de referencia (no corregida al nivel del mar)
        Humedad = float(hum)            #g/m^3 - Humedad extraida por el sensor en gramos / metros cúbicos

        #-----Datos de referencia-----
        Tref = 298.15               #°K // Tref = temperatura de referencia, generalmente 25°C, convertida a absoluta (298,15 para °C)
        Pref = 1013.207             #hPa // pref = presión barométrica de referencia, normalmente a nivel del mar (Otros valores: Pref=29.92 #Hg / Pref=760 #mm Hg / Pref=14.6959 #psi) 

        #-----Cálculos-----       
        ppmC02corregido = ppmC02medido * ((Tmedida*Pref)/(Pmedida*Tref))
        #print(f"ID: {id}, Temperatura: {Tmedida}, Cálculo: {round(ppmC02corregido)}")

        # Agregar los datos calculados al arreglo X y al arreglo grafico_3d
        X.append([ppmC02corregido,Tmedida])
        grafico_3d.append([ppmC02corregido,Tmedida,Humedad])

        # Agregar las etiquetas de los valores calculados al arreglo Y
        if 0 <= ppmC02corregido < 500:
            y.append(0)
        elif 500 <= ppmC02corregido < 700:
            y.append(1)
        elif 700 <= ppmC02corregido < 1000:
            y.append(2)
        elif 1000 <= ppmC02corregido < 2500:
            y.append(3)
        elif 2500 <= ppmC02corregido < 5000:
            y.append(4)
        else:
            y.append(5)
            print("Guardando valor de C02 ¡MUY ALTO!", ppmC02corregido)

        # Limito la cantidad de datos
        conteo+=1
        if conteo == N:
            break

# Cerrar la conexión a la base de datos
cursor.fetchall()
cursor.close()
cnx.close()  

#---------------Cálculos de KNN y su gráfico 2D---------------

# Conversor de la matriz a un array Numpy
X = np.array(X)
y=np.array(y)

# Genera datos aleatorios
#X = np.random.rand(100, 2)     # 100 puntos con dos características cada uno
#y = np.random.randint(0, 2, N) # etiquetas aleatorias (0 o 1)

# Mostrar los valores utilizados
#print(X) 
print("Tamaño de X:",len(X))
print("Tamaño de y:",len(y))
print("Valor máximo en X: ", np.amax(X))

# Crea y entrena el clasificador KNN
print("Calculando KNN...")
knn = KNeighborsClassifier(n_neighbors=valor_KNN)
knn.fit(X, y)

x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predecir la clase de todos los puntos de la malla
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Grafica los colores de la región de decisión del clasificador KNN
plt.figure()
plt.pcolormesh(yy, xx, Z, cmap='coolwarm')

#otros colores: 'jet' 'viridis' 'Pastel1' 'coolwarm'

# Grafica los puntos
plt.scatter(X[:, 1], X[:, 0], c=y, edgecolor='k', cmap='coolwarm')
plt.ylabel('C02 Corregido ppm')
plt.xlabel('Temperatura °K')
plt.title(f'Clasificador KNN K={valor_KNN}')

plt.show()
#plt.savefig("grafico_knn.png", dpi=300, bbox_inches='tight')


#---------------Cálculos de K-Means y su gráfico 2D---------------

# Conversor de la matriz a un array Numpy
X = np.array(X)

# Crea y entrena el algoritmo K-means
print("Calculando K-Means...")
kmeans = KMeans(n_clusters=valor_KMEANS, n_init=5, random_state=4)
kmeans.fit(X)

# Obtiene las etiquetas de los clústeres y los centroides
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Crea una malla para la visualización
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predecir la clase de todos los puntos de la malla
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()


# Grafica los colores de la región de decisión del algoritmo K-Means y los puntos
if valor_KMEANS == 3:
    color_list = ['#58D68D', '#F7DC6F', '#EC7063']
elif valor_KMEANS == 4:
    color_list = ['#F7DC6F', '#F39C12', '#EC7063', '#58D68D']
elif valor_KMEANS == 5:
    color_list = ['#F39C12', '#58D68D', '#F7DC6F', '#EC7063', 'yellowgreen']
elif valor_KMEANS == 6:
    color_list = ['yellowgreen', '#F39C12', '#58D68D', '#C0392B', '#EC7063','#F7DC6F']
else:
    plt.pcolormesh(yy, xx, Z, cmap='coolwarm')
    plt.scatter(X[:, 1], X[:, 0], c=labels, edgecolor='k', cmap='coolwarm')

if valor_KMEANS <= 6:
    plt.pcolormesh(yy, xx, Z, cmap=ListedColormap(color_list))
    plt.scatter(X[:, 1], X[:, 0], c=labels, edgecolor='k', cmap=ListedColormap(color_list))
 
# Grafica los centroides de cada sección
plt.scatter(centroids[:, 1], centroids[:, 0], marker='*', s=300, c='black', edgecolor='k')

#Configura los títulos y etiquetas de los ejes
plt.xlabel('Temperatura °K')
plt.ylabel('C02 Corregido ppm')
plt.title(f'Algoritmo K-means K={valor_KMEANS}')

plt.show()
#plt.savefig("grafico_k-means_2D.png", dpi=300, bbox_inches='tight')

#---------------Cálculos de K-Means y su gráfico 3D---------------

# Conversión de la lista de datos en una matriz Numpy para facilitar su manipulación
grafico_3d = np.array(grafico_3d)

# Creación de una figura y un eje 3D para la visualización
print("Calculando K-Means 3D...")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Instanciación del algoritmo K-Means con el número de clusters especificado
kmeans = KMeans(n_clusters=valor_KMEANS, n_init=10, random_state=4).fit(grafico_3d)

# Obtención de los centroides de los clusters
centroids = kmeans.cluster_centers_

# Asignación de una etiqueta a cada punto según el cluster al que pertenece
labels = kmeans.labels_

#Grafica los puntos en 3D y defino colores de las etiquetas
if valor_KMEANS == 3:
    color_dict = {0: 'lime', 1: 'yellow', 2: 'red'}
elif valor_KMEANS == 4:
    color_dict = {3: 'lime', 0: 'yellow', 1: 'orange', 2: 'red'}
elif valor_KMEANS == 5:
    color_dict = {4: 'lime',1: 'yellowgreen', 3: 'yellow',2: 'red',0: 'orange'}
elif valor_KMEANS == 6:
    color_dict = {2: 'lime',0: 'yellowgreen', 5: 'yellow',4: 'red',1: 'orange',3: 'brown'}
else:
    ax.scatter(grafico_3d[:, 0], grafico_3d[:, 1], grafico_3d[:, 2], c=labels, cmap='coolwarm', edgecolor='k')

if valor_KMEANS <= 6:
    ax.scatter(grafico_3d[:, 0], grafico_3d[:, 1], grafico_3d[:, 2], c=[color_dict[label] for label in labels], edgecolor='k')

#Grafica los centroides en 3D
#ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', s=300, c='black', edgecolor='k')

#Configura los títulos y etiquetas de los ejes
ax.set_xlabel('CO2 Corregido ppm')
ax.set_ylabel('Temperatura °K')
ax.set_zlabel('Humedad g/m^3')
ax.set_title(f'Algoritmo K-means en 3D K={valor_KMEANS}')

plt.show()
#plt.savefig("grafico_k-means_3D.png", dpi=300, bbox_inches='tight')


#---------------Cálculos de GMM y su gráfico 2D---------------

# Crear instancia del modelo GMM y ajustar a los datos
print("Calculando GMM 2D...")
gmm = GaussianMixture(n_components=valor_GMM, random_state=4)
gmm.fit(X)

# Predecir la clase de cada punto en los datos
labels_gmm = gmm.predict(X)

# Graficar los datos con los colores basados en las etiquetas de clase predichas por GMM
plt.scatter(X[:,1], X[:,0], c=labels_gmm, cmap='jet')
plt.ylabel('C02 Corregido ppm')
plt.xlabel('Temperatura °K')
plt.title(f'Algoritmo GMM 2D K={valor_GMM}')

plt.show()
#plt.savefig("grafico_GMM_2D.png", dpi=300, bbox_inches='tight')

#---------------Cálculos de GMM y su gráfico 3D---------------

# Crear instancia del modelo GMM y ajustar a los datos
print("Calculando GMM 3D...")
gmm = GaussianMixture(n_components=valor_GMM, random_state=4)
gmm.fit(grafico_3d)

# Predecir la clase de cada punto en los datos
labels_gmm = gmm.predict(grafico_3d)

# Graficar los datos en 3D con los colores basados en las etiquetas de clase predichas por GMM
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(grafico_3d[:,0], grafico_3d[:,1], grafico_3d[:,2], c=labels_gmm, cmap='jet')

ax.set_xlabel('CO2 Corregido ppm')
ax.set_ylabel('Temperatura °K')
ax.set_zlabel('Humedad g/m^3')
ax.set_title(f'Algoritmo GMM 3D K={valor_GMM}')

plt.show()
#plt.savefig("grafico_GMM_3D.png", dpi=300, bbox_inches='tight')