#!/usr/bin/env python
# coding: utf-8

# # Listas, Arreglos y Numpy
# 
# Numpy es una librería de computación numérica para ciencia de datos, vendría siendo como el matlab (a nivel de operaciones) pues lo hace de forma eficiente.
# 
# Aqui aprenderemos a utilizar numpy importandolo para
# - Crear y manipular arreglos
# - Salvar arreglos
# - Cargar dichos arreglos
# - Partir y seleccionar sub arreglos
# - Indexamiento booleano o cambiar el subset del array
# - Ordenar los arreglos
# - Hacer operaciones de elementos

# Para instalar numpy lo puede hacer con pip o conda, pero con conda debería venir instalado intrinsecamente.  Podemos ver la version de numpy instalada con el siguiente comando:
# 
#     pip show numpy
#     
# <img src="pipshownumpy.PNG" width="850"/>

# Si quisieramos tambien podemos instalar una versión de numpy inferior pero existente, por ejemplo, para instalar una versión anterior podemos hacer el downgrade como sigue:
# 
#     conda install numpy==1.13.0

# Exortamos que se revise en cada momento que no tenga seguridad la librería numpy en su [manual en línea](https://numpy.org/).

# #### ¿Porqué debemos utilizar numpy?
# 
# No es necesario pero esta librería tiene más de lo que necesitamos ademas de algebra de vectores.  Veamos un ejemplo de uso de numpy.

# In[1]:


# importamos librerias
import time as t
import numpy as np


# In[ ]:


# con funciones estandares
x = np.random. random(1000000) # genera un arreglo aleatorio de datos enteros
start = t.time()           # toma el tiempo inicial
sum(x)/len(x)               # promedio
end = t.time() - start     # calcula cuando demora el ciclo
print('function elapsed ',end)


# In[ ]:


# con numpy
x = np.random. random(1000000) # genera un arreglo aleatorio de datos enteros
start = t.time()           # toma el tiempo inicial
np.sum(x)/len(x)               # promedio
end = t.time() - start     # calcula cuando demora el ciclo
print('function elapsed ',end)


# ####  Creando y salvando arreglos
# 
# Podemos crear arreglos de python creando varias funciones built in.  Podemos crear arreglos de numpy tambien por medio de otras listas, que es lo que haremos aquí.  Veamos el ejemplo

# In[ ]:


import numpy as np

x = np.array([10, 20, 30, 40])
print(x, type(x), x.dtype)


# Notamos entonces que podemos ver simplemente los valores de lista, establecer y ver el tipo de dato de numpy y también observamos que .dtype nos da el tipo de datos en el arreglo de numpy que es diferente a el tipo de variable que genera numpy.  Revisar la documentación para ver los diferentes estilos de datos que maneja numpy.

# In[ ]:


x.shape


# Esta funcion built in de numpy nos da el formato del arreglo, en este caso (4,).  Esto significa que es un arreglo de 4 filas, pero, vemos que despues de la coma no existe un 1, que sería lo lógico pues es un arreglo de 4x1 (filas x columnas).  Sucede que intrinsicamente numpy lo arregla de esta manera, es un 1.  En este caso, al arreglo se le conoce como vector.
# 
# Si fueste un arreglo multidimensional tuvieramos esta parte con el número de filas del arreglo.  Veamos el siguiente ejemplo para arreglos multidimensionales.

# In[ ]:


x = np.array([[int(j*i) for i in range(1,5)] for j in range(1,6)])


# In[ ]:


x, x.shape, x.dtype, x.size


# Ahora observamos varias cosas.  El arreglo cuando imprimimos la forma vemos una dimensión más anotada dictaminando las columnas a cuatro, el arreglo es de dos dimensiones 5x4 de tipo int32.
# 
# Tambien observamos que podemos tener la cantidad de elementos en el arreglo por medio del atributo .size.
# 
# A los arreglos de una dimensión se les llama vectores, otra forma de llamarlos es arreglos de rank-1 (rango 1).  Y a estos arreglos de dos dimensiones rank-2.  Existen más dimensiones y mas rangos, pero por el momento lo dejaremos así y veremos más de estos en otra ocasión.

# In[ ]:


x = np.array(['Inteligencia', 'Artificial'])
print(x, x.shape, type(x), x.dtype, x.size)


# Observamos que  entonces no camba el tipo de arreglo, sigue siendo un ndarray, sin embargo el tipo interno de dato es almacenado en un formato llamado Unicode, en este caso es un unicode de 12 caracteres.

# Pregunta!, que sucedería si creamos un arreglo mixto en numpy, vimos anteriormente que las listas (que son arreglos primitivos) no permiten este tipo de iteracción, en el ejemplo veremos que sucede.

# In[ ]:


x = np.array([1,2,3,4, 'Inteligencia', 'Artificial'])
print(x, x.shape, type(x), x.dtype, x.size)


# Bien, ya observó que python y especificamente numpy convierte al tipo de datos que abarque y pueda cumplir con el requerimiento.  Unicode de 12 elementos.  Veamos otro ejemplo de enteros y flotantes.

# In[ ]:


x = np.array([1,2,3.0,4])
print(x, x.shape, type(x), x.dtype, x.size)


# También numpy hace la conversión del arreglo, en esta ocasión tenemos que transforma al tipo de dato por medio de algo llamado casting (upcasting en este caso) de tipo float64.  Con numpy también podemos especificar el tipo de datos para crear el arreglo.  Veamos el ejemplo.

# In[ ]:


x = np.array([1,2,3,4,5,6,8.2],dtype=np.int32)
print(x, x.shape, type(x), x.dtype, x.size)


# Esto sirve para forzar el arreglo para que sea de ese tipo de datos o forzarlos a este tipode datos.

# Podemos entonces salvar el arreglo también y volver a cargarlo a memoria, veamos el ejemplo.

# In[ ]:


x = np.array([1,2,3,4])
np.save('saved_array', x)
del x # borramos la variable, solo como ejemplo


# Si revisa la carpeta (donde se encuentra alojado este notebook), se dará cuenta que existe un archivo *saved_array.npy* que representa el archivo guardado.  Para cargarlo vea el siguiente ejemplo

# In[ ]:


type(x) # al ejecutar nos dará error, porque no existe en memoria, fue borrada anteriormente


# En lineas anteriores vemos que borré la variable x y lo comprobamos por medio de viendo el tipo, ahora vamos a cargar los datos del arreglo en la variable x

# In[ ]:


x = np.load('saved_array.npy')
print(x)


# ### Utilizando funciones built-in para crear ndarrays
# 
# Debajo vemos la manera estándar de creación de arreglos, pero en las siguientes líneas vemos que numpy ofrece crear más tipos de arreglos ademas de estos, son útiles para cualquier tipo de situación y más en la parte científica de python.

# In[ ]:


x = [1,2,3,4,5,6,7,8,9,10]
np.array(x)


# Veamos como crear un arreglo de ceros en numpy.  Simplemente llamamos la función zeros() donde la entrada es un tuple que simboliza el tamaño del arreglo, el cual puede ser unidimensional o multidimensional debido a que numpy ofrece la capacidad de trabajar con n-dimensional arrays.

# In[ ]:


x = np.zeros((5,8))
x


# In[ ]:


x = np.ones((3,4))
y = np.eye(5)
print(x)
print()
print(y)


# Vemos que también podemos crear arreglos de unos y también una matriz identidad, para otros problemas de ciencia de datos este tipo de preparación de arreglos es importante, veamos el siguiente ejemplo de como crear un arreglo con un numero fijo de datos.

# In[ ]:


x = np.full((4,4),2.2)
print(x)


# Excelente, la funcion built-in full() nos permite crear un arreglo con algun tipo de datos insertado.  Veamos como crear una matriz diagnonal con numeros arbitrarios, o mas bien, nuestros propios números.

# In[ ]:


x = np.diag([1, 2, 3, 4])
x


# Muy util, pero que sucede si quiero una manera sencilla de crear vectores, numpy también ofrece una funcion para esto.  Podemos usar en numpy arange(start,step,stop).  Note que la generación de estos números con arange() siempre es entero.

# In[ ]:


x = np.arange(10)
x


# In[ ]:


x = np.arange(2,10)
x


# In[ ]:


x = np.arange(2,20,3)
x


# Para cuando queremos crear otro tipo de datos flotantes no utilizamos esta función sino una diferente llamada linspace() veamos el ejemplo.

# In[ ]:


x = np.linspace(0, 30, 20)
print(x)


# ¿Qué hizo la función?, pues simple, los dos primeros datos son el inicio y el fin del arreglo y el útlimo argumento de la funcion linspace() nos dice cuantos datos existen en el arreglo, la función entonces intentará hacer extrapolaciones de los datos intermedios llenando el arreglo con los datos faltantes.

# existe una posibilidad de hacer con linspace que no nos de (o mas bien trata de hacer fit a los datos que genera, para no tener tanta precision) veamos como.

# In[ ]:


x = np.linspace(0, 30, 20, endpoint=False)
print(x)


# Hasta el momento hemos utilizado las funciones anteriores para arreglos de una dimensión.  Pero nos tocará en casos y en especial en redes neuronales hacer la conversión de datos a un arreglo de diferente dimensión.  Veamos un ejemplo que clarifica este párrafo.

# In[ ]:


x = np.reshape(x, (5,4))
print(x)


# Si observamos anteriormente, el arreglo es de 20x1 pero ahora lo hemos convertidos a un arreglo de 5x4.  ¿Qué sucedería si aplicamos lo mismo pero para un arreglo de más variables de las que posee?

# In[ ]:


x = np.reshape(x, (8,10))
print(x)


# Lógico, estamos pidiendo un arreglo de 80 elementos pero el arreglo solamente tiene 20, como puede ver en la línea 
# 
# *ValueError:  cannot reshape array of size 20 into shape (8,10)*

# Algunas funciones de numpy nos permiten operar como métodos con notación de punto.  Veamos como podemos hacer el mismo resultado con una sola línea de código.

# In[ ]:


x = np.arange(30).reshape((6,5))
print(x)


# In[ ]:


x = np.linspace(1, 30, 20, endpoint=False).reshape((10,2))
print(x)


# Vimos entonces que podemos usar las funciones de numpy para hacer lo mismo en una sola línea, esto acelera la codificación y nos ayuda a entender lo que estamos ahciendo.
# 
# Por ejemplo analizando la última línea de funcion vemos que:
# - Creamos un arreglo de inici en 1 fin en 30 con 20 elementos y comprimido en información
# - Luego lo formateamos y lo volvemos un arreglo multidimensional de 10x2

# Para redes neuronales nos interesa crear numeros o pesos aleatorios, numpy nos permite crear numeros aleatorios de numeros entre 0-1 de la siguiente manera.

# In[ ]:


x = np.random.random((3,9))
print(x)


# Vimos anterioremnte que podemos crear numeros aleatorios de punto fijo, pero también numpy nos permite crear numeros aleatorios de forma de una matriz multidimensional, veamos el ejemplo.

# In[ ]:


x = np.random.randint(2, 20, (4,3))
print(x)


# Para estadistica nos interesa también ver datos de una distribución normal.  Observamos como numpy nos ofrece de manera fácil crear estos datos.  En las siguientes líneas crearemos una distribución normal de media 0 y desviación estándar de 0.2 dimensiones de 5x4

# In[ ]:


x = np.random.normal(0, 0.2, size=(5, 4))
print(x)


# In[ ]:


mn = np.mean(x)
st = np.std(x)
ma = np.max(x)
mi = np.min(x)
su = np.sum(x)
print(f'Media: {mn}\nDesv Est: {st}\nMáximo: {ma}\nMínimo: {mi}\nTotal: {su}')


# #### Accediendo. borrando e insertando elementos en los arreglos
# 
# No nos hemos dado cuenta pero una propiedad de los arreglos de numpy es que son mutables, es decir podemos cambiar su contenido en cualquier momento y sin ningun inconveniente.
# 
# También podemos hacer slicing de los arreglos para separar los datos.  En redes neuronales esto es util cuando hacemos el train y test set ademas de la validación cruzada para validar el training.
# 
# También podemos verlos elemento por elemento o por conjunto de ellos.  Veamos el ejemplo.

# In[ ]:


x = np.array([1,2,3,4,5,6])
print('1st:',x[0])
print('3rd:',x[2])
print('last',x[-1])


# Indices positivos nos ayudan a acceder los elementos desde el inicio del arreglo y los elementos desde el final al inicio con índices negativos, note que el indice positivo comienza desde 0.  Veamos ahora ejemplos de arreglos multidimensionales.

# In[ ]:


x = np.array(np.arange(20).reshape(10,2))
print(x)
print()
print('Elemento en 0,0:', x[0,0])
print('Elemento en 4,1:', x[4,1])
print('Elemento en 9,1:', x[9,1])


# Para modificar elementos en un arreglo simplemente hacemos una asignación al valor del elemento a reemplazar.

# In[ ]:


x[9,1] = -2
print(x)


# Para borrar elementos simplemente podemos utilizar la función delete().  Cabe aclarar aqui que podemos decidir que borrar, borrar por filas o columnas.  Normalmente verá en algunas funciones, no solo en numpy el argumento *axis* 
# - axis=0 equivale a fila
# - axis=1 equivale a columna
# 
# Veamos un ejemplo de la posibilidad de borrar una sección del arreglo

# In[ ]:


y = np.delete(x, 0, axis=0)
print(x)
print()
print(y)


# In[ ]:


y = np.delete(x, 0, axis=1)
print(x)
print()
print(y)


# Podemos también agregar uno o mas elementos a un arreglo, veamos como podemos hacer esto con un arreglo unidimensional

# In[ ]:


x = np.arange(10)
x


# In[ ]:


x = np.append(x, -999)
x


# Digamos ahora que queremos agregar más información que un solo dato.  También podemos hacerlo, observar el ejemplo siguiente.

# In[ ]:


x = np.array(np.arange(10))
x = np.append(x, [20, 30])
print(x)


# Para arreglos multidimensionales opera igualmente y también podemos agregar datos por filas y por columnas, simplemente no la hemos utilizado en el ejemplo anterior debido a que siempre hay una columna para un vector.

# In[ ]:


x = np.array(np.linspace(2, 10, 20)).reshape((5,4))
y = np.append(x,[[1,2,3,4]],axis=0)
print(x,'\n\n',y)


# In[ ]:


x = np.array(np.linspace(2, 10, 20)).reshape((5,4))
y = np.append(x,[[1],[2],[3],[4],[5]],axis=1)
print(x,'\n\n',y)


# Note que para que podamos insertar en un arreglo tenemos que tener el mismo valor de filas y de columnas, dependiendo de que vamos a insertar, de lo contrario tendremos un error como el siguiente

# In[ ]:


x = np.array([[1,2,4],[4,5,6]])
x = np.append(x, [7,8,9,10], axis=0)


# Veamos ahora como insertar elementos dentro de un arreglo.  Podemos insertar elementos entre el arreglo utilizando la funcion insert()

# In[ ]:


x = np.array([2,4,8,16])
x


# In[ ]:


y = np.insert(x, 1, [5,6,7,8], axis=0)
print(x)
print()
print(y)


# In[ ]:


x = np.reshape(x, (2,2))
y = np.insert(x, 1, [32, 64], axis=1)
print(x)
print()
print(y)


# Para arreglos multidimensionales, las inserciones se hacemn por medio de vstack() o hstack(). Cabe recalcar que necesitamos que las partes concuerden en dimensiones.  Veamos un ejemplo.

# Veamos primero stacking vertical

# In[ ]:


x = np.arange(10).reshape((5,2))
print(x)


# In[ ]:


y = np.arange(4).reshape((2,2))
print(y)


# In[ ]:


z = np.vstack((x,y))
print(z)


# Ahora veamos stacking horizontal

# In[ ]:


x = np.arange(10).reshape((5,2))
print(x)


# In[ ]:


y = np.arange(5).reshape((5,-1))
print(y)


# In[ ]:


z = np.hstack((x,y))
print(z)


# #### Division por partes de numpy arrays
# 
# Slicing funciona igual que como hemos visto en las listas.  Podemos indexar de las siguientes maneras
# - [inicio:fin]
# - [inicio:]
# - [:fin]
# 
# Si observó modemos hacer slicing sin especificar algunos indices, fin o inicio respectivamente.
# 
# También podemos hacer sub slicing de arreglos multidimensionales, veamos los siguientes ejemplos.

# In[ ]:


x = np.arange(1,21).reshape((5,4))
print(x)


# In[ ]:


y = x[1:3,1:4]
print(y)


# OK, es un poco confuso a veces verlo.  Para entenderlo mejor tenemos que hacer simple uso de esta manera, el primer indice siempre esta incluido y el último índice esta excluido, por consiguiente:
# - 1:3 (incluye el primer índice y excluye el último, es decir solo realizamos la selección de la fila 1 hasta fila 2 (ó 3-1).
# - 1:4 (incluye el primer índice y excluye el último, es decir solo realizamos la selección de la col 1 hasta col 3 (ó 4-1). 
# 
# Veamos más ejemplos.

# In[ ]:


print(x)
print()
print(x[1: , :3])


# El siguiente solamente toma todos los valores de fila pero solo la columna 2

# In[ ]:


y = x[:,2]
print(y, y.shape)


# Ahora todos los valores de la columna pero solo la tercera fila

# In[ ]:


y = x[3, :]
print(y, y.shape)


# ¿Nota que solamente esta retornanod vectores rango 1 tanto para filas como para columnas? 
# 
# Si queremos seleccionar efectivamente que retorne en modo que deseamos, entonces debemos ser un poco más específicos

# In[ ]:


y = x[3:4, :]
print(y, y.shape)


# In[ ]:


y = x[:,2:3]
print(y, y.shape)


# Un error común de principiantes es cuando se manipulan los arreglos pensar que son una copia del arreglo original, esto incurre en errores de codigo a futuro.
# 
# Veamos un ejemplo de lo que acabamos de mencionar.  Vamos a crear un arreglo, copiar un pedazo a una variable y cambiar el valor para luego imprimir ambas y ver su resultado.

# In[ ]:


x = np.linspace(0,10,20).reshape((5,4))
print(x)


# In[ ]:


y = x[1:, 2:]
print(y)


# In[ ]:


y[-1, -1] = -1
y


# In[ ]:


x


# Si deseamos crear una copia que no interfiera con los elementos del arreglo anterior entonces debemos utilizar la función copy().  Veamos el ejemplo

# In[ ]:


x = np.array(np.arange(20)).reshape(5,4)
print(x)


# In[ ]:


y = x[1:, 1:].copy()
y[-1,-1] = -2
y


# In[ ]:


x


# Numpy también nos ofrece manera de tener acceso a otros elementos por medio de funciones, por ejemplo digamos que queremos la diagonal, tenemos que.

# In[ ]:


np.diag(x)


# Si queremos los elementos en diagonal sobre ella tenemos que especificar k = 1, por defecto k = 0

# In[ ]:


np.diag(x, k=1)


# Ahora si queremos los elementos en diagonal debajo de ella k = -1

# In[ ]:


np.diag(x, k=-1)


# Digamos también que queremos solamente los valores únicos de un arreglo, numpy nos ofrece una función llamada unique() que hace el trabajo.

# In[ ]:


x = np.array([[1,2,3],[4,5,6],[7,8,9],[1,3,9]])
x


# In[ ]:


print(np.unique(x))


# #### Indexamiento booleano
# 
# Hemos visto como hacer indexamiento, pero hay muchas situaciones donde no sabemos los elementos, por ejemplo cuando tenemos un arreglo inmenso de datos, digamos 1M, y de esos datos solamente queremos seleccionar aquellos que cumplan alguna condicion.
# 
# Podemos utilizar argumentos booleanos en vez de indices.

# In[ ]:


x = np.arange(30).reshape((6,5))
x


# In[ ]:


x[x>14]


# In[ ]:


x[x<=5]


# In[ ]:


print(x[(x<10) & (x>4)])


# Tambien nos permite hacer operaciones, por ejemplo reasignación de un rango de valores

# In[ ]:


x[x>24] = -4
print(x)


# Tambien podemos ver arreglos para observar su intersección, diferencia y unión.

# In[ ]:


x = np.arange(1,6)
y = np.array([6,7,8,2,4])

print(np.intersect1d(x,y))
print(np.setdiff1d(x,y))
print(np.union1d(x,y))


# Ahora veamos diferentes maneras de orderar arreglos de 1-D o N-D.  Vamos a ver la marea de ordenar un arreglo sin reemplazar el arreglo original.

# In[ ]:


x = np.random.randint(2, 15, size=(5,))
x


# In[ ]:


print(np.sort(x))
print(x)


# Ahora veamos la manera de alterar y solamente conseguir los números únicos

# In[ ]:


print(np.sort(np.unique(x)))
print(x)


# Si ahora queremos alterar el areglo original y que este quede modificado debemos hacerlo sobre el arreglo, es decir, como vemos en el ejemplo.

# In[ ]:


x = np.random.randint(2,10, size=(1,10))
print(x)
x.sort()
print(x)


# La función sort() también la podemos utilizar para ordenar arreglos multidimensionales, vemos el ejemplo inferior para ordenar arreglos.

# In[ ]:


x = np.random.randint(0,20, size=(5,4))
print(x)


# In[ ]:


print(np.sort(x, axis=0))


# In[ ]:


print(np.sort(x, axis=1))


# ###  Operadores aritméticos y broadcasting
# 
# Veamos como podemos hacer operaciones aritméticas por arreglos en elementos (element-wise).
# 
# Podemos hacerlas por medio de operadores o funciones, veamos como podemos hacer sumas, restas, multiplicacioens y divisiones

# In[ ]:


x = np.random.randint(1, 10, size=(10,))
y = np.random.randint(49, 60, size=(10,))

print(x)
print(y)


# In[ ]:


print(x+y)
print(np.add(x,y))


# In[ ]:


print(x-y)
print(np.subtract(x,y))


# In[ ]:


print(x*y)
print(np.multiply(x,y))


# In[ ]:


print(x/y)
print(np.divide(x,y))


# Para que lo anterior funcione los arreglos, como lo hacemos elemento por elemento deben tener la misma dimensión o ser 'broadcastables' o retransmisibles.
# 
# Podemos tambien aplicar las mismas operaciones a arreglos multidimensionales.

# In[ ]:


x = np.random.randint(1, 10, size=(3,4))
y = np.random.randint(3, 20, size=(3,4))


# In[ ]:


print(x + y)


# In[ ]:


print(x - y)


# In[ ]:


print(x*y)


# In[ ]:


print(x/y)


# Otra ventaja de numpy son sus implementaciones en funciones matemáticas y estadisticas, veamos los ejemplos

# In[ ]:


# tenemos el arreglo original x
x


# In[ ]:


print(np.exp(x))


# In[ ]:


print(np.sqrt(x))


# In[ ]:


print(np.power(x,2))


# In[ ]:


print('Suma:', x.sum())
print('Suma-rows:', x.sum(axis=0))
print('Suma-cols:', x.sum(axis=1))
print('Media:', x.mean())
print('Media-rows:',x.mean(axis=0))
print('Media-cols:',x.mean(axis=1))


# In[ ]:


print('Stdev:', x.std())
print('Max:', x.max())
print('Min:', x.min())


# Ahora veamos como numpy puede agregar valores de constantes sin utilizar ciclos for a arreglos unidimensionales y multidimensionales.

# In[ ]:


print(x)


# In[ ]:


print(x + 2.2)


# In[ ]:


print(1.2-x)


# In[ ]:


print(0.5*x)


# In[ ]:


print(x/4)


# Ahora veamos las funciones de broadcasting o retransimsibles de numpy.  Vamos a realizar dos arreglos multidimensionales pero uno incompleto y realizar operaciones

# In[ ]:


x = np.random.randint(0,20, size=(5,3))
y = np.random.randint(5,15, size=(1,3))
print(x)
print()
print(y)


# In[ ]:


print(x+y)


# Observamos que la función de retransmisibilidad o broadcasting trata de sumar el arreglo si tiene las mismas dimensiones al menos en una dirección, para el caso anterior cada fila del arreglo x fue sumada con los elementos del arreglo y

# #### Intentelo ud. ahora
# 
# Crear un arreglo con las siguientes especificaciones
# - 5x2x3
# - numeros consecutivos de 2 a 60 (incluyendo el número 60)
# - pasos de 2

# In[6]:


import numpy as np

arreglo_1 = np.arange(2, 61, 2).reshape(5, 2, 3)

print(arreglo_1)


# Respuesta esperada:
# ```
# [[[ 2  4  6]
#   [ 8 10 12]]
# 
#  [[14 16 18]
#   [20 22 24]]
# 
#  [[26 28 30]
#   [32 34 36]]
# 
#  [[38 40 42]
#   [44 46 48]]
# 
#  [[50 52 54]
#   [56 58 60]]]
# ```

# - Crear un arreglo 5x6
# - El arreglo debe contener enteros consecutivos desde 1 hasta 30 (incluyendo el 30)
# - Luego seleccione solo los numeros impares en el arreglo
# - NOTA:  No se acepta la creación del arreglo como *np.array([1,2,3...30])*

# In[5]:


import numpy as np

arreglo_numeros = np.arange(1, 31).reshape(5, 6)
arreglo_impares = arreglo_numeros[arreglo_numeros% 2 != 0]

print(arreglo_numeros)


# Respuesta esperada:
# ```
# array([[ 1,  2,  3,  4,  5,  6],
#        [ 7,  8,  9, 10, 11, 12],
#        [13, 14, 15, 16, 17, 18],
#        [19, 20, 21, 22, 23, 24],
#        [25, 26, 27, 28, 29, 30]])
# ```

# In[4]:


import numpy as np

arreglo_numeros = np.arange(1, 31).reshape(5, 6)
arreglo_impares = arreglo_numeros[arreglo_numeros % 2 != 0]

print(arreglo_impares)


# Respuesta esperada:
# ```
# [ 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29]
# ```

# Utilice broadcasting en numpy para crear un arreglo como se especifica:
# - Arreglo 4x8
# - col1 = llena de 1s, col2 = llena de 2s ... col8 = llena de 8s

# In[3]:


import numpy as np

columnas = np.arange(1, 9)
arreglo = np.ones((4, 1)) * columnas

print(columnas)


# Nota de ayuda:
# ```
# [[0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0.]]
# 
# [1 2 3 4 5 6 7 8]
# ```

# In[2]:


import numpy as np

columnas = np.arange(1, 9)
arreglo = np.ones((4, 1)) * columnas

print(arreglo)


# Respuesta esperada:
# ```
# array([[1., 2., 3., 4., 5., 6., 7., 8.],
#        [1., 2., 3., 4., 5., 6., 7., 8.],
#        [1., 2., 3., 4., 5., 6., 7., 8.],
#        [1., 2., 3., 4., 5., 6., 7., 8.]])
# ```
