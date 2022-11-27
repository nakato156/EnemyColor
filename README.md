# Enemy Color
Enemy Color es un algoritmo de regresión lineal diseñado para poder encontrar un contraste de un color (o su inverso).

##Uso
Despues de descargar el proyecto y añadirlo al suyo debe importarlo de esta forma
```python
from enemy_color import EnemyColor
from numpy import array

color = EnemyColor()
color.predict(array([150, 130, 0]))
```

La clase `EnemyColor` puede recibir un parámetro `default_load` de valor booleano que especificará si se usa un modelo ya entrenado o se preparará uno para el entrenamiento. SIempre que se quiera reentrenar al modelo, ya sea con los datos existentes o propios datos se deberá especificar `default_load=False`.

En caso desee entrenar el modelo desde 0 debe hacer lo siguiente:
```python
from enemy_color import EnemyColor
from numpy import array

color = EnemyColor(default_load=False)
#cargamos los datos del csv por default
color.load_data_csv() # si no se especifica ruta del csv se tomará el csv propio del modelo
#entrenamos el modelo
color.train(save=True)  #save indica si el modelo será guardado
color.predict(array([100, 130, 0])) #predecimos
```
En caso quiera usar un set de datos propios deberá especificar la ruta, para ello se pasa el parámetro `path_csv` que especifica la ruta del csv de colores que se utilizará para el entrenamiento del modelo. Asímismo, se puede especificar en el método `train` el paraémtro `score` que indica el score mínimo que debe alcanzar el modelo en su entrenameinto (por defecto es `.25`). 

```python
from enemy_color import EnemyColor
from numpy import array

color = EnemyColor(default_load=False)
color.load_data_csv("ruta_de_tu_archio.csv") #cargamos los datos desde la ruta proporcionada
#entrenamos el modelo
color.train(save=True)  #save indica si el modelo será guardado
color.predict(array([100, 130, 0])) #predecimos
```
El archivo `.csv` deberá tener 2 columnas una llamada `colors` y la otra `contrast`. En la columna `colors` se deberá especificar un color cualquiera en formato hexadecimal (sin el caracter `#`) y en la otra columna el contraste o inverso del color (igualmente en formato hexadecimal y sin el caracter `#`). Sin embargo puede proorcionar el color en un formato rgb y especificar como parámetro `is_hex` de la función `load_data_csv` donde se omitirá la conversión a RGB.

Por defecto se le abrirá una ventana donde podrá ver el color resultante de la predicción. SI no desea verlo especifique el parámetro `preview=False` en el método `predict`.