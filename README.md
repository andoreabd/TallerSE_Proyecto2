# TallerSE_Proyecto2
## Dependencias
Se utiliza Python, OpenCV, TensorFlow
## Entrenamiento de modelo de reconocimiento de emociones
Se necesitan imágenes para entrenar el modelo de FER-2013 y correr el script llamado *Emotions.py* con:
```
python emotions.py --mode train
```
Se genera un file llamado *model.h5* que se puede probar con la cámara de la computadora.
## Prueba del modelo
Para probar que funcione en la computadora hay que correr:
```
python emotions.py --mode display
```
En la carpeta que se corre el comando debe estar el documento *haarcascade_frontalface_default.xml*
## Recetas

### Referencia:
[https://github.com/atulapra/Emotion-detection]
[https://www.kaggle.com/datasets/msambare/fer2013]
[https://www.kaggle.com/code/sankalpsrivastava26/face-emotion-recognition-using-tensorflow]
[https://medium.com/analytics-vidhya/realtime-face-emotion-recognition-using-transfer-learning-in-tensorflow-3add4f4f3ff3]
[https://blog.devgenius.io/facial-expression-recognition-with-tensorflow-90f6174163c3]
