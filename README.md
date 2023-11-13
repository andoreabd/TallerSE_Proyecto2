# TallerSE_Proyecto2
## Modelo de reconocimiento de emociones
### Dependencias
Se utiliza Python, OpenCV, TensorFlow
### Entrenamiento de modelo de reconocimiento de emociones
Se necesitan imágenes para entrenar el modelo de FER-2013 y correr el script llamado *Emotions.py* con:
```
python emotions.py --mode train
```
Se genera un file llamado *model.h5* que se puede probar con la cámara de la computadora.
### Prueba del modelo
Para probar que funcione en la computadora hay que correr:
```
python emotions.py --mode display
```
En la carpeta que se corre el comando debe estar el documento *haarcascade_frontalface_default.xml*
## Configuración de la imagen
## Recetas
### bblayer
```
BBLAYERS ?= " \
  /home/gh/poky/meta \
  /home/gh/poky/meta-poky \
  /home/gh/poky/meta-yocto-bsp \
  /home/gh/poky/meta-openembedded/meta-oe \
  /home/gh/poky/meta-openembedded/meta-python \
  /home/gh/poky/meta-openembedded/meta-xfce \
  /home/gh/poky/meta-openembedded/meta-gnome \
  /home/gh/poky/meta-openembedded/meta-multimedia \
  /home/gh/poky/meta-openembedded/meta-networking \
  /home/gh/poky/meta-raspberrypi \
  /home/gh/poky/meta-tensorflow-lite \
```
### local.conf
```
IMAGE_INSTALL:append = " \
                sudo \
                python3 \
                python3-pip \
                xf86-video-fbdev \
                vim \
                git \
                packagegroup-core-x11 \
                packagegroup-xfce-base \
                python3-tensorflow-lite \
```
### Referencia:
[https://github.com/atulapra/Emotion-detection]
[https://www.kaggle.com/datasets/msambare/fer2013]
[https://www.kaggle.com/code/sankalpsrivastava26/face-emotion-recognition-using-tensorflow]
[https://medium.com/analytics-vidhya/realtime-face-emotion-recognition-using-transfer-learning-in-tensorflow-3add4f4f3ff3]
[https://blog.devgenius.io/facial-expression-recognition-with-tensorflow-90f6174163c3]
