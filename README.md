## Práctica 5. Detección y caracterización de caras

### Descripción

El objetivo de esta práctica es realizar un filtro similar a los que tienen algunas aplicaciones como _Instagram_. En nuestro caso se ha realizado dos filtros diferentes que posteriormente hemos puesto en común. Dichos filtros se describirán a continuación.

### Tareas del proyecto

### Vómito arcoiris

Como primer filtro se implemento que, cuando la persona en cámara abra la boca, se dibuje un GIF sobre la boca, simulando de esta forma que está "vomitando arcoiris". Algunos aspectos a comentar del código:

- Se utiliza la librería mediapipe para la detección de la cara.
- Se utiliza la librería PIL para la carga del GIF y posteriormente se convierte en una lista de fotogramas para su posterior dibujo.
- Se declara la función mouth_open, la cual detecta el labio superior e inferior mediante landmarks y, si la distancia supera cierto umbral (por defecto 0.05), se dibuja el GIF anteriormente mencionado
```py
def mouth_open(landmarks, threshold=0.05):
    """Calcula si la boca está abierta basado en la distancia entre puntos específicos."""
    top_lip = landmarks[13]  # Landmark del labio superior
    bottom_lip = landmarks[14]  # Landmark del labio inferior
    mouth_height = abs(bottom_lip.y - top_lip.y)
    return mouth_height > threshold
```

### Brillo al guiñar un ojo

El segundo filtro realizado consiste en que, al guiñar un ojo, se superponga en la imagen una imagen de unas estrellas/brillos cerca del ojo guiñado. Algunos aspectos a comentar:

- Para la detección del guiño, se calcula la relación entre las distancias verticales y horizontales de los párpados y, en caso de que un ojo esté cerrado y el otro abierto, se interpreta como un guiño y se dibuja la imagen.
```py
def eye_aspect_ratio(eye_landmarks):
    vertical_dist = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) - 
                                   np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
    horizontal_dist = np.linalg.norm(np.array([eye_landmarks[0].x, eye_landmarks[0].y]) - 
                                     np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
    return vertical_dist / horizontal_dist
```
- La estrella/brillo se coloca cerca del ojo que realiza el guiño. Si es el ojo derecho, la imagen se muestra sin reflejar y, si es el izquierdo, se muestra reflejada
```py
def overlay_star(img, position, scale, mirror=False):
    star_resized = star_img.resize((scale, scale), Image.LANCZOS)
    star_np = np.array(star_resized)

    star_bgr = cv2.cvtColor(star_np, cv2.COLOR_RGBA2BGR)
    alpha = star_np[..., 3] / 255.0  # Canal alfa normalizado

    # Si se debe reflejar la estrella en el eje horizontal, reflejar también el canal alfa
    if mirror:
        star_bgr = cv2.flip(star_bgr, 1)
        alpha = np.flip(alpha, axis=1)  # Reflejar el canal alfa también

    # Posición en la imagen de destino
    x, y = int(position[0] - scale // 2), int(position[1] - scale // 2)

    h, w = img.shape[:2]
    star_h, star_w = star_bgr.shape[:2]
    if x < 0 or x + star_w > w or y < 0 or y + star_h > h:
        return

    # Superponer la estrella sobre la imagen, utilizando el canal alfa para la mezcla
    for c in range(3):  # Canales de color BGR
        img[y:y+star_h, x:x+star_w, c] = (
            star_bgr[:, :, c] * alpha + 
            img[y:y+star_h, x:x+star_w, c] * (1 - alpha)
        )
```

### Resultados

### Requisitos

### [Anaconda Prompt](https://www.anaconda.com/)
```
conda create --name VC_P5 python=3.9.5
conda activate VC_P5
```
### OpenCV
```
pip install opencv-python
```
### NumPy
```
pip install numpy
```
### Mediapipe
```
pip install mediapipe
```

## Autoría
[Sara Expósito Suárez](https://github.com/SaraE5)

[Alejandro Padrón Ossorio](https://github.com/apadoss)