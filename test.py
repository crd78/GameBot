import tensorflow as tf
import tensorflow_hub as hub

# Charger le modèle pré-entraîné depuis TensorFlow Hub
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Préparation d'une image pour la détection
def preprocess_image(image):
    image = tf.image.resize(image, (320, 320))
    image = tf.cast(image, tf.uint8)  # Convertir en uint8
    return tf.expand_dims(image, axis=0)  # Ajouter une dimension batch

# Exemple d'image de test (à remplacer par votre propre image)
image = tf.zeros([320, 320, 3], dtype=tf.uint8)  # Image noire de test
processed_image = preprocess_image(image)

# Effectuer une prédiction
output = model(processed_image)

# Extraire et afficher les résultats
detection_boxes = output['detection_boxes'].numpy()
detection_classes = output['detection_classes'].numpy()
detection_scores = output['detection_scores'].numpy()

print("Detection Boxes:", detection_boxes)
print("Detection Classes:", detection_classes)
print("Detection Scores:", detection_scores)