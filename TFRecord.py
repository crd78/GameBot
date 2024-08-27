import tensorflow as tf
import json
import os
import glob

def json_to_tfrecord(json_file, label_map):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Construire le chemin complet du fichier image
    image_path = os.path.join('images', data['imagePath'])
    
    # Vérifier si le fichier image existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found.")
    
    image = tf.io.read_file(image_path)
    image_shape = tf.image.decode_jpeg(image).shape
    
    width = image_shape[1]
    height = image_shape[0]
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        
        xmins.append(points[0][0] / width)
        xmaxs.append(points[1][0] / width)
        ymins.append(points[0][1] / height)
        ymaxs.append(points[1][1] / height)
        classes_text.append(label.encode('utf8'))
        classes.append(label_map[label])
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['imagePath'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['imagePath'].encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy()])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpeg'])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    
    return tf_example

def create_tfrecord(output_path, json_files, label_map):
    with tf.io.TFRecordWriter(output_path) as writer:
        for json_file in json_files:
            try:
                tf_example = json_to_tfrecord(json_file, label_map)
                writer.write(tf_example.SerializeToString())
            except FileNotFoundError as e:
                print(e)

label_map = {
    'minerais_t4': 1,
    'herbe_t3': 1,
    'minerais_t3': 1,
    'minerais_t2': 1,
    'herbe_t2': 1,
    'herbe_t3': 1,
    'herbe_t4': 1,
    'PIERRE': 1,
    'ENNEMIE': 2,


    # Ajoutez d'autres étiquettes ici
}

json_files = glob.glob('annotations/*.json')
create_tfrecord('output.tfrecord', json_files, label_map)