import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Configuración
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Cargar datos
train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=30, zoom_range=0.2, horizontal_flip=True)
train_data = train_datagen.flow_from_directory('data/train', target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1.0/255)
val_data = val_datagen.flow_from_directory('data/val', target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# Modelo base
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Agregar capas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Descongelar las últimas capas de la base
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompliar con un optimizador más bajo para evitar grandes actualizaciones de pesos
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Continuar entrenando
model.fit(train_data, validation_data=val_data, epochs=5)

# Guardar el modelo
model.save('bird_classifier_complete.h5')

# Imprime un resumen para verificar las capas
model.summary()