import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

# Caminho do dataset
DATASET_DIR = "/home/user/species-classification/dataset"  # Substitua pelo caminho correto do dataset
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Configurações
IMAGE_SIZE = (128, 128)  # Tamanho das imagens
BATCH_SIZE = 32
NUM_CLASSES = 15
LEARNING_RATE = 0.001
EPOCHS = 500
CSV_OUTPUT_DIR = "./csv_10"

# Criar pasta de saída para CSVs
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

# Função para criar o modelo
def create_model(input_shape=(128, 128, 3), num_classes=15):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Preparar os geradores de dados
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Criar o modelo
model = create_model(input_shape=(128, 128, 3), num_classes=NUM_CLASSES)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

# Salvar o modelo treinado
model.save("deep_model_15_classes.h5")

# Função para gerar os gráficos de treinamento
def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 6))

    # Gráfico de acurácia
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Gráfico de perda
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

plot_training(history)

# Função para gerar previsões e salvar CSV
def generate_predictions_csv(generator, model, output_csv, dataset_name):
    predictions = model.predict(generator, verbose=1)
    y_true = generator.classes
    class_labels = list(generator.class_indices.keys())

    # Identificadores das imagens
    image_ids = [os.path.basename(filepath) for filepath in generator.filepaths]

    # DataFrame para armazenar os dados
    df = pd.DataFrame(predictions, columns=class_labels)
    df.insert(0, 'real_class', [class_labels[i] for i in y_true])
    df.insert(0, 'image_id', image_ids)

    # Salvar o DataFrame no CSV
    csv_path = os.path.join(CSV_OUTPUT_DIR, output_csv)
    df.to_csv(csv_path, index=False)
    print(f"{dataset_name} CSV salvo em {csv_path}")

# Gerar CSVs para os conjuntos de dados
generate_predictions_csv(test_generator, model, "train_predictions.csv", "Train")
generate_predictions_csv(val_generator, model, "val_predictions.csv", "Validation")
generate_predictions_csv(test_generator, model, "test_predictions.csv", "Test")

# Função para plotar matriz de correlação
def plot_correlation_matrix(generator, model):
    predictions = model.predict(generator)
    correlation_matrix = np.corrcoef(predictions, rowvar=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm',
                xticklabels=list(generator.class_indices.keys()),
                yticklabels=list(generator.class_indices.keys()))
    plt.title('Correlation Matrix of Predicted Probabilities')
    plt.show()

# Plotar a matriz de correlação para o conjunto de teste
plot_correlation_matrix(test_generator, model)

