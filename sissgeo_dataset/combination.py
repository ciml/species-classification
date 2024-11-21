import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Configurações iniciais
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 15
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Dataset personalizado
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = plt.imread(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformações para imagens
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carregar dados (exemplo, adapte para seu caso)
# image_paths e labels vêm do seu dataset
df = pd.read_csv("dataset.csv")
image_paths = df['image_path'].values
labels = df['class'].values
coordinates = df[['latitude', 'longitude']].values

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
train_dataset = CustomImageDataset(X_train, y_train, transform=transform)
test_dataset = CustomImageDataset(X_test, y_test, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modelo ResNet-50
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# Treinamento da ResNet-50
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_resnet():
    model.train()
    for epoch in range(EPOCHS):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate_resnet(loader):
    model.eval()
    predictions, ground_truths = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())
    return ground_truths, predictions

train_resnet()
y_train_true, y_train_pred = evaluate_resnet(train_loader)
y_test_true, y_test_pred = evaluate_resnet(test_loader)

# Métricas
print("Acurácia no treino:", accuracy_score(y_train_true, y_train_pred))
print("Acurácia no teste:", accuracy_score(y_test_true, y_test_pred))
print("Matriz de confusão no teste:")
print(confusion_matrix(y_test_true, y_test_pred))

# MaxEnt
logreg = LogisticRegression(max_iter=1000)
logreg.fit(coordinates, labels)

# Previsão MaxEnt
def predict_maxent(coordinates):
    return logreg.predict_proba(coordinates)

# Combine tabelas
resnet_probs = model(torch.tensor(image_paths)).cpu().detach().numpy()
maxent_probs = predict_maxent(coordinates)

tables = pd.DataFrame({
    'Record': range(len(labels)),
    'True_Class': labels,
    **{f'ResNet_Class_{i+1}': resnet_probs[:, i] for i in range(NUM_CLASSES)},
    **{f'MaxEnt_Class_{i+1}': maxent_probs[:, i] for i in range(NUM_CLASSES)},
})

# Algoritmo genético (fusão dos modelos)
# Aqui você vai criar duas versões: 
# 1. Um único peso para todos os modelos
# 2. Pesos específicos para cada classe.

from deap import base, creator, tools, algorithms

# Configurações do DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", lambda: np.random.uniform(0, 1))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)  # Um peso único
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Avaliação de indivíduos
def evaluate_single_weight(individual):
    weight = individual[0]
    combined_probs = weight * resnet_probs + (1 - weight) * maxent_probs
    combined_preds = np.argmax(combined_probs, axis=1)
    acc = accuracy_score(labels, combined_preds)
    return acc,

toolbox.register("evaluate", evaluate_single_weight)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Algoritmo genético para peso único
def genetic_algorithm_single_weight():
    population = toolbox.population(n=50)
    ngen = 40
    cxpb = 0.5
    mutpb = 0.2

    population, log = algorithms.eaSimple(
        population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=True
    )

    best_individual = tools.selBest(population, k=1)[0]
    return best_individual[0]

# Rodar o algoritmo genético para peso único
best_weight = genetic_algorithm_single_weight()
print(f"Melhor peso único encontrado: {best_weight}")

# Configuração para pesos por classe
toolbox.unregister("individual")
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NUM_CLASSES)

def evaluate_class_weights(individual):
    weights = np.array(individual)
    combined_probs = weights * resnet_probs + (1 - weights) * maxent_probs
    combined_preds = np.argmax(combined_probs, axis=1)
    acc = accuracy_score(labels, combined_preds)
    return acc,

toolbox.register("evaluate", evaluate_class_weights)

# Algoritmo genético para pesos por classe
def genetic_algorithm_class_weights():
    population = toolbox.population(n=50)
    ngen = 40
    cxpb = 0.5
    mutpb = 0.2

    population, log = algorithms.eaSimple(
        population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=True
    )

    best_individual = tools.selBest(population, k=1)[0]
    return np.array(best_individual)

# Rodar o algoritmo genético para pesos por classe
best_class_weights = genetic_algorithm_class_weights()
print(f"Melhores pesos por classe encontrados: {best_class_weights}")

# Combinação dos modelos com os pesos encontrados
final_probs_single_weight = best_weight * resnet_probs + (1 - best_weight) * maxent_probs
final_probs_class_weights = best_class_weights * resnet_probs + (1 - best_class_weights) * maxent_probs

# Geração da tabela final de comparação
final_table = pd.DataFrame({
    "Record": range(len(labels)),
    "True_Class": labels,
    "Single_Weight_Pred": np.argmax(final_probs_single_weight, axis=1),
    "Class_Weights_Pred": np.argmax(final_probs_class_weights, axis=1),
    **{f"ResNet_Class_{i+1}": resnet_probs[:, i] for i in range(NUM_CLASSES)},
    **{f"MaxEnt_Class_{i+1}": maxent_probs[:, i] for i in range(NUM_CLASSES)},
    **{f"Combined_Single_Class_{i+1}": final_probs_single_weight[:, i] for i in range(NUM_CLASSES)},
    **{f"Combined_Class_Weights_{i+1}": final_probs_class_weights[:, i] for i in range(NUM_CLASSES)},
})

# Visualização e métricas finais
print("Matriz de confusão (peso único):")
print(confusion_matrix(labels, np.argmax(final_probs_single_weight, axis=1)))

print("Matriz de confusão (pesos por classe):")
print(confusion_matrix(labels, np.argmax(final_probs_class_weights, axis=1)))

print("Acurácia (peso único):", accuracy_score(labels, np.argmax(final_probs_single_weight, axis=1)))
print("Acurácia (pesos por classe):", accuracy_score(labels, np.argmax(final_probs_class_weights, axis=1)))

# Gráficos comparativos
methods = ["ResNet", "MaxEnt", "Single Weight", "Class Weights"]
accuracies = [
    accuracy_score(labels, np.argmax(resnet_probs, axis=1)),
    accuracy_score(labels, np.argmax(maxent_probs, axis=1)),
    accuracy_score(labels, np.argmax(final_probs_single_weight, axis=1)),
    accuracy_score(labels, np.argmax(final_probs_class_weights, axis=1)),
]

plt.bar(methods, accuracies, color=["blue", "green", "orange", "red"])
plt.ylabel("Accuracy")
plt.title("Comparison of Methods")
plt.show()
