# Animal species-classification: Integration of Deep-Learning and Species Distribution Models for Classification of Animal Species of the Brazilian Fauna

This project seeks to bring a combination between image classification models and species distribution models, focusing on the classification of animal species of the Brazilian fauna.

The automated classification of animals from photos is important in ecology and conservation biology for organizing and understanding the immense diversity of species, as well as facilitating effective conservation and management practices. It is equally important for disease surveillance systems, allowing prompt detection of anomalies in species distributions and boosting citizen-scientist platforms by making user-reported data more accurate and convenient. Image classification uses photos and can also rely on the geographical locations of animals to improve performance. While image classification models have difficulties in classifying low-quality images, unbalanced datasets, and with a small number of images, species distribution models have difficulty in classifying species that coexist in a given region. The present work presents an integration of image classification models based on deep neural networks with species distribution models. It is applied to a real-world dataset comprising fifteen classes of animals from the Brazilian fauna obtained from Fiocruz's citizen-scientist Wildlife Health Information System (SISS-Geo). The SISS-Geo photos portray the reality of animals in their environments, with varying quality, and pose numerous difficulties for classification.

A estrutura do projeto é a seguinte:

 - class_id: Given a CSV file containing rows of the form: id_register,id_animal,class_index; and a bunch of JPEG filenames as id_register-image_number.jpg, this will assign the class index to each of these filenames in the form - id_register-class_index-image_number.jpg
 - cnn_model: Image classification model; The ResNet-50 network is used to classify images of animals from the Brazilian fauna.
 - data_augmentation: From the images, the data augmentation process is carried out to treat and balance the set of images available for the execution of the deep neural network. The operations of mirroring, zooming, rotating and tiling the images are carried out. Furthermore, the dimensions of the images are standardized.
 - genetic_alg: Genetic algorithm to combine image classification and species distribution models.
 - sdm_model: Species distribution model, used to estimate the existence of animals according to their geographic location.
 - sissgeo_dataset: Dataset with records of Brazilian fauna animals used for classification.
