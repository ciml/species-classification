This file guides the use of the code assign_class_index1.py
Para executar o código, utilizar o seguinte comando:

pyhton3 assign_class_index.py -c id_registro-id_animal-id_tipo.csv images/*

Onde:
- id_registro-id_animal-id_tipo.csv é um arquivo que contém a identificação de cada registro, com sua referida classe.
- images é a pasta que contém as imagens que devem ser renomeadas.

O código irá renomear cada imagem de acordo com seu registro e sua classe, no seguinte formato:

id_register-class_index-image_number.jpg

id: identificação do registro do SissGeo;
class: Classe presente na imagem;
image number: número da imagem para registros que possuem mais de uma imagem.
