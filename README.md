# Semantic-Seg-Unet
L'objectif de ce projet est d'entrainer un modèle de segmentation d'images,le modèle est un encodeur-décodeur 
qui a été entrainé sur un ensemble de données de 105 images des présidents des états-unis, et testé sur 27 images

La première étape :
collecter la base de données en utilisant un programme python qui permet de scrapper les images de "obama" et de "trump" par le web

La deuxième étape :
Normaliser les images de notre base de données en redimensionnant ces images sous la meme taille.

La troisième étape :
Ségmenter manuellement les images de notre base de données à l'aide de l'outil VGG Annotator qui permet de produire un fichier csv. Ce dernier contient les coordonnées des différents polygones de segmentation

La quatrième étape :
Créer un masque pour chaque image de notre base de donnée

La cinquième étape :
Augmenter la taille de notre base de données d'entrainement en appliquant divers transformations sur les images de notre base de données et leurs masques.

La sixième étape :
Créer le modèle Unet et l'entrainer sur la base de données d'entrainement

La septième étape :
Déployer le modèle crée en utilisant Flask
