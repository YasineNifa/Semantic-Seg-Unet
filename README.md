# Semantic-Seg-Unet
L'objectif de ce projet est d'entrainer un modèle de segmentation d'images,le modèle est un encodeur-décodeur 
qui a été entrainé sur un ensemble de données de 105 images des présidents des états-unis, et testé sur 27 images  
les images ont d'abord été segmentées manuellement à l'aide de l'outil VGG Annotator qui produit un fichier csv qui contient 
les coordonnées des différents polygones de segmentation, en utilisant ce fichier, un masque a été créé pour chaque image, 
car l'ensemble de données est un peu petit, un pipeline de données a été configuré pour 
appliquer diverses transformations aux images et masques d'apprentissage afin d'aider le modèle à mieux se généraliser. 
