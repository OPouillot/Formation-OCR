# Formation Data Scientist

Ce dépôt regroupe les livrables que j'ai fourni aux différents projets de ma formation **Data Scientist** d'OpenClassRooms.

## Description du dossier

Ce dossier contient :
- Un dossier par projet, dans lequel se trouve les livrables (code et présentation de soutenance)
- Des slides présentant un résumé de chacune des réalisations

## Description des projets

### Projet 1 - Analysez des données de systèmes éducatifs : 
Analyse exploratoire sur les données sur l’éducation de la banque mondiale pour proposer une stratégie d'expansion à l'internationale pour une startup de formation en ligne.  
[Lien vers dataset (World Bank - EdStats)](https://datacatalog.worldbank.org/search/dataset/0038480)

### Projet 2 - Concevoir une application au service de la santé
Participation à un appel à projet de Santé Publique France. Réalisation d'une analyse de faisabilité d'une application de santé publique :  
**VégiFit**, application de suivi des apports nutritionnels pour sportif végétarien.  
[Lien vers dataset (OpenFoodFacts)](https://world.openfoodfacts.org/)

### Projet 3 - Anticipez les besoins en consommation de bâtiments
Concevoir un modèle de prédiction de consommation énergétique & d'émissions de CO2 de bâtiments pour la ville de Seattle.  
Réalisation d'une analyse exploratoire ainsi qu'une sélection de modèle basée sur le score de prédiction.  
[Lien vers dataset (Seattle Open Data)](https://data.seattle.gov/dataset/2016-Building-Energy-Benchmarking/2bpz-gwpy)

### Projet 4 - Segmenter des clients d'un site e-commerce
Etablir une segmentation des clients du site Olist, utilisable pour les campagnes de communication. Fournir à l’équipe marketing une description actionable de votre segmentation et de sa logique sous-jacente pour une utilisation optimale.  
Fournir également une proposition de contrat de maintenance basée sur une analyse de la stabilité des segments au cours du temps (simulation de data drift pour actualisation du modèle basé sur l'obsolescence des données).  
[Lien vers dataset (kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

### Projet 5 - Classification automatique des biens de consommation
- Etude de faisabilité d'un moteur de classification d'articles, basé sur une image et une description pour une nouvelle marketplace e-commerce.
  1. Par description : Count, TF-IDF,  Word2Vec, BERT, USE  
  2. Par photos : SIFT, CNN  
- Réalisation d'une classification supervisée à partir des images (CNN, transfert learning)  
[Lien vers dataset (OpenClassRooms)](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Textimage+DAS+V2/Dataset+projet+pre%CC%81traitement+textes+images.zip)

### Projet 6 - Implémenter un modèle de scoring
- Objectif : Mettre en œuvre un outil de “scoring crédit” pour la société financière "Prêt à dépenser", permettant de calculer la probabilité qu’un client rembourse son crédit afin de déterminer l'accord ou non d'un prêt.  
- Conception du modèle de scoring (prétraitement, score métier, tracking MLFlow)  
- Déploiement du modèle sous forme d'une API (FastAPi, Azure Web App)  
- Réalisation d'un Dashboard interactif d'interprétation des résultats (Streamlit)  
[Lien vers dataset (kaggle)](https://www.kaggle.com/c/home-credit-default-risk/data)

### Projet 7 - Déployer un modèle dans le cloud
Etablir une chaîne de traitement en vu d'implémenter un moteur de classification d'images de fruits.  
Mise en place des premières briques de traitement dans un environnement Big Data AWS EMR.  
[Lien vers dataset (kaggle)](https://www.kaggle.com/datasets/moltean/fruits)
