# [Projet de chatbot intelligent]

## À propos

Ce projet a pour objectif de déployer un chatbot intellignet utilisant un systéme RAG

## Table des matières

- 🪧 [À propos](#à-propos)
- 📦 [Prérequis](#prérequis)
- 🚀 [Installation](#installation)
- 🛠️ [Utilisation](#utilisation)
- 🤝 [Contribution](#contribution)
- 🏗️ [Construit avec](#construit-avec)
- 📝 [Authentification](#Authentification)
- 📚 [Documentation](#documentation)
- 🏷️ [Gestion des versions](#gestion-des-versions)
- 📝 [Licence](#licence)

## Prérequis

IL faut avoir d'installé sur la machine python, pip et PostgreSQL

## Installation

Dans le dossier du projet via un terminal installer avec la commande pip install les lib suivante :
langchain-community==0.4.2
langchain-huggingface==1.2.2
faiss-cpu==1.15.1
sentence-transformers==6.1.0
mistralai==3.0.0
python-dotenv==1.2.3
pandas==2.3.3
requests==2.33.1
python-dateutil==2.9.0.post0
langchain-core==1.6.6
fastapi==0.139.2
uvicorn[standard]
langchain-mistralai==1.1.6
numpy==2.4.6
tiktoken==0.14.0

ragas
datasets
pandas
pytest
pytest-cov

ou sinon via pip install -r requirements.txt

## Utilisation

### lancer un conteneur docker
docker run --rm `                                                                       
>>   --name sys-rag-api `                                   
>>   --env-file .env `
>>   -p 8000:8000 `
>>   sys-rag-api:latest

puis se rendre ici : http://127.0.0.1:8000/

### lancement tests avec rapport de couverture
python -m pytest --cov=app --cov=db --cov-report=term-missing

rapport dans le dossier htmlcov dans le fichier index.html

### obtenir les lib et leurs version actuellemnt installé dans le projet
python -m pip freeze     

## Contribution

Developpeur : N.Ulrick

## Construit avec

### Langages & Frameworks

Le langage python, le gestionnaire Git et gitflow, test écrit avec pytest, évaluation du systéme avec Ragas

## Authentification

GitHub : avoir une clé ssh valide généré coté github

## Documentation

github : https://github.com/
swagger : http://127.0.0.1:8000/docs#/default

## Gestion des versions

la gestion des version des versions et tag sont faite à la main via les commandes suivante :
git tag -a vX.X.X -m "Release vX.X.X"    
git push origin vX.X.X   

## Licence

Voir le fichier [LICENSE](./LICENSE.md) du dépôt.
