# 📊 Reddit Sentiment Analyzer

Un outil d'analyse de sentiment pour Reddit permettant de visualiser les tendances émotionnelles par subreddit via un dashboard interactif.

## 🎯 Objectifs du Projet

Ce projet vise à :
- Analyser les sentiments des commentaires Reddit
- Identifier les tendances émotionnelles par subreddit
- Visualiser les résultats via un dashboard interactif
- Détecter les changements de sentiment dans le temps
- Comparer les sentiments entre différents subreddits

## 🚀 Fonctionnalités

### 1. Preprocessing du Texte
- Nettoyage des données (HTML, URLs, mentions)
- Tokenisation
- Suppression des mots vides
- Lemmatisation
- Versions Light et Hard du preprocessing

### 2. Algorithmes d'Embedding 🧠
Le projet implémente **5 algorithmes d'embedding** différents pour transformer le texte en vecteurs numériques :

#### Algorithmes Basiques :
- **Bag of Words (BoW)** - Comptage simple des mots
- **TF-IDF** - Pondération par fréquence et rareté des mots

#### Algorithmes Avancés :
- **Word2Vec** - Embeddings contextuels denses 
- **FastText** - Word2Vec avec gestion des mots hors vocabulaire
- **BERT** - Transformer pré-entraîné état de l'art

#### 📊 Tests et Comparaisons :
- Notebook complet de test des algorithmes (`test_embedding_algorithms.ipynb`)
- Comparaisons visuelles (graphiques, heatmaps de similarité)
- Évaluation des performances pour l'analyse de sentiment Reddit
- **Recommandation** : BERT ou FastText pour capturer les nuances émotionnelles

### 3. Analyse de Sentiment (🚧 À venir)
- Classification des sentiments (positif/négatif/neutre)
- Détection des émotions spécifiques
- Analyse temporelle des tendances
- Comparaison entre subreddits

### 4. Dashboard (🚧 À venir)
- Visualisation des tendances de sentiment
- Filtres par période et subreddit
- Graphiques interactifs
- Export des données

## 📁 Structure du Projet

```
reddit-analyser/
├── embedding/                    # Modules d'embedding
│   ├── bag_of_words.py          # Bag of Words
│   ├── tf_idf.py                # TF-IDF  
│   ├── word2vec.py              # Word2Vec
│   ├── fasttext.py              # FastText
│   └── bert.py                  # BERT
├── preprocessing/               # Modules de préprocessing
│   ├── basic/                   # Version complète
│   └── light/                   # Version allégée
├── test_*.ipynb                 # Notebooks de test
└── requirements.txt             # Dépendances Python
```

## 🔬 État Actuel du Projet

### ✅ **Terminé :**
- Implémentation des 5 algorithmes d'embedding
- Modules de preprocessing (basic + light)
- Tests complets des algorithmes (notebook avec visualisations)

### 🚧 **En cours / À faire :**
- **Modèle d'analyse de sentiment** (prochaine étape)
- **Tests des combinaisons** preprocessing + embedding + modèle
- Collecte de données Reddit via API
- Interface dashboard
- Déploiement

### 🎯 **Prochaines étapes :**
1. Développer le modèle de classification de sentiment
2. Tester toutes les combinaisons algorithmiques
3. Évaluer les performances sur des données Reddit réelles
4. Sélectionner la meilleure pipeline pour la production

## 🛠️ Installation

### Prérequis
- Python 3.11+ recommandé (gensim non compatible avec Python 3.13)
- Visual Studio Build Tools (pour Windows)

### Installation des dépendances
```bash
# Créer l'environnement virtuel
python -m venv venv311

# Activer l'environnement (Windows)
venv311\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Packages principaux
- `scikit-learn` - Machine learning
- `gensim` - Word2Vec/FastText
- `transformers` - BERT
- `nltk` - Preprocessing
- `matplotlib/seaborn` - Visualisations

## 📊 Tests et Évaluation

Exécuter les notebooks de test pour explorer les algorithmes :
- `test_basic_preprocessing.ipynb` - Tests preprocessing complet
- `test_light_preprocessing.ipynb` - Tests preprocessing allégé  
- `test_embedding_algorithms.ipynb` - **Tests et comparaisons des embeddings**
