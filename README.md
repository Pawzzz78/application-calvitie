# Application de Prédiction de Calvitie

Cette application utilise le machine learning pour prédire la probabilité de calvitie en fonction de différents paramètres.

## Déploiement sur Render

### Prérequis
- Un compte GitHub
- Un compte Render connecté à GitHub

### Étapes pour déployer sur Render

1. **Créer un nouveau Web Service**
   - Connectez-vous à Render
   - Cliquez sur "New" puis "Web Service"
   - Connectez votre dépôt GitHub

2. **Configuration**
   - Nom: `app-prediction-calvitie` (ou le nom de votre choix)
   - Runtime: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`

3. **Variables d'environnement**
   - Ajoutez `RENDER=true` dans les variables d'environnement

4. **Déployer**
   - Cliquez sur "Create Web Service"

## Développement local

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Exécution
```bash
python app.py
```

L'application sera accessible à l'adresse http://localhost:5000

## Modèle

L'application utilise un modèle RandomForest pour la prédiction. Pour que le modèle fonctionne correctement, vous devez disposer du fichier de données `bald_probability.csv`.

Si le fichier CSV n'est pas disponible (par exemple sur Render), l'application utilisera un mode de fallback pour générer des prédictions basées sur des règles simples.

## Structure du projet

- `app.py` : Application Flask principale
- `requirements.txt` : Dépendances Python
- `templates/` : Templates HTML
- `Procfile` : Configuration pour les plateformes de déploiement (Heroku, Render)
- `.env` : Variables d'environnement locales
- `.gitignore` : Fichiers à ignorer pour git

## Remarques importantes

- Le fichier modèle `modele_calvitie_rf.joblib` est très volumineux (80+ MB) et est exclu de Git. Il sera automatiquement regénéré lors du premier démarrage si le fichier CSV est disponible.
- Sur Render, si le fichier CSV n'est pas disponible, l'application utilisera un mode de fallback pour les prédictions.

## Fonctionnalités

- Interface utilisateur intuitive avec formulaire de saisie
- Prédiction basée sur un modèle de régression linéaire
- Résultats visuels avec interprétation
- API JSON pour l'intégration avec d'autres services

## Installation locale

1. Clonez ce dépôt
2. Installez les dépendances requises :
   ```bash
   pip install -r requirements.txt
   ```
3. Assurez-vous que le fichier `bald_probability.csv` est présent dans le répertoire racine
4. Lancez l'application :
   ```bash
   python app.py
   ```
5. Accédez à l'application dans votre navigateur à l'adresse `http://localhost:5000`

## Déploiement sur des plateformes gratuites

### Déploiement sur PythonAnywhere

1. Créez un compte sur [PythonAnywhere](https://www.pythonanywhere.com/)
2. Allez dans l'onglet "Web" et créez une nouvelle application web
3. Choisissez Flask et Python 3.9
4. Configurez le chemin vers votre application : `/home/yourusername/mysite/app.py`
5. Téléchargez les fichiers de votre application via l'onglet "Files"
6. Installez les dépendances :
   ```bash
   pip3 install --user -r requirements.txt
   ```
7. Redémarrez votre application

## Structure des fichiers

- `app.py` : Application Flask principale
- `bald_probability.csv` : Fichier de données pour entraîner le modèle
- `modele_calvitie.joblib` : Modèle entraîné (généré automatiquement)
- `templates/` : Dossier contenant les templates HTML
  - `index.html` : Page d'accueil avec formulaire
  - `result.html` : Page de résultat
  - `error.html` : Page d'erreur
- `requirements.txt` : Liste des dépendances

## API JSON

Pour utiliser l'API, envoyez une requête POST à `/api/predict` avec un payload JSON contenant les caractéristiques de la personne :

```json
{
  "age": 45,
  "genre": "male",
  "role_professionnel": "Employee",
  "province": "Paris",
  "salaire": 50000,
  "est_marie": 1,
  "est_hereditaire": 1,
  "poids": 75,
  "taille": 180,
  "shampoing": "Head & Shoulders",
  "est_fumeur": 0,
  "education": "Bachelor Degree",
  "stress": 6
}
```

La réponse sera au format JSON :

```json
{
  "probabilite": 0.62,
  "categorie": "Risque élevé",
  "explication": "Vous avez un risque élevé de développer une calvitie. Il serait conseillé de consulter un dermatologue."
}
```

## Licence

Ce projet est sous licence MIT.

## Avertissement

Cette application est fournie à titre informatif uniquement et ne constitue pas un avis médical professionnel. Les prédictions sont basées sur un modèle statistique simple et ne doivent pas être utilisées comme unique source d'information pour prendre des décisions médicales. 