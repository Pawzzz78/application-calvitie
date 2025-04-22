# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import requests
from io import StringIO

app = Flask(__name__)

# Configuration pour Render
# Spécifier le répertoire des fichiers statiques (si nécessaire)
if os.environ.get('RENDER'):
    app.config['STATIC_FOLDER'] = os.path.join(os.getcwd(), 'static')

# URL du fichier CSV (remplacer par l'URL réelle de votre fichier)
CSV_URL = "https://raw.githubusercontent.com/Pawzzz78/application-calvitie/main/bald_probability.csv"

def telecharger_csv():
    """
    Télécharge le fichier CSV depuis GitHub ou une autre source
    """
    try:
        print("Tentative de téléchargement du fichier CSV...")
        response = requests.get(CSV_URL)
        if response.status_code == 200:
            with open("bald_probability.csv", "w", encoding="utf-8") as f:
                f.write(response.text)
            print("Fichier CSV téléchargé avec succès")
            return True
        else:
            print(f"Erreur lors du téléchargement du CSV: {response.status_code}")
            return False
    except Exception as e:
        print(f"Exception lors du téléchargement du CSV: {str(e)}")
        return False

def entrainer_modele(fichier_csv="bald_probability.csv", use_random_forest=True):
    """
    Entraîne le modèle de prédiction de calvitie et le sauvegarde.
    """
    # Vérifier si le fichier existe
    if not os.path.exists(fichier_csv):
        print(f"Erreur: Le fichier {fichier_csv} n'existe pas.")
        print(f"Répertoire courant: {os.getcwd()}")
        print(f"Contenu du répertoire: {os.listdir('.')}")
        raise FileNotFoundError(f"Le fichier {fichier_csv} est introuvable")
    
    try:
        # Charger les données
        df = pd.read_csv(fichier_csv)
        
        # Nettoyer les données (supprimer les lignes avec des valeurs manquantes)
        df_clean = df.dropna()
        
        # Vérifier si la colonne cible existe
        if "bald_prob" not in df_clean.columns:
            raise ValueError("La colonne 'bald_prob' n'existe pas dans le fichier CSV")
        
        # Séparer les caractéristiques et la cible
        X = df_clean.drop("bald_prob", axis=1)
        y = df_clean["bald_prob"]
        
        # Identifier les caractéristiques catégorielles
        cat_features = [col for col in X.columns if X[col].dtype == "object"]
        num_features = [col for col in X.columns if X[col].dtype != "object"]
        
        # Création du préprocesseur pour les variables catégorielles
        preprocesseur = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ], remainder="passthrough")
        
        # Choisir le modèle en fonction du paramètre
        if use_random_forest:
            # Modèle RandomForest optimisé pour de meilleures prédictions
            regressor = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        else:
            regressor = LinearRegression()
        
        # Créer le pipeline avec le préprocesseur et le modèle de régression
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocesseur),
            ("regressor", regressor)
        ])
        
        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entraîner le modèle
        pipeline.fit(X_train, y_train)
        
        # Sauvegarder le modèle
        model_filename = 'modele_calvitie_rf.joblib' if use_random_forest else 'modele_calvitie.joblib'
        joblib.dump(pipeline, model_filename)
        
        return pipeline
        
    except Exception as e:
        print("Erreur lors de l'entraînement du modèle:", str(e))
        print("Type d'erreur:", type(e).__name__)
        import traceback
        traceback.print_exc()
        raise e

def predire_calvitie(age, genre, role_professionnel, province, salaire, est_marie, est_hereditaire,
                   poids, taille, shampoing, est_fumeur, education, stress, use_random_forest=True):
    """
    Prédit la probabilité de calvitie en fonction des caractéristiques de la personne.
    """
    # Mode test pour Render si le CSV n'est pas disponible
    if os.environ.get('RENDER') and not os.path.exists('bald_probability.csv'):
        print("Mode test activé pour Render (pas de fichier CSV disponible)")
        # En mode test, on retourne une valeur basée sur certains critères simples
        if genre == "male":
            base_proba = 0.5
        else:
            base_proba = 0.3
            
        # Facteurs d'ajustement
        if est_hereditaire == 1:
            base_proba += 0.2
        if age > 40:
            base_proba += 0.15
        if stress > 7:
            base_proba += 0.1
        if est_fumeur == 1:
            base_proba += 0.05
            
        # Limiter la valeur entre 0 et 1
        return max(0, min(1, base_proba))
    
    # Comportement normal avec le modèle
    # Déterminer le fichier de modèle en fonction du type de modèle
    model_file = 'modele_calvitie_rf.joblib' if use_random_forest else 'modele_calvitie.joblib'
    
    # Charger le modèle ou l'entraîner s'il n'existe pas
    if os.path.exists(model_file):
        try:
            pipeline = joblib.load(model_file)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {str(e)}")
            pipeline = entrainer_modele(use_random_forest=use_random_forest)
    else:
        try:
            pipeline = entrainer_modele(use_random_forest=use_random_forest)
        except FileNotFoundError:
            # Si le fichier CSV n'est pas trouvé, on utilise le mode test
            return predire_calvitie(age, genre, role_professionnel, province, salaire, est_marie, 
                                 est_hereditaire, poids, taille, shampoing, est_fumeur, 
                                 education, stress, use_random_forest=use_random_forest)
    
    # Créer un DataFrame avec les caractéristiques de la personne
    donnees_personne = pd.DataFrame({
        'age': [age],
        'gender': [genre],
        'job_role': [role_professionnel],
        'province': [province],
        'salary': [salaire],
        'is_married': [est_marie],
        'is_hereditary': [est_hereditaire],
        'weight': [poids],
        'height': [taille],
        'shampoo': [shampoing],
        'is_smoker': [est_fumeur],
        'education': [education],
        'stress': [stress]
    })
    
    # Faire la prédiction
    try:
        probabilite = pipeline.predict(donnees_personne)[0]
        
        # Limiter la valeur entre 0 et 1
        probabilite = max(0, min(1, probabilite))
        
        return probabilite
    except Exception as e:
        # En cas d'erreur, utiliser le mode test
        print(f"Erreur lors de la prédiction, passage en mode test: {str(e)}")
        if hasattr(pipeline, 'feature_names_in_'):
            print("Colonnes attendues par le modèle: {}".format(pipeline.feature_names_in_.tolist()))
            
        # Utiliser le mode test comme fallback
        return predire_calvitie(age, genre, role_professionnel, province, salaire, est_marie, 
                             est_hereditaire, poids, taille, shampoing, est_fumeur, 
                             education, stress, use_random_forest=use_random_forest)

def interpreter_probabilite(probabilite):
    """
    Interprète la probabilité de calvitie et fournit une explication.
    """
    if probabilite < 0.3:
        categorie = "Risque faible"
        explication = "Il y a peu de chances que vous développiez une calvitie significative."
        couleur = "green"
    elif probabilite < 0.6:
        categorie = "Risque modéré"
        explication = "Vous avez un risque modéré de développer une calvitie. Certaines mesures préventives pourraient être utiles."
        couleur = "orange"
    else:
        categorie = "Risque élevé"
        explication = "Vous avez un risque élevé de développer une calvitie. Il serait conseillé de consulter un dermatologue."
        couleur = "red"
    
    return {
        "probabilite": probabilite,
        "categorie": categorie,
        "explication": explication,
        "couleur": couleur
    }

# Initialisation du modèle
def initier_modele():
    """Fonction pour initialiser le modèle si nécessaire"""
    try:
        # Vérifier si on est sur Render
        if os.environ.get('RENDER'):
            print("Déploiement sur Render détecté")
            # Si le fichier CSV n'existe pas, le télécharger
            if not os.path.exists('bald_probability.csv'):
                print("Fichier CSV non trouvé, tentative de téléchargement...")
                telecharger_csv()
            
        # Si le fichier CSV existe, on peut entraîner le modèle
        if os.path.exists('bald_probability.csv'):
            if not os.path.exists('modele_calvitie_rf.joblib'):
                print("Initialisation du modèle...")
                entrainer_modele(use_random_forest=True)
                print("Modèle créé avec succès")
        else:
            print("Attention: Fichier CSV introuvable, le modèle utilisera un mode de fallback")
    except Exception as e:
        print("Erreur lors de l'initialisation du modèle: {}".format(str(e)))
        import traceback
        traceback.print_exc()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/reset_model', methods=['GET'])
def reset_model():
    """Endpoint pour forcer la réinitialisation du modèle"""
    try:
        # Supprimer les anciens modèles
        if os.path.exists('modele_calvitie.joblib'):
            os.remove('modele_calvitie.joblib')
            
        if os.path.exists('modele_calvitie_rf.joblib'):
            os.remove('modele_calvitie_rf.joblib')
        
        # Entraîner un nouveau modèle RandomForest
        entrainer_modele(use_random_forest=True)
        
        return jsonify({
            "success": True,
            "message": "Le modèle RandomForest a été réinitialisé avec succès."
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        data = request.form
        
        # Convertir les données
        age = int(data.get('age'))
        genre = data.get('genre')
        role_professionnel = data.get('role_professionnel')
        province = data.get('province')
        salaire = float(data.get('salaire'))
        est_marie = int(data.get('est_marie'))
        est_hereditaire = int(data.get('est_hereditaire'))
        poids = float(data.get('poids'))
        taille = float(data.get('taille'))
        shampoing = data.get('shampoing')
        est_fumeur = int(data.get('est_fumeur'))
        education = data.get('education')
        stress = int(data.get('stress'))
        
        # Faire la prédiction
        probabilite = predire_calvitie(
            age, genre, role_professionnel, province, salaire, est_marie, est_hereditaire,
            poids, taille, shampoing, est_fumeur, education, stress
        )
        
        # Interpréter le résultat
        resultat = interpreter_probabilite(probabilite)
        
        return render_template('result.html', 
                              probabilite=resultat["probabilite"]*100, 
                              categorie=resultat["categorie"], 
                              explication=resultat["explication"],
                              couleur=resultat["couleur"])
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Récupérer les données JSON
        data = request.json
        
        # Convertir les données
        age = int(data.get('age'))
        genre = data.get('genre')
        role_professionnel = data.get('role_professionnel')
        province = data.get('province')
        salaire = float(data.get('salaire'))
        est_marie = int(data.get('est_marie'))
        est_hereditaire = int(data.get('est_hereditaire'))
        poids = float(data.get('poids'))
        taille = float(data.get('taille'))
        shampoing = data.get('shampoing')
        est_fumeur = int(data.get('est_fumeur'))
        education = data.get('education')
        stress = int(data.get('stress'))
        
        # Faire la prédiction
        probabilite = predire_calvitie(
            age, genre, role_professionnel, province, salaire, est_marie, est_hereditaire,
            poids, taille, shampoing, est_fumeur, education, stress
        )
        
        # Interpréter le résultat
        resultat = interpreter_probabilite(probabilite)
        
        return jsonify({
            "probabilite": resultat["probabilite"],
            "categorie": resultat["categorie"],
            "explication": resultat["explication"]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Initialiser le modèle avant de démarrer l'application
    initier_modele()
    
    # Démarrer l'application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 
