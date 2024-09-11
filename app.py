import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf
import torch
import plotly.graph_objs as go
import os

from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.models import Sequential, load_model, Model # type: ignore
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Embedding, Concatenate, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, Dataset, SequentialSampler
from streamlit_elements import elements, mui

# Définir la configuration de la page pour utiliser toute la largeur de l'écran
st.set_page_config(layout="wide")

# Définir le CSS directement dans le script
css = """
<style>
body, h1, h2, h3, h4, h5, h6, p, div, span, li, a, input, button, .stText, .stMarkdown, .stSidebar, .stTitle, .stHeader, .stRadio {
    font-family: 'Arial', sans-serif;
}
.css-1d391kg, .css-18e3th9, .css-1l02zno, .css-1v3fvcr, .css-qbe2hs, .css-1y4p8pa {
    font-family: 'Arial', sans-serif;
}
.small-title {
    font-size: 14px;  /* Ajustez cette valeur selon vos besoins */
    font-weight: bold;
}
.reduced-spacing p {
    margin-bottom: 5px;  /* Ajustez cette valeur selon vos besoins */
    margin-top: 5px;     /* Ajustez cette valeur selon vos besoins */
}
.red-title {
    color: #BF0000;
}
.stRadio > div > div {
    background-color: #BF0000 !important;
}
.center-title {
    text-align: center;
}
</style>
"""

# Injecter le CSS dans l'application Streamlit
st.markdown(css, unsafe_allow_html=True)

# st.sidebar.title("PROJET")

# Ajouter le logo de Rakuten dans la barre latérale avec un lien hypertexte
st.sidebar.markdown(f"""
<a href="https://challengedata.ens.fr/participants/challenges/35/" target="_blank">
    <img src='https://fr.shopping.rakuten.com/visuels/0_content_square/autres/rakuten-logo6.svg' style="width: 100%;">
</a>
""", unsafe_allow_html=True)

st.sidebar.title("Sommaire")
pages = ["Présentation", "Données", "Pré-processing", "Machine Learning", "Deep Learning", "Conclusion", "Démo"]
page = st.sidebar.radio("Aller vers:", pages)

@st.cache_data
def load_data(csv_path):
    return pd.read_csv(csv_path)

@st.cache_data
def load_model_and_tokenizer(model_path, tokenizer_path, le_path):
    model = load_model(model_path)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])  # Ajout de la compilation du modèle
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(le_path, 'rb') as handle:
        le = pickle.load(handle)
    return model, tokenizer, le

@st.cache_data
def preprocess_data(df, _label_encoder):
    df.loc[:, "prdtypecode"] = _label_encoder.fit_transform(df["prdtypecode"])
    train_texts, val_texts, train_labels, val_labels = train_test_split(df["token_text"], df["prdtypecode"], test_size=0.2, random_state=42)
    return train_texts, val_texts, train_labels, val_labels

def tokenize_texts(_tokenizer, texts, max_len=128):
    encodings = _tokenizer(list(texts), truncation=True, padding=True, return_tensors="pt", max_length=max_len)
    return encodings

def prepare_datasets(texts, _labels, _tokenizer, max_len=128):
    class CustomDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.long)
            }

    dataset = CustomDataset(texts=texts.tolist(), labels=_labels.tolist(), tokenizer=_tokenizer, max_len=max_len)
    return dataset

def predict_and_evaluate(_model, loader, device):
    _model.eval()
    predictions = []
    true_labels = []
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in loader:
        input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

        with torch.no_grad():
            outputs = _model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        logits = outputs.logits

        total_eval_loss += loss.item()
        preds = torch.argmax(logits, dim=1).flatten()
        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        total_eval_accuracy += accuracy
        nb_eval_steps += 1

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    avg_val_accuracy = total_eval_accuracy / nb_eval_steps
    avg_val_loss = total_eval_loss / nb_eval_steps

    return np.array(predictions), np.array(true_labels), avg_val_accuracy, avg_val_loss

# Présentation du projet
if page == "Présentation":
    st.markdown("<h1 class='red-title center-title'>Projet Rakuten - Classification Multimodal</h1>", unsafe_allow_html=True)

    # Tabs for different sections
    tab1, tab2 = st.tabs(["Contexte", "Objectif du projet"])

    with tab1:
        st.write("""
        Dans le cadre d'un challenge organisé par l'ENS et de notre formation data scientist au sein de DataScientest, nous avons pu travailler sur la classification de produits à grande échelle, en développant un projet visant à prédire le type de chaque produit tel que défini dans le catalogue de Rakuten France.
        """)

        col1, col2, col3 = st.columns(3)
        with col2:
            st.image('rakuten_image_entreprise.jpg', caption='Siège social de Rakuten à Futakotamagawa, Tokyo', width=550, use_column_width=True)
        
        st.write("""
        **Rakuten, Inc. (Rakuten Kabushiki-gaisha)**  
        Société japonaise de services internet créée en février 1997. Depuis juin 2010, Rakuten a acquis PriceMinister, premier site de commerce électronique en France.
        """)

    with tab2:
        st.write("""
        **Objectif du projet**  
        L’objectif du projet est la classification multimodale à grande échelle des données de produits en codes de types de produits.  
        Il s’agit alors de prédire le code type des produits à partir de données textuelles et images.
        """)

        st.image('objectif_projet.png', caption='Objectif de projet', width=1000, use_column_width=False)

# Exploration des données
if page == "Données":
    st.markdown("<h1 class='red-title center-title'>Exploration des données</h1>", unsafe_allow_html=True)
    st.write('Le projet comporte 3 jeux de données textuelles et 1 jeu de donnée images.')

    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("- X_train\n- X_test\n- Y_train\n- Fichier Images scindé en 2 fichiers image_train & image_test")
        st.markdown("</div>", unsafe_allow_html=True)

    # Ajouter un selectbox pour choisir le jeu de données
    selected_dataset = st.selectbox("**Sélectionnez le jeu de données :**", ["X_train", "X_test", "Y_train", "Fichier Images Train"])

    # Charger les données
    df_train = load_data('X_train_update.csv')
    df_test = load_data('X_test_update.csv')
    df_target = load_data('Y_train_CVw08PX.csv')

    # Fonctions pour afficher des graphiques
    def plot_missing_values_heatmap(df):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(df.isnull(), cmap=sns.color_palette(['#828282', '#BF0000']), cbar=False, ax=ax)
        st.pyplot(fig)

    def plot_nan_percentage(df, column_name):
        nan_percentage = (df[column_name].isnull().sum() / len(df)) * 100
        non_nan_percentage = 100 - nan_percentage
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(["Valeurs non-NaN", "NaN"], [non_nan_percentage, nan_percentage], color=["#004BAA", "#BF0000"])
        ax.set_xlim(0, 100)
        for i, v in enumerate([non_nan_percentage, nan_percentage]):
            ax.text(v + 1, i, f"{v:.2f}%", color="black", va="center")
        st.pyplot(fig)

    def plot_duplicate_percentage(df, column_name):
        duplicate_counts = df[column_name].duplicated().value_counts()
        unique_percentage = (duplicate_counts[False] / len(df)) * 100
        duplicate_percentage = (duplicate_counts[True] / len(df)) * 100
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(["Valeurs uniques", "Doublons"], [unique_percentage, duplicate_percentage], color=["#004BAA", "#BF0000"])
        ax.set_xlim(0, 100)
        for i, v in enumerate([unique_percentage, duplicate_percentage]):
            ax.text(v + 1, i, f"{v:.2f}%", color="black", va="center")
        st.pyplot(fig)
        
    # Afficher les informations en fonction du choix
    if selected_dataset == "X_train":
        st.write("Vous avez sélectionné le jeu de données X_train.")
        st.data_editor(
            df_train.head(),
            column_config={
                "productid": st.column_config.NumberColumn(format="%d"),
                "imageid": st.column_config.NumberColumn(format="%d")
            },
            hide_index=True,
        )
        st.image('image_variable X_train_file.png', caption='Schéma des variables du jeu de données X_train', width=1000, use_column_width=False)
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        with col1:
            plot_missing_values_heatmap(df_train)
        with col2:
            plot_nan_percentage(df_train, column_name='description')
        with col3:
            plot_duplicate_percentage(df_train, column_name='designation')
        with col4:
            st.image('histogramme_langues_X_train.png', width=100, use_column_width=True)
            
    elif selected_dataset == "X_test":
        st.write("Vous avez sélectionné le jeu de données X_test.")
        st.data_editor(
            df_test.head(),
            column_config={
                "productid": st.column_config.NumberColumn(format="%d"),
                "imageid": st.column_config.NumberColumn(format="%d")
            },
            hide_index=True,
        )
        st.image('image variable X_test_file.png', caption='Schéma des variables du jeu de données X_test', width=1000, use_column_width=False)
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        with col1:
            plot_missing_values_heatmap(df_test)
        with col2:
            plot_nan_percentage(df_test, column_name='description')
        with col3:
            plot_duplicate_percentage(df_test, column_name='designation')
        with col4:
            st.image('histogramme_langues_X_test.png', width=100, use_column_width=True)

    elif selected_dataset == "Y_train":
        st.write("Vous avez sélectionné le jeu de données Y_train.")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df_target.head())
        with col2:
            st.image('image_variable Y_train_file.png', caption='Schéma des variables du jeu de données Y_train', width=1000, use_column_width=True)
        col3, col4 = st.columns(2)
        with col3:
            plot_missing_values_heatmap(df_target)
        with col4:
            st.image('visualisation_pdt_categorie.png', caption='Visualisation du nombre de produit par catégorie', width=100, use_column_width=True)
        st.image('categorie_produit.png', caption='27 Catégories de produits', width=500, use_column_width=False)

    elif selected_dataset == "Fichier Images Train":
        st.write("Vous avez sélectionné le fichier d'images du jeu de données images.")
        st.write("Le jeu de données images comporte 2 fichiers : 1 image_train et 1 image_test.")
        col1, col2 = st.columns(2)
        with col1:
            st.image('visualisation_fichier_image.png', caption='Visualisation du fichier images_train', width=100, use_column_width=True)
        with col2:
            st.image('dataframe_images_train.png', caption='DataFrame du fichier images_train', width=100, use_column_width=True)
        show_rapprochement = st.checkbox('**Rapprochement Textes-Images-Cible**')
        if show_rapprochement:
            st.markdown("<h2 style='text-align: center; color: #004BAA; font-family: Calibri, sans-serif;'>Rapprochement Textes - Images - Variable Cible</h2>", unsafe_allow_html=True)
            st.write('Voici un exemple une visualisation de rapprochement de textes, des images avec la variable cible correspondante')
            st.image('rapprochement texte_image_cible.png', caption='rapprochement texte/image & catégorie de produit', width=1000, use_column_width=False)

# prétraitement
if page == "Pré-processing":
    st.markdown("<h1 class='red-title center-title'>Pré-processing</h1>", unsafe_allow_html=True)
    
    st.write("""
    Pour ce projet, nous avons utilisé 2 catégories de données non structurées : texte et image.
    Ces données peuvent être vectorisées de plusieurs manières différentes. Nous avons donc essayé plusieurs scénarios pour trouver le plus performant en testant les modèles.
    Voici les différents scénarios de pré-processing testés.
    """)

    st.image('scenario_preprocessing.png', caption='Représentation des scénarios de pré-processing', width=1000, use_column_width=False)

    show_scenarios = st.checkbox('**Afficher les scénarios**')
    if show_scenarios:
        st.markdown('**Scénario A :** Vectorisation des images par CNN, vectorisation du texte avec SPACY sans traduction de texte')
        st.markdown('**Scénario B :** Vectorisation des images par CNN, vectorisation du texte avec TF-IDF, après tokenisation, lemmatisation, application des stop-words, sans traduction de texte et une réduction par TruncatedSVD')
        st.markdown('**Scénario C :** Même vectorisation que le scénario B, sans application de réduction')
        st.markdown('**Scénario D :** Vectorisation des images par une succession de transformations des images (transformations en gris, application de filtre Gaussien puis Laplacian, et réduction de la taille des images), vectorisation du texte avec TF-IDF, après tokenisation, lemmatisation, application des stop-words, sans traduction de texte et une réduction par TruncatedSVD')
        st.markdown('**Scénario E :** Même vectorisation que scénario B avec traduction de texte dans la langue majoritaire, à savoir le français')

elif page == "Machine Learning":
    st.markdown("<h1 class='red-title center-title'>Machine Learning</h1>", unsafe_allow_html=True)
    
    # Définir les onglets
    tabs = st.tabs(["Scénario A", "Scénario B", "Scénario E", "Amélioration", "Optimisation"])

    # Fonction pour ajouter les expanders de modèles avec des images spécifiques
    def add_model_expanders(images):
        with st.expander(f"**XGboost** Score F1-pondéré: {images['XGboost']['score']}"):
            st.write(f"Détails sur le modèle XGboost.")
            st.image(images['XGboost']['path'], caption=None, width=1200)
        with st.expander(f"**SGD Classifier** Score F1-pondéré: {images['SGD Classifier']['score']}"):
            st.write(f"Détails sur le modèle SGD Classifier.")
            st.image(images['SGD Classifier']['path'], caption=None, width=1200)
        with st.expander(f"**Random Forest** Score F1-pondéré: {images['Random Forest']['score']}"):
            st.write(f"Détails sur le modèle Random Forest.")
            st.image(images['Random Forest']['path'], caption=None, width=1200)
        with st.expander(f"**Voting Classifier 'Soft'** Score F1-pondéré: {images['Voting Classifier Soft']['score']}"):
            st.write(f"Détails sur le modèle Voting Classifier 'Soft'.")
            st.image(images['Voting Classifier Soft']['path'], caption=None, width=1200)
        with st.expander(f"**Voting Classifier 'Hard'** Score F1-pondéré: {images['Voting Classifier Hard']['score']}"):
            st.write(f"Détails sur le modèle Voting Classifier 'Hard'.")
            st.image(images['Voting Classifier Hard']['path'], caption=None, width=1200)
        with st.expander(f"**Naive Bayes Gaussien** Score F1-pondéré: {images['Naive Bayes Gaussien']['score']}"):
            st.write(f"Détails sur le modèle Naive Bayes Gaussien.")
            st.image(images['Naive Bayes Gaussien']['path'], caption=None, width=1200)

    # Images pour chaque scénario avec scores
    images_scenario_A = {
        'XGboost': {'path': 'A_XGboost.png', 'score': 0.73},
        'SGD Classifier': {'path': 'A_SGD Classifier.png', 'score': 0.68},
        'Random Forest': {'path': 'A_Random Forest.png', 'score': 0.65},
        'Voting Classifier Soft': {'path': 'A_VCS.png', 'score': 0.73},
        'Voting Classifier Hard': {'path': 'A_VCH.png', 'score': 0.72},
        'Naive Bayes Gaussien': {'path': 'A_NBG.png', 'score': 0.46}
    }

    images_scenario_B = {
        'XGboost': {'path': 'B_XGboost.png', 'score': 0.77},
        'SGD Classifier': {'path': 'B_SGD Classifier.png', 'score': 0.62},
        'Random Forest': {'path': 'B_Random Forest.png', 'score': 0.76},
        'Voting Classifier Soft': {'path': 'B_VCS.png', 'score': 0.77},
        'Voting Classifier Hard': {'path': 'B_VCH.png', 'score': 0.76},
        'Naive Bayes Gaussien': {'path': 'B_NBG.png', 'score': 0.51}
    }

    images_scenario_E = {
        'XGboost': {'path': 'E_XGboost.png', 'score': 0.76},
        'SGD Classifier': {'path': 'E_SGD Classifier.png', 'score': 0.58},
        'Random Forest': {'path': 'E_Random Forest.png', 'score': 0.75},
        'Voting Classifier Soft': {'path': 'E_VCS.png', 'score': 0.75},
        'Voting Classifier Hard': {'path': 'E_VCH.png', 'score': 0.74},
        'Naive Bayes Gaussien': {'path': 'E_NBG.png', 'score': 0.50}
    }

    # Contenu du tab Scénario A
    with tabs[0]:
        st.header("Scénario A")
        st.write("Vectorisation des images par CNN, vectorisation du texte avec SPACY sans traduction de texte")
        st.write("")
        st.write("Les modèles:")
        add_model_expanders(images_scenario_A)

    # Contenu du tab Scénario B
    with tabs[1]:
        st.header("Scénario B")
        st.write("Vectorisation des images par CNN, vectorisation du texte avec TF-IDF, après tokenisation, lemmatisation, application des stop-words, sans traduction de texte et une réduction par TruncatedSVD")
        st.write("")
        st.write("Les modèles:")
        add_model_expanders(images_scenario_B)

    # Contenu du tab Scénario E
    with tabs[2]:
        st.header("Scénario E")
        st.write("Même vectorisation que scénario B avec traduction de texte dans la langue majoritaire, à savoir le français")
        st.write("")
        st.write("Les modèles:")
        add_model_expanders(images_scenario_E)

    # Contenu du tab Amélioration
    with tabs[3]:
        st.header("Amélioration")
        st.write("Étape 1 : Recherche des Meilleurs Hyperparamètres")
        st.write("")
        with st.expander("**Amélioration B** Score F1-pondéré: 0.79"):
            st.write("Détails sur Amélioration B.")
            st.image("ameb.png", caption=None, width=400)

    # Contenu du tab Optimisation
    with tabs[4]:
        st.header("Optimisation")
        st.write("Étape 2 : Validation Croisée")
        st.write("Étape 3 : Rééchantillonnage et Évaluation")
        st.write("")
        with st.expander("**SMOTE** Score F1-pondéré: 0.80"):
            st.write("Détails sur SMOTE.")
            st.image("SMOTE.png", caption=None, width=400)
        with st.expander("**RandomUnderSampler** Score F1-pondéré: 0.74"):
            st.write("Détails sur RandomUnderSampler.")
            st.image("RUS.png", caption=None, width=400)

elif page == "Deep Learning":
    st.markdown("<h1 class='red-title center-title'>Les modèles de Deep Learning étudiés</h1>", unsafe_allow_html=True)
 
    # Tab creation
    tab1, tab2, tab3, tab4 = st.tabs(["Benchmark", "Les modèles étudiés", "Synthèse des scores de F1-pondéré", "Interprétation"])

    with tab1:
        st.header("Description du Benchmark - Challenge Rakuten")
        st.write("""
        Tout en gardant notre fil de conduite, nous avons réfléchi à plusieurs stratégies de modélisation Deep Learning.  
        Dans un premier temps, nous avons essayé de reproduire les modèles proposés par le « Challenge » afin d’atteindre le score du Benchmark.  
        (Il a été notre référence minimum de score de F1-pondéré)         
                 
        L'algorithme du challenge utilise deux modèles distincts pour les images et le texte.

        **Pour les données d'image :**
        - Il utilise une version du modèle Residual Networks (ResNet50) de Keras. 
        - Le modèle est pré-entraîné avec un ensemble de données ImageNet.
        
        **Pour les données textuelles :**
        - Il utilise un classificateur CNN simplifié.
        - La taille d'entrée est la longueur de désignation maximale possible, 34 dans ce cas.
        - L'architecture se compose d'une couche d'intégration et de 6 blocs convolutionnels à pooling maximal.
        
        **Performances de référence**  
        Voici le score F1 pondéré obtenu à l'aide des 2 modèles du Benchmark décrits ci-dessus :
        - Images : **0,5534**
        - Texte : **0,8113**

        Comme le modèle utilisant du texte est plus performant, Rakuten a utilisé celui-ci comme référence de son Benchmark.
        """)

    with tab2:
        st.header("Les modèles")
        st.write("**Scénario B**")
        st.write("Vectorisation des images par CNN, vectorisation du texte avec TF-IDF, après tokenisation, lemmatisation, application des stop-words, sans traduction de texte et une réduction par TruncatedSVD")

        with st.expander("**Réseau de neurones denses avec Keras** Score F1-pondéré: 0.77"):
            st.write("""
            Les DNN sont une architecture de réseau neuronal artificiel couramment utilisée pour la classification et la prédiction.    
            
            **Description de l'Architecture du Modèle DNN**
            **4 couches** entièrement connectées (ou couches denses).

            - **Couche Dense 1** : La première couche dense prend une entrée et la transforme en une sortie de 512 neurones, avec 113 152 paramètres entraînables.
            - **Couche Dense 2** : La deuxième couche dense réduit cette sortie à 256 neurones, avec 131 328 paramètres entraînables.
            - **Couche Dense 3** : La troisième couche dense réduit encore la sortie à 128 neurones, avec 32 896 paramètres entraînables.
            - **Couche Dense 4** : La dernière couche dense produit la sortie finale de 27 neurones, avec 3 483 paramètres entraînables.

            **Nombre Total de Paramètres**
            - Total des paramètres entraînables : 280 859
            - Total des paramètres non-entraînables : 0 

            **Avantages :**
            - Capacité d'apprentissage des relations complexes données textuelles et images
            - Flexibilité de l’architecture 
            - Robuste aux variations
            """)

            st.write("")
            st.write("")

            # Charger les images
            image1a = "rep_dnn.png"
            image2a = "mtx_dnn.png"

            # Créer deux colonnes
            col1, col2 = st.columns(2)

            # Afficher la première image dans la première colonne
            with col1:
                st.image(image1a, caption='Rapport de classification - Réseau de neurones denses avec Keras', width=450)

            # Afficher la deuxième image dans la deuxième colonne
            with col2:
                st.image(image2a, caption='Matrice de confusion - Réseau de neurones denses avec Keras')

        with st.expander("**ResNet50 - LSTM** Score F1-pondéré: 0.74"):
            st.write("""
            Ce modèle hybride combinant:  
            ResNet50 est une architecture de réseau neuronal convolutif (CNN) performante pour l'extraction de caractéristiques visuelles à partir d'images.  
            LSTM (Long Short-Term Memory) est un type de réseau neuronal récurrent (RNN) spécialisé dans le traitement de données séquentielles, telles que le texte.  
            
            **Description de l'Architecture du Modèle Hybride ResNet50-LSTM**  
            **12 couches**

            - **input_layer (InputLayer)** : (None, 100) - Entrée pour les données textuelles.
            - **input_layer_1 (InputLayer)** : (None, 128, 128, 3) - Entrée pour les images.
            - **embedding (Embedding)** : (None, 100, 128) - Transforme les données textuelles en représentations denses de 128 dimensions.
            - **resnet50 (Functional)** : (None, 4, 4, 2048) - Réseau ResNet50 pré-entraîné pour l'extraction de caractéristiques visuelles.
            - **lstm (LSTM)** : (None, 64) - Réseau LSTM pour capturer les dépendances temporelles dans les données textuelles.
            - **global_average_pooling2d (GlobalAveragePooling2D)** : (None, 2048) - Réduction de la sortie de ResNet50.
            - **dense (Dense)** : (None, 64) - Couche dense appliquée à la sortie du LSTM.
            - **dense_1 (Dense)** : (None, 128) - Couche dense appliquée à la sortie du GlobalAveragePooling.
            - **concatenate (Concatenate)** : (None, 192) - Concatenation des sorties denses.
            - **dense_2 (Dense)** : (None, 256) - Couche dense pour fusionner les représentations.
            - **dropout (Dropout)** : (None, 256) - Couche de régularisation pour éviter le surapprentissage.
            - **dense_3 (Dense)** : (None, 27) - Couche de sortie finale pour la classification.

            **Nombre Total de Paramètres**
            - Total des paramètres : 25,239,899 (96.28 MB)
            - Paramètres entraînables : 1,652,187 (6.30 MB)
            - Paramètres non-entraînables : 23,587,712 (89.98 MB)  

            **Avantages :**
            - Exploitation efficace des informations multimodales : ResNet50 - images, LSTM - texte
            - Apprentissage des relations temporelles : LSTM - compréhension du sens des phrases 
            - Classification précise : Combinaison ResNet50-LSTM classification précise et robuste
            """)

            st.write("")
            st.write("")

            # Charger les images
            image1b = "rep_dnn.png"
            image2b = "mtx_dnn.png"

            # Créer deux colonnes
            col1, col2 = st.columns(2)

            # Afficher la première image dans la première colonne
            with col1:
                st.image(image1b, caption='Rapport de classification - ResNet50 - LSTM', width=450)

            # Afficher la deuxième image dans la deuxième colonne
            with col2:
                st.image(image2b, caption='Matrice de confusion - ResNet50 - LSTM')    
            
        with st.expander("**DistilBERT** Score F1-pondéré: 0.92"):
            st.write("""
            DistilBERT est une version allégée du modèle BERT, qui est un modèle de langage pré-entraîné révolutionnaire pour le traitement du langage naturel (NLP).  
            Il est spécialement conçu pour des tâches de classification, de compréhension et de génération de texte.
            
            **Description Brève de l'Architecture du Modèle DistilBERT**  
            88 couches

            - **distilbert.embeddings.word_embeddings.weight** : (30522, 768) - 23,440,896 paramètres
            - **distilbert.embeddings.position_embeddings.weight** : (512, 768) - 393,216 paramètres
            - **distilbert.embeddings.LayerNorm.weight** : (768,) - 768 paramètres
            - **distilbert.embeddings.LayerNorm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.attention.q_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.0.attention.q_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.attention.k_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.0.attention.k_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.attention.v_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.0.attention.v_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.attention.out_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.0.attention.out_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.sa_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.sa_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.ffn.lin1.weight** : (3072, 768) - 2,359,296 paramètres
            - **distilbert.transformer.layer.0.ffn.lin1.bias** : (3072,) - 3,072 paramètres
            - **distilbert.transformer.layer.0.ffn.lin2.weight** : (768, 3072) - 2,359,296 paramètres
            - **distilbert.transformer.layer.0.ffn.lin2.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.output_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.0.output_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.attention.q_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.1.attention.q_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.attention.k_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.1.attention.k_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.attention.v_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.1.attention.v_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.attention.out_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.1.attention.out_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.sa_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.sa_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.ffn.lin1.weight** : (3072, 768) - 2,359,296 paramètres
            - **distilbert.transformer.layer.1.ffn.lin1.bias** : (3072,) - 3,072 paramètres
            - **distilbert.transformer.layer.1.ffn.lin2.weight** : (768, 3072) - 2,359,296 paramètres
            - **distilbert.transformer.layer.1.ffn.lin2.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.output_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.1.output_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.attention.q_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.2.attention.q_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.attention.k_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.2.attention.k_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.attention.v_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.2.attention.v_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.attention.out_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.2.attention.out_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.sa_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.sa_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.ffn.lin1.weight** : (3072, 768) - 2,359,296 paramètres
            - **distilbert.transformer.layer.2.ffn.lin1.bias** : (3072,) - 3,072 paramètres
            - **distilbert.transformer.layer.2.ffn.lin2.weight** : (768, 3072) - 2,359,296 paramètres
            - **distilbert.transformer.layer.2.ffn.lin2.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.output_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.2.output_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.attention.q_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.3.attention.q_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.attention.k_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.3.attention.k_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.attention.v_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.3.attention.v_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.attention.out_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.3.attention.out_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.sa_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.sa_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.ffn.lin1.weight** : (3072, 768) - 2,359,296 paramètres
            - **distilbert.transformer.layer.3.ffn.lin1.bias** : (3072,) - 3,072 paramètres
            - **distilbert.transformer.layer.3.ffn.lin2.weight** : (768, 3072) - 2,359,296 paramètres
            - **distilbert.transformer.layer.3.ffn.lin2.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.output_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.3.output_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.attention.q_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.4.attention.q_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.attention.k_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.4.attention.k_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.attention.v_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.4.attention.v_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.attention.out_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.4.attention.out_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.sa_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.sa_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.ffn.lin1.weight** : (3072, 768) - 2,359,296 paramètres
            - **distilbert.transformer.layer.4.ffn.lin1.bias** : (3072,) - 3,072 paramètres
            - **distilbert.transformer.layer.4.ffn.lin2.weight** : (768, 3072) - 2,359,296 paramètres
            - **distilbert.transformer.layer.4.ffn.lin2.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.output_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.4.output_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.attention.q_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.5.attention.q_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.attention.k_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.5.attention.k_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.attention.v_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.5.attention.v_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.attention.out_lin.weight** : (768, 768) - 589,824 paramètres
            - **distilbert.transformer.layer.5.attention.out_lin.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.sa_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.sa_layer_norm.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.ffn.lin1.weight** : (3072, 768) - 2,359,296 paramètres
            - **distilbert.transformer.layer.5.ffn.lin1.bias** : (3072,) - 3,072 paramètres
            - **distilbert.transformer.layer.5.ffn.lin2.weight** : (768, 3072) - 2,359,296 paramètres
            - **distilbert.transformer.layer.5.ffn.lin2.bias** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.output_layer_norm.weight** : (768,) - 768 paramètres
            - **distilbert.transformer.layer.5.output_layer_norm.bias** : (768,) - 768 paramètres
            - **pre_classifier.weight** : (768, 768) - 589,824 paramètres
            - **pre_classifier.bias** : (768,) - 768 paramètres
            - **classifier.weight** : (27, 768) - 20,736 paramètres
            - **classifier.bias** : (27,) - 27 paramètres

            **Nombre Total de Paramètres**
            - Total des paramètres : 66,974,235

            **Avantages :**
            - Excellente capacité à capturer les nuances et la sémantique des données textuelles
            - Réduit les coûts de calcul et d'entraînement grâce à sa taille plus petite
            - Favorise une utilisation efficace des ressources matérielles et temps de traitement plus rapides
            - Adaptabilité élevée à différentes tâches NLP grâce à son architecture flexible et pré-entraînée
            """)
                     
            st.write("")
            st.write("")
            
            # Charger les images
            image1c = "rep_d_bert.png"
            image2c = "mtx_d_bert.png"

            # Créer deux colonnes
            col1, col2 = st.columns(2)

            # Afficher la première image dans la première colonne
            with col1:
                st.image(image1c, caption='Rapport de classification - DistilBERT', width=450)

            # Afficher la deuxième image dans la deuxième colonne
            with col2:
                st.image(image2c, caption='Matrice de confusion - DistilBERT')                   
                  
        with st.expander("**EfficientNetB0 - LSTM** Score F1-pondéré: 0.96"):
            st.write("""
            Ce modèle hybride combinant:
            EfficientNetB0 est une famille de modèles de réseaux neuronaux convolutifs (CNN) qui se distinguent par leur efficacité en termes de précision et de coût computationnel pour le traitement d'images.   
            LSTM, quant à lui, est un réseau neuronal récurrent (RNN) spécialisé dans le traitement de données séquentielles, telles que le texte. 
            
            **Description de l'Architecture du Modèle Hybride EfficientNetB0-LSTM**  
            **12 couches**

            - **input_layer (InputLayer)** : (None, 100) - Entrée pour les données textuelles.
            - **input_layer_1 (InputLayer)** : (None, 128, 128, 3) - Entrée pour les images.
            - **embedding (Embedding)** : (None, 100, 128) - Transforme les données textuelles en représentations denses de 128 dimensions.
            - **efficientnetb0 (Functional)** : (None, 4, 4, 1280) - Réseau EfficientNetB0 pré-entraîné pour l'extraction de caractéristiques visuelles.
            - **lstm (LSTM)** : (None, 128) - Réseau LSTM pour capturer les dépendances temporelles dans les données textuelles.
            - **global_average_pooling2d (GlobalAveragePooling2D)** : (None, 1280) - Réduction de la sortie de EfficientNetB0.
            - **dense (Dense)** : (None, 128) - Couche dense appliquée à la sortie du LSTM.
            - **dense_1 (Dense)** : (None, 128) - Couche dense appliquée à la sortie du GlobalAveragePooling.
            - **concatenate (Concatenate)** : (None, 256) - Concatenation des sorties denses.
            - **dense_2 (Dense)** : (None, 256) - Couche dense pour fusionner les représentations.
            - **dropout (Dropout)** : (None, 256) - Couche de régularisation pour éviter le surapprentissage.
            - **dense_3 (Dense)** : (None, 27) - Couche de sortie finale pour la classification.

            **Nombre Total de Paramètres**
            - Total des paramètres : 5,714,366 (21.80 MB)
            - Paramètres entraînables : 1,664,795 (6.35 MB)
            - Paramètres non-entraînables : 4,049,571 (15.45 MB)

            **Avantages :**
            - Excellente capacité à extraire des caractéristiques pertinentes à partir d'images avec une efficacité computationnelle élevée
            - Favorise une utilisation judicieuse des ressources matérielles grâce à son architecture efficace
            - Permet une compréhension profonde des données séquentielles, telles que les descriptions textuelles, grâce à LSTM
            - Offre une classification précise et robuste en combinant les forces de l'extraction d'informations visuelles d'EfficientNetB0 et de la compréhension du sens des phrases de LSTM
            """)
            
            st.write("")
            st.write("")
            
            # Charger les images
            image1d = "rep_eff_lstm.png"
            image2d = "mtx_eff_LSTM.png"

            # Créer deux colonnes
            col1, col2 = st.columns(2)

            # Afficher la première image dans la première colonne
            with col1:
                st.image(image1d, caption='Rapport de classification - EfficientNetB0 - LSTM', width=450)

            # Afficher la deuxième image dans la deuxième colonne
            with col2:
                st.image(image2d, caption='Matrice de confusion - EfficientNetB0 - LSTM')       
           
        with st.expander("**Modèles Non-Aboutis**"):
            st.write("""
            Il a été testé d’autres stratégies de modèle de Deep Learning qui n’ont pas aboutis principalement pour des raisons techniques, de matériel ou de surcoût.

            **Un modèle hybride profond combinant VGG16 et BERT**
            - Le modèle se compose de deux branches principales : 
            - Une Branche VGG16 : Cette branche traite les images d'entrée. Elle utilise l'architecture VGG16 pré-entraînée pour extraire des caractéristiques visuelles de haut niveau à partir des images.
            - Une Branche BERT : Cette branche traite les descriptions textuelles d'entrée. Elle utilise le modèle BERT pré-entraîné pour extraire des représentations sémantiques des textes.
            - Les sorties des deux branches sont ensuite fusionnées pour obtenir une représentation combinée qui capture à la fois des informations visuelles et textuelles. Cette représentation combinée est ensuite utilisée pour effectuer la tâche finale.

            **Le modèle VisualBert**
            - Il permet de comprendre et de traiter conjointement du texte et des images. Il s'appuie sur l'architecture Transformer et utilise deux mécanismes d'apprentissage préliminaire innovants pour capturer des relations sémantiques entre les mots et les régions d'une image.
            - Son fonctionnement est le suivant : 
            - Intégration des informations : VisualBert traite d'abord le texte et l'image séparément à l'aide de deux encodeurs Transformer.
            - Attention mutuelle : Ensuite, il utilise un mécanisme d'attention mutuelle pour permettre aux encodeurs de s'influencer mutuellement et d'aligner les mots du texte avec les régions pertinentes de l'image.
            - Représentation finale : Les sorties des deux encodeurs sont ensuite combinées pour obtenir une représentation finale qui capture à la fois les informations textuelles et visuelles.
            - Ainsi, malgré une compréhension approfondie du modèle et de son architecture, les limitations matérielles ont empêché une exécution complète et efficace de VisualBERT sur ma machine.
        
            **Un modèle multimodal**
            - Capable de traiter et de comprendre des informations provenant de plusieurs sources sensorielles, telles que la vision, l'ouïe et le langage. Son fonctionnement est le suivant :
            - Intégration des données : Le modèle reçoit des données provenant de différentes modalités, comme des images, des sons ou du texte.
            - Extraction de caractéristiques : Il extrait ensuite des caractéristiques pertinentes de chaque modalité de données.
            - Fusion des caractéristiques : Ces caractéristiques sont ensuite fusionnées en une représentation commune qui capture les informations essentielles de toutes les modalités.
            - Apprentissage et prédiction : Le modèle utilise cette représentation commune pour apprendre une tâche spécifique, comme la classification d'objets, la reconnaissance vocale ou la traduction automatique.
            - Limitation des Ressources : Bien que ce modèle détecte les images, il connaît le même problème que VisualBERT. L'exécution du modèle sur ma machine entraîne des arrêts dus à l'insuffisance de ressources matérielles.
            """)        

    with tab3:
        st.header("Synthèse des scores de F1-pondéré")
        st.write("""
                Voici un récapitulatif des scores des modèles étudiés qui ont fonctionnés.  
                """)
        
        col1, col2 = st.columns(2)

        with col1:
            st.write("""
                    Le modèle EfficientNetB0-LSTM se distingue comme le meilleur modèle parmi ceux testés, avec une exactitude, une précision, un rappel et un score F1 pondéré les plus élevés (0.96).   
                    Avec ce modèle nous avons dépassé non seulement le score du Benchmark (0.81), mais aussi le meilleur score du « Challenge » (0.92).  
                    A noté que le modèle DistilBERT atteint lui aussi le niveau du meilleur score, sans le dépasser.  
                    Si l’on analyse séparément les scores texte et image, l’on constate que le modèle LSTM a probablement contribué à atteindre ce score.  
                    Mais ce résultat, provient aussi de l’étape de pré-processing effectuée sur le texte en filtrant un maximum d’anomalies et en préparant le texte à la modélisation.  

                    En revanche, le modèle ResNet50-LSTM affiche les performances les plus faibles, notamment avec un score F1 pondéré (0.74), ce qui est significativement inférieur aux autres modèles.   
                    Ces résultats suggèrent que les modèles basés sur des réseaux de neurones convolutifs sont particulièrement efficaces pour cette tâche de classification, tandis que les DNN pourraient nécessiter des améliorations ou ne pas être aussi bien adaptés à ce type spécifique de données ou de problème.

                    **Nous avons donc choisi le modèle EfficientNetB0-LSTM pour passer à l'étape de prédictions de nos données.**  
                    """)
            
        with col2:       
            st.image("score_deep.png", caption="Synthèse des scores de F1-pondéré")   

    with tab4:
        st.header("Interprétation du texte et des images par le Modèle EfficientNetB0-LSTM")
        st.write("""
        Pour asseoir la viabilité du modèle le plus performant, il faut aussi vérifier son interprétabilité, afin de mettre en exergue ces forces et ces limites, et en établir possiblement des points d’améliorations.  
        De ce fait, nous avons soumis notre, modèle en séparant texte et images, à des outils d'interprétation, pour mieux comprendre comment celui-ci fonctionne.   
        """)

        with st.expander("**Interprétation du texte**"):
            st.write("""
            Les modèles Deep LSTM, particulièrement puissants pour traiter des données textuelles, peuvent cependant s'avérer complexes à interpréter.  
            Il a été essayé plusieurs algorithmes comme LIME, SHAP, Visualisation de l’attention, Integrated Gradients, Saliency Maps, Occlusion, Modèles interprétables intrinsèques, Visualisation des états cachés.  
        
            Seuls 2 algorithmes ont ressorti un résultat qui restent flou à décrypter.  
            """)
            
            st.write("""
            **Interprétation intrinsèque**  
            L'interprétation d'un modèle Deep LSTM avec des techniques intrinsèques implique l'exploration des éléments clés qui contribuent à ses prédictions :  
            L’importance des mots, l’attention, le déploiement des cellules LSTM, et la visualisation des représentations sémantiques.
              
            Dans le cas de notre modèle, voici ce qu’il ressort de l’interprétation selon cette méthode :
            """)   
            st.image("txt_inter_1.png", caption="Interprétation intrinsèque", width=400)

            st.write("""
            **Visualisation des états cachés**  
            Les états cachés d'un modèle LSTM représentent une mémoire interne dynamique qui capture les informations contextuelles au cours du traitement du texte.  
            En visualisant ces états cachés, nous pouvons observer comment le modèle intègre les différentes parties d'une séquence textuelle et comment il utilise ces informations pour formuler ses prédictions. 
            
             Dans le cas de notre modèle, voici ce qu’il ressort de l’interprétation selon cette méthode :  
            """)
            
            col1, col2 = st.columns(2)

            with col1:
                st.image("txt_inter_2.png", caption="Visualisation des états cachés - États cachés", width=400)
                
            with col2:
                st.image("txt_inter_3.png", caption="Visualisation des états cachés - Sorties LSTM", width=400)
            
            st.write("""
            **Remarques:**  
            Il est important de noter que l'interprétation visuelle de ces deux outils peut être subjective et sujette à des biais.  
            Une analyse plus approfondie des graphiques et de leur relation avec les prédictions du modèle de code produit serait nécessaire pour obtenir une compréhension complète du fonctionnement du modèle.  
            De plus, la qualité de l'interprétation dépend de la qualité des données textuelles utilisées pour entraîner le modèle.  
            Des descriptions de produits claires, concises et informatives sont essentielles pour que le modèle puisse identifier correctement les caractéristiques clés et attribuer les codes produits appropriés.  
            L'analyse de ces graphiques peut fournir des informations précieuses sur la façon dont le modèle attribue des codes produits à partir de désignations et descriptions d'objets.  
            En identifiant les mots clés et les structures de description auxquels le modèle accorde le plus d'attention, on peut mieux comprendre les facteurs qui influencent ses prédictions.  
            Cependant, une interprétation plus approfondie et une analyse de la qualité des données sont nécessaires pour une compréhension complète du modèle.
            """)         

        with st.expander("**Interprétation des images**"):
            st.write("""
            **Gradients intégrés (Integrated Gradients)**  
            Cette technique vise à comprendre l'importance de chaque caractéristique d'entrée en mesurant son impact sur la sortie du modèle.  
              
            Dans le cas de notre modèle, voici ce qu’il ressort de l’interprétation selon cette méthode :
            """)   
            st.image("img_inter_1.png", caption="Gradients intégrés", width=400)
            
            st.write("""
            **Cartes de saillance (Saliency Maps)**  
            Est une technique utilisée pour visualiser les parties d'une entrée qui ont le plus contribué à la décision d'un modèle de réseau de neurones.    
            Elles sont couramment utilisées pour interpréter les modèles de vision par ordinateur.  
            Le concept des cartes de saillance repose sur le calcul des gradients des sorties du modèle par rapport aux entrées.  
              
            Dans le cas de notre modèle, voici ce qu’il ressort de l’interprétation selon cette méthode :
            """)   
            st.image("img_inter_2.png", caption="Cartes de saillance", width=400)
                        
            st.write("""
            **SmoothGrad**  
            Est une technique d'interprétation des réseaux de neurones visant à réduire le bruit dans les cartes de saillance et à produire des visualisations plus stables et compréhensibles.    
            Cette méthode améliore les cartes de saillance en moyennant les gradients calculés sur plusieurs versions bruitées de l'entrée d'origine.  
  
            Dans le cas de notre modèle, voici ce qu’il ressort de l’interprétation selon cette méthode :
            """)   
            st.image("img_inter_3.png", caption="SmoothGrad", width=400)
            
            st.write("""
            **Remarques:**  
            Ces 3 algorithmes d’interprétation des images mettent en évidence que le modèle prend en considération principalement les contours des objets.  
            Cependant, cela fait aussi ressortir que pour toutes les catégories livres, revue, DVD, les objets ont les mêmes formes et les détails sont difficilement détectables, d’où l’importance du texte pour catégoriser l’objet.  
            C’est une limite du modèle et de notre stratégie de Deep Learning.
            """)        
                
elif page == "Conclusion":
    st.markdown("<h1 class='red-title center-title'>Conclusion</h1>", unsafe_allow_html=True)

    # Texte principal
    texte_principal = """
    Les choix effectués tout au long du projet ont été guidés par des objectifs de performance et de robustesse.  
      
    Les techniques de réduction de dimension et le choix des algorithmes ont permis d'optimiser les résultats malgré la nature complexe et non structurée des données.  
      
    Le modèle hybride alliant **EfficientNetB0 et LSTM** s'est avéré le plus adapté pour la classification des produits e-commerce de Rakuten.      
    
    **En conclusion**:   
    Objectif atteint! **Score final: F1-pondéré de 0.96** (Benchmark: 0.81 et meilleur score "Challenge: 0.92)  
    Très bonne prédiction des catégories. (avec le jeu d'entrainement fournit)
    
    """

    # Texte pour la section Limite du modèle 
    limite_modele = """
    **Limites du modèle**  
    
    - La traduction du texte a peu d'impact sur les performances du modèle, car le français est la langue la plus présente (~60%).  
    
    - 5 catégories où le français représente moins de la moitié des textes.  
    
    - La précision de la catégorisation dépend de la qualité de la description textuelle.  
    
    - Pour les catégories de livres, le modèle a des difficultés à discerner les détails fins des images et s'appuie principalement sur la description textuelle.
    
    - Le modèle est moins efficace lorsqu'une catégorie comporte beaucoup d'éléments très différents (Jeux, Jouets, Puericulture, accessoires décorations,...) - gris et jaune 
    
    - Le modèle est moins efficace lorsqu'il y a très peu de différences entre les catégories (Lives, Magasines,...) - rouge 

    """


    # Texte pour la section Préconisations et Améliorations
    preconisations_ameliorations = """
    **Préconisations et Améliorations**

    - Traduction en Français :  
    Réduire le temps de traitement et le coût de la traduction en l'appliquant uniquement sur les catégories n’ayant pas le français comme langue majoritaire, pour améliorer l'efficacité de la modélisation.

    - Rééquilibrage des Données :    
    Rééquilibrer les données durant le pré-processing, afin que chaque catégorie soit représentée de manière équitable.

    - Augmentation des Données :    
    Utilisation des techniques d'augmentation des données pour enrichir le jeu de données existant. Transformations des images (rotation, recadrage, ajout de bruit...).

    - Investigation sur de Nouveaux Modèles de Deep Learning :    
    Explorer de nouveaux modèles de Deep Learning mieux adaptés à notre jeu de données (moins gourmands en ressources, en temps et plus efficace).  
    
    """
     
    # Affichage dans Streamlit
    st.write(texte_principal) 

    tabs = st.tabs(["Limites du modèle", "Préconisations et Améliorations"])

    with tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(limite_modele)

        with col2:
            img_tabs = st.tabs(["Français 1", "Fraçais 2", "Français 3", "Précison", "Images", "Prédictions"])

            with img_tabs[0]:
                st.image("pie_1.png", caption="Le français en rouge", width=600)
            with img_tabs[1]:
                st.image("pie_2.png", caption="Le français en rouge", width=600)
            with img_tabs[2]:
                st.image("pie_3.png", caption="Le français en rouge", width=600)
            with img_tabs[3]:
                st.image("exemple_text.png", caption="Exemple de désignation", width=600)
                st.write("""
                        On remarque qu'ici que dans la variable "designation" la description de l'objet n'est pas détaillée!  
                        Le modèle n'arrivera pas à définir qu'il s'agit d'un livre en se basant uniquement sur le texte.  
                        Et même s'il y arriverait, il n'arrivera pas à choisir la bonne catégorie parmi les 4 qui définissent des livres ou magazines.  
                        
                        Donc le score ne dépend pas uniquement du modèle et de son entraînement, mais aussi de la qualité de la description des objets.  
                        
                        """)
            with img_tabs[4]:
                st.image("img_inter_2.png", caption="Cartes de saillance", width=400)
            with img_tabs[5]:
                st.image("mtx_eff_LSTM_rem.png", caption="Matrice de confusion EfficientNetB0-LSTM", width=600)                    

    with tabs[1]:
        st.markdown(preconisations_ameliorations)

elif page == "Démo":
    st.markdown("<h1 class='red-title center-title'>Classification de produit</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(['Jeu de données "Test" Rakuten', "Autres données"])

    with tab1:
        st.header('Jeu de données "Test" Rakuten')

        # Code spécifique au tab "Jeu de données Rakuten"
        st.write("""
        Ce tableau contient les 20 premières lignes du DataFrame final et affiche les codes des catégories d'objets prédits, du jeu de données "Test" Rakuten , calculés par le "Modèle hybride EfficientNetB0-LSTM".
        """)

        # Charger et afficher les 20 premières lignes du dataframe
        df_prediction_final = load_data("df_prediction_final.csv")
        
        st.data_editor(
            df_prediction_final.head(20),
            column_config={
                "Code catégorie prédite": st.column_config.NumberColumn(
                    format="%d",
                ),
                "index": st.column_config.NumberColumn(
                    format="%d",  # Utilisation de %d pour enlever la virgule des milliers dans l'index
                )
            },
            hide_index=True,
        )
               
        # st.dataframe(df_prediction_final.head(20))

    with tab2:
        st.header("Autres données")

        st.write("Il est possible de faire une prédiction de code de catégorie, pour tout objet unitaire, sur la base d'une descripion (texte) et d'une image de cet objet, et à condition qu'il respecte les **limitations de l'outil*.")

        col1, col2 = st.columns(2)

        with col1:
            # Charger les ressources pour la démo
            def load_demo_resources():
                try:
                    tokenizer_path_d = "tokenizer.pkl"
                    le_path_d = "label_encoder.pkl"
                    model_path_d = "model_EfficientNetB0-LSTM.keras"
                    categories_csv_path_d = "categories_prdtypecode.csv"

                    with open(tokenizer_path_d, 'rb') as handle:
                        tokenizer_demo = pickle.load(handle)

                    with open(le_path_d, 'rb') as handle:
                        le_demo = pickle.load(handle)

                    model_demo = tf.keras.models.load_model(model_path_d)
                    categorie_demo = pd.read_csv(categories_csv_path_d, sep=';')

                    return tokenizer_demo, le_demo, model_demo, categorie_demo

                except Exception as e:
                    st.error(f"Erreur lors du chargement des ressources: {e}")
                    return None, None, None, None

            tokenizer_demo, le_demo, model_demo, categorie_demo = load_demo_resources()

            # Prétraitement du texte
            def preprocess_text(tokenizer_demo, text, max_len=100):
                sequences = tokenizer_demo.texts_to_sequences([text])
                text_data = pad_sequences(sequences, maxlen=max_len)
                return text_data

            # Prétraitement de l'image
            def load_and_preprocess_image(img_path_d, image_size=128):
                img = load_img(img_path_d, target_size=(image_size, image_size))
                img = img_to_array(img)
                img = tf.keras.applications.efficientnet.preprocess_input(img)
                return img

            if tokenizer_demo is None or le_demo is None or model_demo is None or categorie_demo is None:
                st.error("Les ressources nécessaires n'ont pas pu être chargées.")
            else:
                # Entrée du texte de description du produit
                description_text = st.text_input("Entrez la description du produit :")

                # Téléchargement de l'image du produit
                uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

                if st.button("Prédire"):
                    if description_text and uploaded_file:
                        # Sauvegarder l'image téléchargée temporairement
                        image_path_d = os.path.join("temp_dir", uploaded_file.name)
                        os.makedirs("temp_dir", exist_ok=True)  # Créer le dossier temporairement s'il n'existe pas
                        with open(image_path_d, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Prétraiter les nouvelles données de texte et d'image
                        text_data = preprocess_text(tokenizer_demo, description_text)
                        image_data = np.expand_dims(load_and_preprocess_image(image_path_d), axis=0)

                        # Faire des prédictions
                        predictions = model_demo.predict([text_data, image_data])
                        predicted_class = np.argmax(predictions, axis=1)[0]
                        predicted_label = le_demo.inverse_transform([predicted_class])[0]

                        # Trouver la désignation de la catégorie prédite
                        category_row = categorie_demo[categorie_demo['code type'] == predicted_label]
                        category_name = category_row['désignation de catégorie'].values[0] if not category_row.empty else "Inconnue"

                        # Afficher l'image sélectionnée avec ses désignations, nom et prédiction
                        img = Image.open(image_path_d)
                        st.image(img, caption=f"Description: {description_text}\n\nNom: {os.path.basename(image_path_d)}\n\nPrédiction: {predicted_label} - {category_name}", width=300)

                        # Afficher la catégorie prédite
                        st.write(f"Catégorie prédite: {predicted_label} - {category_name}")

                        # Supprimer l'image temporaire
                        os.remove(image_path_d)
                    else:
                        st.write("Veuillez entrer une description et télécharger une image.")

        with col2:
            st.write("""
                     **Voici des liens pour accèder à d'autres objets:**
                     """)
            
            # Ajouter les logos avec hyperliens
            st.markdown("""
            <div style='display: flex; justify-content: space-around;'>
                <a href='https://fr.shopping.rakuten.com' target='_blank'>
                    <img src='https://fr.shopping.rakuten.com/visuels/0_content_square/autres/rakuten-logo6.svg' alt='Rakuten France' style='width: 100px;'>
                </a>
                <a href='https://www.amazon.fr' target='_blank'>
                    <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Amazon_logo.svg/2560px-Amazon_logo.svg.png' alt='Amazon France' style='width: 100px;'>
                </a>
                <a href='https://www.ikea.com/fr/fr/' target='_blank'>
                    <img src='https://www.ikea.com/fr/fr/static/ikea-logo.3ee105eef6b5939c1269.svg' alt='Ikea France' style='width: 100px;'>
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.write("")
                     
            st.write("""
                    **Limitations de l'outil** (*)  
                    Cet outil a été conçu pour répondre à la demande du « Challenge Rakuten ».  
                    Par conséquent les catégories prédites sont celles définies par ce dernier dans le jeu de données « Y_train ».  
                    Tout objet ne faisant pas partie de celles-ci, ne pourra pas être classifié correctement.   
                    L’algorithme indiquera tout de même, la catégorie qui lui semble la plus proche.  

                    **Rappel:** (Les catégories)
                    """)
            
            # Charger et afficher le dataframe catégories
            df_categorie = pd.read_csv("categories_prdtypecode.csv", sep=';')
            
            st.data_editor(
                df_categorie,
                column_config={
                    "code type": st.column_config.NumberColumn(
                        format="%d",
                    )
                },
                hide_index=True,
            )

# Ajoutez des espaces vides pour pousser les noms en bas
for i in range(2):
    st.sidebar.text("")

# Utilisez le Markdown avec une classe CSS personnalisée pour les petits titres
st.sidebar.markdown('''
<div class="reduced-spacing">
    <p class="small-title">Auteurs:</p>
    <p><a href="https://www.linkedin.com/in/labordecaroline/" target="_blank">Caroline LABORDE</a></p>
    <p><a href="https://www.linkedin.com/in/soulaiman-cheddadi/" target="_blank">Soulaiman CHEDDADI</a></p>
    <p><a href="https://www.linkedin.com/in/jean-gabrielmarchetta/" target="_blank">Jean-Gabriel MARCHETTA</a></p>
    <p class="small-title">Mentor:</p>
    <p>Eliott DOUIEB</p>
</div>
''', unsafe_allow_html=True)

# Ajouter une ligne vide
st.sidebar.text("")
# Ajouter une ligne vide
st.sidebar.text("")

# Ajouter le logo de DataScientest dans la barre latérale avec un lien hypertexte
st.sidebar.markdown(f"""
<a href="https://datascientest.com/" target="_blank">
    <img src="https://datascientest.com/wp-content/uploads/2022/03/logo-2021.png" style="width: 100%;">
</a>
""", unsafe_allow_html=True)
st.sidebar.text("Datascientist - Bootcamp mars 2024")
