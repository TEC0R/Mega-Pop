import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit_survey as ss
import random
import base64
import os



#------------------------------------------------------------------------------------
#                               Data et config
#------------------------------------------------------------------------------------

df_film = pd.read_csv("film.csv")
df_contributeur = pd.read_csv("contributeur.csv")
st.set_page_config(layout="wide")

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


components.html(load_css("style.css"),height=0)
#------------------------------------------------------------------------------------
#                               Algo et Nettoyage
#------------------------------------------------------------------------------------

def load_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def image(image_base64, picture_texte,width=100,height=100):
    return f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{image_base64}" alt={picture_texte} width={width}% height={height} padding= 0>
        </div>
    """

def rating(note):
    return "⭐️" if 0 < note <= 2 else "⭐️⭐️" if 2 < note <= 4 else "⭐️⭐️⭐️" if 4 < note <= 6 else "⭐️⭐️⭐️⭐️" if 6 < note <= 8 else "⭐️⭐️⭐️⭐️⭐️"

def runtime(minutes):
    heures = minutes // 60
    minutes_restantes = minutes % 60
    return f"{heures}h {minutes_restantes}m"

df_contributeur['birthday'] = pd.to_datetime(df_contributeur['birthday']).dt.strftime('%d-%m-%Y')

df_film['genres'] = df_film['genres'].str.replace(",", ", ")

liste_contributeur = df_contributeur[pd.notna(df_contributeur['biography'])] \
                        .drop_duplicates(subset=['title']) \
                        .groupby('primaryName') \
                        .filter(lambda x: len(x) >= 3) \
                        .sort_values(by='popularity', ascending=False) \
                        .groupby('primaryName').agg({
                            'title': lambda x: ', '.join(x[:3]),
                            'popularity': 'first',
                            'biography': 'first',
                            'profile_path': 'first',
                            'birthday' : 'first',
                            'tconst': 'first'
                        }).reset_index()

#------------------------------------------------------------------------------------
#                               ML Reco
#------------------------------------------------------------------------------------

df_key_word = df_film.copy()
df_key_word_contributeurs = df_contributeur.copy()
df_key_word = df_key_word.reset_index(drop = True)

df_key_word['genres'] = df_key_word['genres'].apply(lambda x : x.replace(",Drama", ""))
df_actor = df_key_word_contributeurs[['tconst','primaryName']]
df_actor['primaryName'] = df_actor['primaryName'].str.replace(" ", "")
df_actor_pt = pd.pivot_table(data = df_actor, index = "tconst", values= "primaryName", aggfunc=lambda chaine : " ".join(chaine)).reset_index()

df_key_word = df_key_word[['tmdb_id','tconst','title','genres','mot_cle','tagline','overview','startYear']]
df_key_word_final = pd.merge(df_actor_pt, df_key_word, on = "tconst", how = "right")

df_key_word_final['mot_cle'] = df_key_word_final['mot_cle'].str.replace("-"," ").str.replace("..."," ").str.replace("."," ").str.replace(","," ").str.replace(";"," ")
df_key_word_final['genres'] = df_key_word_final['genres'].str.replace("..."," ").str.replace("."," ").str.replace(", "," ").str.replace(";"," ").str.replace(","," ")

data = df_key_word_final.reset_index(drop=True)
data['features'] =(data['genres'] + ' ' + data['mot_cle']+ ' ' +data['primaryName'] + ' ' + str(data['startYear'])).str.lower()

count = CountVectorizer()
count_matrix = count.fit_transform(data['features'].fillna(""))

cosine_sim = cosine_similarity(count_matrix, count_matrix)
indices = pd.Series(data['title'])

def recommend(title, cosine_sim = cosine_sim):
    recommended_movies = []
    df_film_reco = []
    idx = indices[indices.str.contains(title)].index[0]  # to get the index of the movie title matching the input movie
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)   # similarity scores in descending order
    top_5_indices = list(score_series.iloc[1:6].index)   # to get the indices of top 6 most similar movies
    # [1:6] to exclude 0 (index 0 is the input movie itself)
    for i in top_5_indices:  # pour ajouter les titres des 5 films les plus similaires à la liste des films recommandés
        recommended_movies.append(list(data['title'])[i])
    
    for reco in recommended_movies:
      df_film_reco.append(df_film[df_film['title'].str.contains(reco)])
    

    return pd.concat(df_film_reco)

#------------------------------------------------------------------------------------
#                               ML Reco Inspi
#------------------------------------------------------------------------------------

if 'dico' not in st.session_state:
    st.session_state.dico = {}

def recommend_inspi(dico_keyword = st.session_state.dico, n_suggestions=50):
    year = []
    for i in range(int(dico_keyword['year'][0]),int(dico_keyword['year'][1]) + 1):
      year.append(str(i))
    dico_keyword['year'] = year

    keywords = " ".join(map(str, dico_keyword.values())).replace("[","").replace("]","").replace("'","").replace("(","").replace(")","").replace(",","").replace("  "," ").lower()
    
    keywords_vector = count.transform([keywords])

    similarities = cosine_similarity(keywords_vector, count_matrix)

    closest_indices = similarities[0].argsort()[-n_suggestions:][::-1]

    recommended_movies = []
    df_film_reco = []
    
    for i, index in enumerate(closest_indices):
        recommended_movies.append(data.loc[index, 'title'])
    for reco in recommended_movies:
        df_film_reco.append(df_film[df_film['title'].str.contains(reco)])
    concat = pd.concat(df_film_reco)
    return concat[(int(dico_keyword['year'][0]) <= concat["startYear"]) & (concat["startYear"] <= int(dico_keyword['year'][-1]))]


#------------------------------------------------------------------------------------
#                               logo et barre de recherche
#------------------------------------------------------------------------------------



image_path = os.path.join("picture", f"{random.randint(1, 12)}.png")
image_base64 = load_image_as_base64(image_path)
st.markdown(image(image_base64,"banniere",100,"auto"), unsafe_allow_html=True)

vide, search,vide = st.columns([1,3,1])


with search:
    st.title(" ")
    film, acteurs, inspiration = st.tabs(['Recherche de films','Recherche des acteurs/actrice','Recherche d\'inspiration'])
    
    with film:
        search_film = st.selectbox(
            "",
            index=None,
            placeholder="Recherche de films ...",
            options=df_film.sort_values(by='popularity',ascending=False)['title'].tolist(),
            key="movie_selectbox"
        )

    with acteurs:
        search_actor = st.selectbox(
            "",
            index=None,
            placeholder="Recherche des acteurs/actrice ...",
            options=liste_contributeur.sort_values(by='popularity',ascending=False)['primaryName'].unique().tolist(),
            key="actor_selectbox"
        )
    with inspiration:
        
        form_submitted = [False]

        def submit_callback():
            form_submitted[0] = True
            st.success("Your responses have been recorded. Thank you!")
            
            
        questionnaire = ss.StreamlitSurvey("Besoin d'inspiration ?").pages(9, on_submit= submit_callback)
        questionnaire.submit_button = lambda questionnaire: st.button("Soumettre", type="secondary")

        with questionnaire:

            if questionnaire.current == 0:
                genre = st.multiselect(
                    "Quel genre de film, souhaitez-vous ?",
                    list(df_film['genres'].str.split(',').explode('genres').unique())
                )
                if genre:
                    st.session_state.dico['genre'] = genre

            if questionnaire.current == 1:
                pouvoir = st.selectbox(
                    "Quel super-pouvoir aimeriez-vous avoir dans la vraie vie ?",
                    ["Voler", "Téléportation", "Invisibilité", "Télépathie", "Super force", "Contrôle du temps", "Autre / Aucun"]

                )
                
                if pouvoir:
                    st.session_state.dico['pouvoir'] = pouvoir

            if questionnaire.current == 2:
                animal = st.selectbox(
                    "Quel serait votre animal de compagnie idéal, inspiré des films ?",
                    ["Falkor (L'Histoire sans fin)", "Dobby (Harry Potter)", "Toothless (Dragons)", "Garfield", "Scooby-Doo", "Simba (Le Roi Lion)", "Autre / Aucun"]
                )
                
                if animal:
                    st.session_state.dico['animal'] = animal

            if questionnaire.current == 3:
                replique = st.selectbox(
                    "Quelle est votre réplique de film préférée ?",
                    ["Que la Force soit avec toi.", "Je suis le roi du monde !", "Je vais faire une offre qu'il ne pourra pas refuser.", "Hasta la vista, baby.", "Tu peux pas test.", "Houston, on a un problème.", "Autre / Je ne sais pas"]
                )
                
                if replique:
                    st.session_state.dico['replique'] = replique

            if questionnaire.current == 4:
                theme = st.selectbox(
                    "Si vous deviez organiser une soirée cinéma à thème, quel serait le thème principal ?",
                    ["Horreur", "Années 80", "Super-héros", "Musicaux", "Aventure", "Noël", "Autre / Je ne sais pas"]
                )
                
                if theme:
                    st.session_state.dico['theme'] = theme

            if questionnaire.current == 5:
                personnage = st.selectbox(
                    "Si vous deviez être un personnage de film, lequel seriez-vous ?",
                    ["James Bond", "Hermione Granger", "Deadpool", "Bridget Jones", "Yoda", "Jack Sparrow", "Autre / Je ne sais pas"]
                )
                
                if personnage:
                    st.session_state.dico['personnage'] = personnage

            if questionnaire.current == 6:
                role = st.selectbox(
                    "Si vous deviez jouer dans un film, quel serait votre rôle idéal ?",
                    ["Le héros courageux", "Le méchant machiavélique", "Le sidekick comique", "Le génie excentrique", "Le mentor sage", "Le personnage mystérieux", "Autre / Je ne sais pas"]
                )
                
                if role:
                    st.session_state.dico['role'] = role

            if questionnaire.current == 7:
                film = st.selectbox(
                    "Si vous deviez voyager dans le temps pour assister à la première de n'importe quel film, lequel choisiriez-vous ?",
                     ["Titanic", "Star Wars: Un nouvel espoir", "Retour vers le futur", "Le Parrain", "Avengers: Endgame", "Le Magicien d'Oz", "Autre / Je ne sais pas"]
                )
                
                if film:
                    st.session_state.dico['film'] = film

            if questionnaire.current == 8:
                year = st.select_slider(
                    "Choisissez une periode de sortie",
                    options=list(df_film['startYear'].sort_values().unique()),
                    value=(df_film['startYear'].min(), df_film['startYear'].max())
                )
                if year:
                    st.session_state.dico['year'] = year

            total_questions = 9
            progression_percentage = int((len(st.session_state.dico.keys()) / total_questions) * 100)
            st.progress(progression_percentage)


#------------------------------------------------------------------------------------
#                               Page Accueuil
#------------------------------------------------------------------------------------

def page_accueil():

    filtre_film_annee = df_film[(df_film['startYear'] == 2024) & (pd.notna(df_film['poster_path']))]
    indices_aleatoires = random.sample(range(filtre_film_annee.shape[0]), 3)
    film_1 = indices_aleatoires[0]
    film_2 = indices_aleatoires[1]
    film_3 = indices_aleatoires[2]

    st.header("Les films du moment")
    st.header("") 

    titre1, titre2, titre3 = st.columns(3)
    with titre1:
        st.subheader(f"{filtre_film_annee['title'].iloc[film_1]} ({filtre_film_annee['startYear'].iloc[film_1]})")
    with titre2:
        st.subheader(f"{filtre_film_annee['title'].iloc[film_2]} ({filtre_film_annee['startYear'].iloc[film_2]})")
    with titre3:
        st.subheader(f"{filtre_film_annee['title'].iloc[film_3]} ({filtre_film_annee['startYear'].iloc[film_3]})")

    st.title(" ")
    film1, film2, film3 = st.columns(3)

    with film1:
        poster_path, infos = st.columns([1.5,3])
        poster_path.image(filtre_film_annee['poster_path'].iloc[film_1], width=140)

        with infos:
            st.write(filtre_film_annee['genres'].iloc[film_1])
            st.write("Intervenants :")
            st.write(filtre_film_annee['contributeur'].iloc[film_1])
            st.write(f"Note : {rating(filtre_film_annee['averageRating'].iloc[film_1])}")
        st.write(filtre_film_annee['overview'].iloc[film_1])
    with film2:
        poster_path, infos = st.columns([1.5,3])
        poster_path.image(filtre_film_annee['poster_path'].iloc[film_2], width=140)

        with infos:
            st.write(filtre_film_annee['genres'].iloc[film_2])
            st.write("Intervenants :")
            st.write(filtre_film_annee['contributeur'].iloc[film_2])
            st.write(f"Note : {rating(filtre_film_annee['averageRating'].iloc[film_2])}")
        st.write(filtre_film_annee['overview'].iloc[film_2])
    with film3:
        poster_path, infos = st.columns([1.5,3])
        poster_path.image(filtre_film_annee['poster_path'].iloc[film_3], width=140)

        with infos:
            st.write(filtre_film_annee['genres'].iloc[film_3])
            st.write("Intervenants :")
            st.write(filtre_film_annee['contributeur'].iloc[film_3])
            st.write(f"Note : {rating(filtre_film_annee['averageRating'].iloc[film_3])}")
        st.write(filtre_film_annee['overview'].iloc[film_3])

    st.header("")    
    st.header("Les Acteurs du moment")
    st.header("") 
    
    filtre_contributeur_liste = liste_contributeur[liste_contributeur['popularity'] >= liste_contributeur['popularity'].quantile(0.50)]
    indices_aleatoires = random.sample(range(filtre_contributeur_liste.shape[0]), 3)
    contri_1 = indices_aleatoires[0]
    contri_2 = indices_aleatoires[1]
    contri_3 = indices_aleatoires[2]


    contri1, contri2, contri3 = st.columns(3)

    with contri1:
        photo, infos_contri = st.columns([1.5,3])
        photo.image(filtre_contributeur_liste['profile_path'].iloc[contri_1], width=140)
        with infos_contri:
            st.subheader(filtre_contributeur_liste['primaryName'].iloc[contri_1])
            st.write(f"Anniversaire : {filtre_contributeur_liste['birthday'].iloc[contri_1]}")
            st.write(f"Filmographie : {filtre_contributeur_liste['title'].iloc[contri_1]}")
    with contri2:
        photo, infos_contri = st.columns([1.5,3])
        photo.image(filtre_contributeur_liste['profile_path'].iloc[contri_2], width=140)
        with infos_contri:
            st.subheader(filtre_contributeur_liste['primaryName'].iloc[contri_2])
            st.write(f"Anniversaire : {filtre_contributeur_liste['birthday'].iloc[contri_2]}")
            st.write(f"Filmographie : {filtre_contributeur_liste['title'].iloc[contri_2]}")
    with contri3:
        photo, infos_contri = st.columns([1.5,3])
        photo.image(filtre_contributeur_liste['profile_path'].iloc[contri_3], width=140)
        with infos_contri:
            st.subheader(filtre_contributeur_liste['primaryName'].iloc[contri_3])
            st.write(f"Anniversaire : {filtre_contributeur_liste['birthday'].iloc[contri_3]}")
            st.write(f"Filmographie : {filtre_contributeur_liste['title'].iloc[contri_3]}")


#------------------------------------------------------------------------------------
#                               Page Film
#------------------------------------------------------------------------------------

def page_film():
    filtre_film_titre = df_film[df_film['title'] == search_film]

    st.header(f"{filtre_film_titre['title'].iloc[0]} ({filtre_film_titre['startYear'].iloc[0]})")
    vide, poster, infos, contri = st.columns([0.5,2,3,3])
    try:
        poster.image(filtre_film_titre['poster_path'].iloc[0], width=250)
    except:
        poster.image('https://img.freepik.com/premium-vector/poster-with-inscription-error-404_600765-3956.jpg?w=360', width=250)

    with infos: 
        info2, bande_annonce = st.tabs(['Informations','Bande-annonce'])
        with info2:
            st.write(f"Genre : {filtre_film_titre['genres'].iloc[0]}")
            st.write(f"Durée : {runtime(filtre_film_titre['runtimeMinutes'].iloc[0])}")
            st.write(f"Note : {rating(filtre_film_titre['averageRating'].iloc[0])}")
            st.write("Synopsis :")
            st.write(filtre_film_titre['overview'].iloc[0])
        with bande_annonce:
            try:
                st.video(filtre_film_titre['bande_annonce'].iloc[0])
            except:
                st.image('https://images.wondershare.com/recoverit/article/2019/11/common-video-errors-01.jpg', width=400)
    with contri:
        st.header("Intervenant")
        filtre_contributeur_liste = df_contributeur[df_contributeur['tconst'] == filtre_film_titre['tconst'].iloc[0]].sort_values(by='popularity', ascending=False)  

        vide,photo, infos_contri,vide = st.columns([1,2,3,1])
        photo.image(filtre_contributeur_liste['profile_path'].iloc[0], width=90)
        with infos_contri:
            st.subheader(filtre_contributeur_liste['primaryName'].iloc[0])
            st.write(f"Anniversaire : {filtre_contributeur_liste['birthday'].iloc[0]}")
        vide,infos_contri,photo,vide = st.columns([1,3,2,1])
        photo.image(filtre_contributeur_liste['profile_path'].iloc[1], width=90)
        with infos_contri:
            st.subheader(filtre_contributeur_liste['primaryName'].iloc[1])
            st.write(f"Anniversaire : {filtre_contributeur_liste['birthday'].iloc[1]}")
        vide,photo, infos_contri,vide = st.columns([1,2,3,1])
        photo.image(filtre_contributeur_liste['profile_path'].iloc[2], width=90)
        with infos_contri:
            st.subheader(filtre_contributeur_liste['primaryName'].iloc[2])
            st.write(f"Anniversaire : {filtre_contributeur_liste['birthday'].iloc[2]}")
    
    filtre_film_reco = recommend(search_film)

    st.title(" ")
    st.title(" ")
    st.header("Recommandations")
    st.header("") 

    titre_reco1, titre_reco2, titre_reco3 = st.columns(3)
    with titre_reco1:
        st.subheader(f"{filtre_film_reco['title'].iloc[0]} ({filtre_film_reco['startYear'].iloc[0]})")
    with titre_reco2:
        st.subheader(f"{filtre_film_reco['title'].iloc[1]} ({filtre_film_reco['startYear'].iloc[1]})")
    with titre_reco3:
        st.subheader(f"{filtre_film_reco['title'].iloc[2]} ({filtre_film_reco['startYear'].iloc[2]})")

    film_reco1, film_reco2, film_reco3 = st.columns(3)

    with film_reco1:
        poster_path, infos = st.columns([1.5,3])
        poster_path.image(filtre_film_reco['poster_path'].iloc[0], width=140)

        with infos:
            st.write(filtre_film_reco['genres'].iloc[0])
            st.write("Intervenants :")
            st.write(filtre_film_reco['contributeur'].iloc[0])
            st.write(f"Note : {rating(filtre_film_reco['averageRating'].iloc[0])}")
        st.write(filtre_film_reco['overview'].iloc[0])
    with film_reco2:
        poster_path, infos = st.columns([1.5,3])
        poster_path.image(filtre_film_reco['poster_path'].iloc[1], width=140)

        with infos:
            st.write(filtre_film_reco['genres'].iloc[1])
            st.write("Intervenants :")
            st.write(filtre_film_reco['contributeur'].iloc[1])
            st.write(f"Note : {rating(filtre_film_reco['averageRating'].iloc[1])}")
        st.write(filtre_film_reco['overview'].iloc[1])
    with film_reco3:
        poster_path, infos = st.columns([1.5,3])
        poster_path.image(filtre_film_reco['poster_path'].iloc[2], width=140)

        with infos:
            st.write(filtre_film_reco['genres'].iloc[2])
            st.write("Intervenants :")
            st.write(filtre_film_reco['contributeur'].iloc[2])
            st.write(f"Note : {rating(filtre_film_reco['averageRating'].iloc[0])}")
        st.write(filtre_film_reco['overview'].iloc[2])

#------------------------------------------------------------------------------------
#                               Page Contributeur
#------------------------------------------------------------------------------------

def page_contributeur():
    st.title(" ")
    filtre_contri_name = df_contributeur[df_contributeur['primaryName'] == search_actor]

    photo, infos = st.columns([2,3])
    photo.image(filtre_contri_name['profile_path'].iloc[0], width=500)

    with infos: 
        st.header(filtre_contri_name['primaryName'].iloc[0])
        st.write(f"Anniversaire : {filtre_contri_name['birthday'].iloc[0]}")
        st.write(f"Filmographie : {filtre_contri_name['title'].iloc[0]}")
        st.write(filtre_contri_name['biography'].iloc[0])

    filtre_film_contri= df_film[df_film['tconst'].isin(filtre_contri_name['tconst'])].sort_values(by='startYear', ascending=False)

    st.title(" ")
    st.title(" ")
    titre1, titre2, titre3 = st.columns(3)
    with titre1:
        st.subheader(f"{filtre_film_contri['title'].iloc[0]} ({filtre_film_contri['startYear'].iloc[0]})")
    with titre2:
        st.subheader(f"{filtre_film_contri['title'].iloc[1]} ({filtre_film_contri['startYear'].iloc[1]})")
    with titre3:
        st.subheader(f"{filtre_film_contri['title'].iloc[2]} ({filtre_film_contri['startYear'].iloc[2]})")
    film1, film2, film3 = st.columns(3)

    with film1:
        poster_path, infos = st.columns([1.5,3])
        poster_path.image(filtre_film_contri['poster_path'].iloc[0], width=140)

        with infos:
            st.write(filtre_film_contri['genres'].iloc[0])
            st.write("Intervenants :")
            st.write(filtre_film_contri['contributeur'].iloc[0])
            st.write(f"Note : {rating(filtre_film_contri['averageRating'].iloc[0])}")
        st.write(filtre_film_contri['overview'].iloc[0])
    with film2:
        poster_path, infos = st.columns([1.5,3])
        poster_path.image(filtre_film_contri['poster_path'].iloc[1], width=140)

        with infos:
            st.write(filtre_film_contri['genres'].iloc[1])
            st.write("Intervenants :")
            st.write(filtre_film_contri['contributeur'].iloc[1])
            st.write(f"Note : {rating(filtre_film_contri['averageRating'].iloc[1])}")
        st.write(filtre_film_contri['overview'].iloc[1])
    with film3:
        poster_path, infos = st.columns([1.5,3])
        poster_path.image(filtre_film_contri['poster_path'].iloc[2], width=140)

        with infos:
            st.write(filtre_film_contri['genres'].iloc[2])
            st.write("Intervenants :")
            st.write(filtre_film_contri['contributeur'].iloc[2])
            st.write(f"Note : {rating(filtre_film_contri['averageRating'].iloc[2])}")
        st.write(filtre_film_contri['overview'].iloc[2])
                    
#------------------------------------------------------------------------------------
#                               Page Inspiration
#------------------------------------------------------------------------------------

def inspi():
    filtre_film_inspi = recommend_inspi().drop_duplicates('title')
    for nb_inspi in range(filtre_film_inspi.shape[0]):
        if (nb_inspi % 2) == 0:
            st.header(f"{filtre_film_inspi['title'].iloc[nb_inspi]} ({filtre_film_inspi['startYear'].iloc[nb_inspi]})")
            vide, poster, infos, contri = st.columns([0.5, 2, 3, 3])
            try:
                poster.image(filtre_film_inspi['poster_path'].iloc[nb_inspi], width=250)
            except:
                poster.image('https://img.freepik.com/premium-vector/poster-with-inscription-error-404_600765-3956.jpg?w=360', width=250)

            with infos:
                info2, bande_annonce = st.tabs(['Informations', 'Bande-annonce'])
                with info2:
                    st.write(f"Genre : {filtre_film_inspi['genres'].iloc[nb_inspi]}")
                    st.write(f"Durée : {runtime(filtre_film_inspi['runtimeMinutes'].iloc[nb_inspi])}")
                    st.write(f"Note : {rating(filtre_film_inspi['averageRating'].iloc[nb_inspi])}")
                    st.write("Synopsis :")
                    st.write(filtre_film_inspi['overview'].iloc[nb_inspi])
                with bande_annonce:
                    try:
                        st.video(filtre_film_inspi['bande_annonce'].iloc[nb_inspi])
                    except:
                        st.image('https://images.wondershare.com/recoverit/article/2019/11/common-video-errors-01.jpg', width=400)
                    st.header(" ")      

        else:
            st.header(f"{filtre_film_inspi['title'].iloc[nb_inspi]} ({filtre_film_inspi['startYear'].iloc[nb_inspi]})")
            vide, infos, poster, vide = st.columns([3,3,2, 0.5])

            with infos:
                info2, bande_annonce = st.tabs(['Informations', 'Bande-annonce'])
                with info2:
                    st.write(f"Genre : {filtre_film_inspi['genres'].iloc[nb_inspi]}")
                    st.write(f"Durée : {runtime(filtre_film_inspi['runtimeMinutes'].iloc[nb_inspi])}")
                    st.write(f"Note : {rating(filtre_film_inspi['averageRating'].iloc[nb_inspi])}")
                    st.write("Synopsis :")
                    st.write(filtre_film_inspi['overview'].iloc[nb_inspi])
                with bande_annonce:
                    try:
                        st.video(filtre_film_inspi['bande_annonce'].iloc[nb_inspi])
                    except:
                        st.image('https://images.wondershare.com/recoverit/article/2019/11/common-video-errors-01.jpg', width=400)
            try:
                poster.image(filtre_film_inspi['poster_path'].iloc[nb_inspi], width=250)
            except:
                poster.image('https://img.freepik.com/premium-vector/poster-with-inscription-error-404_600765-3956.jpg?w=360', width=250)

            st.header(" ")   



#------------------------------------------------------------------------------------
#                               Navigation
#------------------------------------------------------------------------------------

if not search_film and not search_actor and form_submitted[0] == False:
    page_selection = "Accueil"
elif search_film:
    page_selection = "Film"
elif search_actor:
    page_selection = "Contributeur"
elif form_submitted[0] == True:
    page_selection = "Inspi"


if page_selection == "Accueil":
    page_accueil()
elif page_selection == "Film":
    page_film()
elif page_selection == "Contributeur":
    page_contributeur()
elif page_selection == "Inspi":
    inspi()
