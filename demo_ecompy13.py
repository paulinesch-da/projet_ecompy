import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import streamlit as st

import streamlit as st
import os

# Side bar components .........................
st.sidebar.title("Projet e-ComPy")

st.sidebar.image("DataScientest logo.png", width=160)


#st.sidebar.header("Menu")
# Main menu status
status = st.sidebar.radio("", ("Accueil", "Dataset", "Data visualisation",
                               "Système de recommandation", "Preprocessing", "Modélisation", "Conclusion")
                          )


# st.sidebar.info("Projet réalisé dans le cadre de la formation **Data Analyst** de DataScientest.com \
#                Promotion **Bootcamp** octobre **2021**.")

if status == 'Accueil':
    st.title("Etude du comportement d’achat : suivi des utilisateurs sur un site de e-commerce")
    st.image("main_photo.jpg")
    st.markdown(
        "> Projet réalisé dans le cadre de la formation **Data Analyst** de DataScientest.com Promotion **Bootcamp** octobre **2021**. ")

    st.info("**Auteurs :** Noor Malla, Denise Nguene, Pauline Schieven, Andréas Trupin")
    
    st.header("Introduction")
    st.markdown(
        "Les données de notre projet ont été collectées sur un site de e-commerce, sur une période de 4 mois et demi.")
    st.markdown(
        "Dans cette présentation, nous vous montrerons quelques visualisations pour mieux comprendre le contexte ainsi qu'un  système de recommandation simplifié et nos modélisations.")

# .................................. End of Accueil .......................................

elif status == "Dataset":
    st.header("Notre jeu de données")
    
    st.subheader("1. Les actions sur le site :")
    st.markdown(
        "Pour chaque action faite sur le site, une ligne est ajoutée à la base de données  ‘events’, avec la date de l’événement et les informations de cette action. ")

    @st.cache
    def load_data(file_name, nrows):
        data = pd.read_csv(file_name)
        return data.head(nrows)

    events = load_data("events.csv", 10)

    st.write(events)

    st.info("Cette table contient 3 types d'évenement :  **view**, **addtocart** et **transaction**.")
    st.markdown(
        "Suite à une rapide description de la table events, nous savons que : Le site est visité par **1,407,580 visiteurs** qui réalisent **2,750,455 actions** (events) sur **235,061 produits**. On peut estimer à près de 2 le nombre d'actions en moyenne par visiteur sur toute la durée du dataset, à savoir 4 mois et demi.")

    st.subheader("2. Propriétés des produits :")
    st.markdown(
        "Cette table regroupe, par date, chaque modification effectuée sur une propriété d’un item. C’est sur ces données que la majorité est modifiée pour des raisons de confidentialité, et donc cela limite fortement l’exploitation de cette table.")
    # st.image("item_properties.jpg")

    # J'ai choisi d'afficher que une petite partie comme exemplaire de la table item_properties_part1 afin d'eviter de faire "concat" online
    properties = load_data('item_properties_part1.csv', 5)

    st.write(properties)

    st.subheader("3. Category Tree :")
    st.markdown("La dernière table regroupe des id de catégories assignés à des id parent.")
    # st.image("categories.jpg")

    categories = load_data("category_tree.csv", 5)

    st.write(categories)

# .................................. End of section .......................................



# .................................. Start of Dataviz section ............................
elif status == "Data visualisation":
    import plotly.graph_objects as go
    from matplotlib import cm

    st.title("Data visualisation")

    st.subheader("Evolution des événements au cours du temps")
    events_weekday =  pd.read_csv('events_weekday.csv')
    events_month =  pd.read_csv('events_month.csv')
    events_day =  pd.read_csv('events_day.csv')

    choix_temporalité = st.selectbox("Choisir la période à visualiser : ", ["Jour de la semaine", "Mois",
                                                               "Tous les jours"])
    if choix_temporalité == "Jour de la semaine":
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=events_weekday['weekday'],
                                 y=events_weekday['event_view'],
                                 name='Visites',
                                 line=dict(color='#66cdaa', width=2, dash='solid')))

        fig.add_trace(go.Scatter(x=events_weekday['weekday'],
                                 y=events_weekday['event_addtocart'],
                                 name='Ajouts au panier',
                                 line=dict(color='#b698f9', width=2, dash='solid')))

        fig.add_trace(go.Scatter(x=events_weekday['weekday'],
                                 y=events_weekday['event_transaction'],
                                 name='Transactions',
                                 line=dict(color='#ffa500', width=2, dash='solid')))

        fig.update_layout(title='Evenements par jour de la semaine',
                          xaxis_title='Jour de la semaine',
                          yaxis_title='Nombre d événements')

        st.plotly_chart(fig)

    elif choix_temporalité == "Mois":
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=events_month['month'],
                                 y=events_month['event_view'],
                                 name='Visites',
                                 line=dict(color='#66cdaa', width=2, dash='solid')))

        fig.add_trace(go.Scatter(x=events_month['month'],
                                 y=events_month['event_addtocart'],
                                 name='Ajouts au panier',
                                 line=dict(color='#b698f9', width=2, dash='solid')))

        fig.add_trace(go.Scatter(x=events_month['month'],
                                 y=events_month['event_transaction'],
                                 name='Transactions',
                                 line=dict(color='#ffa500', width=2, dash='solid')))

        fig.update_layout(title='Evenements par mois',
                          xaxis_title='Mois',
                          yaxis_title='Nombre d événements')
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[5, 6, 7, 8, 9],
                ticktext=['Mai', 'Juin', 'Juillet', 'Aout', 'Septembre']
            )
        )
        st.plotly_chart(fig)

    elif choix_temporalité == "Tous les jours":

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=events_day['date'],
                             y=events_day['event_view'],
                             name='Visites',
                             line=dict(color='#66cdaa', width=2, dash='solid')))

        fig.add_trace(go.Scatter(x=events_day['date'],
                             y=events_day['event_addtocart'],
                             name='Ajouts au panier',
                             line=dict(color='#b698f9', width=2, dash='solid')))

        fig.add_trace(go.Scatter(x=events_day['date'],
                             y=events_day['event_transaction'],
                             name='Transactions',
                             line=dict(color='#ffa500', width=2, dash='solid')))

        fig.update_layout(title='Evenements par journée',
                      xaxis_title='Journée',
                      yaxis_title='Nombre d événements')

        st.plotly_chart(fig)

    st.subheader("Proportion des différents types d'actions")
    st.markdown("> Nombre d'événements enregistrés : 2 756 100")
    events = pd.read_csv('events.csv')
    labels = events.event.value_counts().index
    values = events.event.value_counts().values
    fig_events = plt.figure(figsize=(7, 7))
    plt.pie(values, explode=(0, 0.2, 0.3), labels=labels, autopct='%1.1f%%', colors=['#66cdaa', '#b698f9', '#ffa500'])
    st.pyplot(fig_events)
    st.info("Nous voyons clairement que le nombre de visites représente la part la plus importante des événements du site, plus de 96%. \
            Le nombre d'achats, quant à lui, représente moins d'1% des actions totales du site. Il s'agit donc d'un site de e-commerce qui est très attractif et compte beaucoup de visites mais qui a une conversion assez faible.")

    st.subheader("Top Visiteurs")
    events_type = events.join(pd.get_dummies(events['event'], prefix='event'))
    events_type.head(10)

    # Création d'un df groupé par visiteur en fonction de leur nombre total de views, addtocart, transaction
    visitor_event = events_type.groupby('visitorid').agg({'event_addtocart': 'sum',
                                                          'event_transaction': 'sum',
                                                          'event_view': 'sum', }).astype('int')

    # Visualisation des Top visiteurs par type d'event
    top_visitor_view = visitor_event.sort_values(by='event_view', ascending=False).reset_index()
    top_visitor_cart = visitor_event.sort_values(by='event_addtocart', ascending=False).reset_index()
    top_visitor_trans = visitor_event.sort_values(by='event_transaction', ascending=False).reset_index()

    # Création du graphique représetant le Top 30 des meilleurs acheteurs et leur activité
    top_visitor_trans.head(30).plot.bar(x='visitorid',
                                        y=['event_transaction', 'event_addtocart', 'event_view'],
                                        label=['Achat', 'Panier', 'Visite'],
                                        color=['#ffa500', '#b698f9', '#66cdaa'],
                                        stacked=True,
                                        rot=25,
                                        figsize=(15, 10))

    plt.title("Répartition de l'activité  des 30 meilleurs acheteurs", fontsize=16)
    plt.xlabel('ID Visiteur')
    plt.ylabel("Nombre d'actions");
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.info(
        "Nous observons que la proportion des événements du premier graphique (camembert) correspond visuellement à la proportion des événements par visiteur pour ces 30 acheteurs.")
    st.markdown(
        "> Nous pouvons en déduire que le nombre de visites et de mises au panier n'est pas parfaitement proportionnel au nombre d'achats, mais il est fortement lié.")



    st.subheader("Top 10 Produits par action")

    items_view = events[(events['event'] == 'view')][['itemid', 'event']].groupby("itemid").count()
    items_view = items_view.sort_values(by='event', ascending=False)
    items_add = events[(events['event'] == 'addtocart')][['itemid', 'event']].groupby("itemid").count()
    items_add = items_add.sort_values(by='event', ascending=False)
    items_transaction = events[(events['event'] == 'transaction')][['itemid', 'event']].groupby("itemid").count()
    items_transaction = items_transaction.sort_values(by='event', ascending=False)


    fig_vus =plt.figure(figsize=(9, 9))
    ax = sns.barplot(x=items_view.index[:10], y=items_view['event'].head(10), order=items_view.index[:10], palette='RdYlBu');
    plt.title('Top 10 des produits visités')
    plt.xlabel('ID Produit')
    plt.ylabel('Nombre de visites')
    st.pyplot(fig_vus)

    st.markdown(
        "Sur le premier graphique, on voit que le produit le plus vu n’est pas dans le top 10 des plus ajoutés au panier ou achetés. Ce produit a certainement une mauvaise conversion. Même commentaire pour les autres produits (les 3ème, 4ème ou encore 6ème produits les plus vus n'apparaissent pas dans les Top 10 suivants).")
    st.info(
        "C'est donc le deuxième item le plus vu qui est le produit le plus ajouté au panier et également le plus acheté (item 461686). De plus, ce produit n°1 compte le double d'actions que le deuxième en termes d'ajouts au panier, c'est une différence importante.")

    fig_ajoutés = plt.figure(figsize=(9, 9))
    plt.title('Top 10 des produits ajoutés au panier', fontsize=16)
    sns.barplot(x=items_add.index[:10], y=items_add['event'].head(10), order=items_add.index[:10],
                palette='gist_earth');
    plt.xlabel('Top 10 des produits ajoutés au panier')
    plt.xlabel('ID Produit')
    plt.ylabel("Nombre d'ajouts au panier")
    st.pyplot(fig_ajoutés)
    st.markdown(
        "Le deuxième produit le plus ajouté au panier (312728), et également le quatrième produit le plus acheté, n’est pas dans les produits les plus visités. Celui-ci est alors très acheté et mis au panier avec peu de visites au préalable. On peut donc imaginer qu’il a une très bonne conversion ou peut être un produit qui est acheté régulièrement dans le cas d’un site B2B.")

    fig_achetés = plt.figure(figsize=(9, 9))
    plt.title('Top 10 des produits achetés', fontsize=16)
    sns.barplot(x=items_transaction.index[:10], y=items_transaction['event'].head(10),
                order=items_transaction.index[:10], palette='CMRmap')
    plt.xlabel('Top 10 des produits achetés')
    plt.xlabel('ID Produit')
    plt.ylabel("Nombre de transactions");
    st.pyplot(fig_achetés)


    st.markdown(
        "Les deuxième et troisième produits les plus achetés (119736 et 213834) ne sont pas parmi les plus vus et mis au panier : on peut supposer un manque de visibilité sur ces produits qui ont l’air de fonctionner à la vente. Ou également des produits qui sont régulièrement achetés directement par les entreprises (visiteurs dans le cas d’un e-commerce B2B) et qui n’ont donc pas beaucoup de visites.")

    st.subheader("Nombre de transactions par item")
    st.image("nb_transaction_par_item.jpg", width=500)
    st.info(
        "La grande majorité des produits n’ont pas été achetés sur la période étudiée. Uniquement 5 % des items ont été concernés par des transactions.")


    st.subheader("Nombre de vues avant l’achat : ")
    st.markdown("> ** L'objectif : savoir combien d’achats sont faits après X vues ?** ")
    st.image("nb_de_vu_avant_achat.jpg", width=500)
    st.info("La moitié des achats sont réalisés instantanément et même les ¾ sont réalisés avec une vue ou moins.")

    st.subheader("Horaires de fréquentation du site : ")
    st.image ("Transactions_par_heure.jpg")
    st.info("Les transactions sont essentiellement réalisées entre 15h et 23h.")

# .................................. End of section ......................................


# .................................. Start of sys. recommandation section .................

elif status == "Système de recommandation":
    st.header("Système de recommandation : ")
    st.subheader("Pour connaître les produits achetés par les autres visiteurs qui auraient acheté le même produit que vous.")
    @st.cache
    @st.cache
    def load_data(file_name, nrows):
        data = pd.read_csv(file_name)
        return data.head(nrows)
    events = load_data("events.csv", 100000)
    
    # Garder que les produits achétés 'transactionid not null'
    acheteurs = events[events.transactionid.notnull()].visitorid.unique()
    items_achetés = []
    # Liste de tous les visiteurs qui ont acheté

    # Liste des produits achetés
    for visitor in acheteurs:
        items_achetés.append(
            list(events.loc[(events.visitorid == visitor) & (events.transactionid.notnull())].itemid.values))
    # Fonction pour identifier objets achetés avec l objet d interet
    def recommendations(item_id, items_achetés):
        # on va créer un DF avec pour index chaque produit acheté & en valeurs la liste des produits achetés avec
        liste_recommendations = []
        for x in items_achetés:
            if item_id in x:
                liste_recommendations += x
        # on retire le produit lui même conseillé de cette liste pour ne pas qu'il réaparaisse dans les propositions
        liste_recommendations = list(set(liste_recommendations) - set([item_id]))
        return liste_recommendations


    col_one_list = events[events.transactionid.notnull()].itemid.unique()
    col_one_list.sort()
    selectbox_01 = st.selectbox('Choisissez un itemid pour lequel vous souhaitez avoir des recommendations :', col_one_list)

    st.write("Les autres visiteurs ayant acheté l'item ", selectbox_01, ", ont également acheté les items :", recommendations(selectbox_01, items_achetés))

    if st.checkbox("Cochez cette case si vous souhaitez connaître le nombre de personne ayant déjà acheté ce produit. "):
        def nb_acheteurs(item):
            visitor_id = list(events.loc[events.itemid == item].visitorid.values)
            return len(visitor_id)
        resultat=nb_acheteurs(selectbox_01)
        if resultat> 1:
            st.write("Voici le nombre de personnes ayant déjà acheté ce produit :", nb_acheteurs(selectbox_01)-1)

        else:
            st.write("Aucun autre visiteur n a acheté ce produit")

# .................................. End of section ................................................




# .................................. Start of Preprocessing section ......................
elif status == "Preprocessing":
    st.header("Preprocessing et features engineering : ")
    
    st.markdown("#### 1. Création d'un profil de produit (nouvelles variables)\
                : *'event_addtocart'* , *'event_transaction'* et *'event_view'* :")
    st.markdown("Le but d'obtenir pour chaque produit le nombre de fois qu'il a été vu, mis au panier, acheté.")
    #st.markdown("On ne récupère que les lignes qui ont fait l'objet d'une transaction, puisqu'on veut uniquement identifier le profil des produits achetés.")
    st.info("Cette table sera utilisée dans la partie (Modélisation) pour le clustering des produits.")
    @st.cache
    def load_data(file_name, nrows):
        data = pd.read_csv(file_name)
        return data.head(nrows)

    profil_transaction = load_data("profil_transaction.csv", 10)
    st.write(profil_transaction)
    
    
    st.markdown("#### 2. Création des variables *'Nombre de visites'* et *'Nombre de transactions'* :")
    #st.markdown("On fait un regroupement des clients du site en fonction des actions (events) enregistrées pendant ces 4 mois et demi afin que l’entreprise puisse mener des actions ciblées en fonction du type de visiteur.")
    st.info("Cette table sera utilisée dans la partie (Modélisation) pour le clustering des visiteurs.") 
    @st.cache
    def load_data(file_name, nrows):
        data = pd.read_csv(file_name)
        return data.head(nrows)

    df_kmeans_visiteurs = load_data("df_kmeans_visiteurs.csv", 10)
    st.write(df_kmeans_visiteurs)



    st.markdown("#### 3. Création des variables *'nb_items_vus'* , *'nb_vues'* et *'delta_view_add'* :")
    st.markdown(
        "Pour chaque visiteurid ayant vu & ajouté au moins un produit à leur panier, nous avons construit les variables : ")
    st.markdown("- nb_items_vus : nombre d'items différents vus par le visiteurid")
    st.markdown("- nb_vues : nombre de visites sur le site (tous produits confondus) du visiteurid")
    st.markdown("- delta_view_add : le temps moyen passé (en secondes) par le visiteurid avant d'ajouter un produit à son panier")
    st.info("Cette table sera utilisée dans la partie 'Modélisation', afin de prédire si un visitorid va acheter.")
    @st.cache
    def load_data(file_name, nrows):
        data = pd.read_csv(file_name)
        return data.head(nrows)

    predictions2 = load_data("predictions2.csv", 10)

    st.write(predictions2)

# .................................. End of section ......................................


# .................................. Start of modélisation section ..................................
elif status == "Modélisation":
    
    modele = st.sidebar.selectbox("Choisir le modèle à appliquer : ", ["Clustering des produits", "Clustering des visiteurs", "Modèle de prédiction d'achat"])
    
    #modele = st.selectbox("Choisir le modèle à appliquer : ", ["Clustering des produits", "Clustering des visiteurs",
    #                                                           "Modèle de prédiction d'achat"])
    
    if modele == "Clustering des produits":
        
        st.header("Clustering des produits")
        
        from bokeh.plotting import figure
        
        #@st.cache(allow_output_mutation=True)
        def load_data(file_name, nrows):
            data = pd.read_csv(file_name)
            return data.head(nrows)
        events = load_data("events.csv",3000000)
        
        # Preprocessing pour le cluster Produit et visiteur
        # Décomposer la date en année, mois, jour et heure. A voir pour une exploitation future dans la table event
        events['heure_event']= pd.to_datetime(events['timestamp'], unit = 'ms')
        events['jour'] = events['heure_event'] .dt.day
        events['mois'] = events['heure_event'] .dt.month
        events['annee'] = events['heure_event'] .dt.year
        events['heure'] = events['heure_event'] .dt.hour
        

        # les colonnes à supprimer :
        #>> timestamp et heure event : car elles sont décomposées dans d'autres colonnes
        events.drop(['timestamp', 'heure_event'], axis=1, inplace=True)
            
        # Création d'un sub dataframe avec les colonnes à utiliser dans le cadre du clustering
        subevents = events[["visitorid","itemid", "event","transactionid","heure"]]

    
        # Décomposition de la variable events en variable numériques
        subevents_type = subevents.join(pd.get_dummies(events['event'], prefix='event'))
        #subevents_type.head()
    
        # Comptabilisation des events par produit jusqu'à l'acte d'achat
        profil_produit = subevents_type.groupby('itemid').agg({'event_addtocart': 'sum',
                                                               'event_transaction': 'sum',
                                                               'event_view': 'sum', 
                                                               'visitorid': 'count'}).astype('int') 

        st.markdown("Objectif : Segmenter les produits selon le type d'évènement qui détermine leur achat.")

        # A partir du profil_produit filtrons à présent les lignes pour lesquelles il y a eu une transaction
        profil_transaction = profil_produit[profil_produit.event_transaction>=1].sort_values(by='event_transaction', ascending=False)

        # checkbox widget
        checkbox = st.checkbox("Afficher le dataframe filtré pour le clustering")
        print (checkbox)
        
        if checkbox:
            st.dataframe(data=profil_transaction.head(100))
    
        # Relation entre les variables
        # cor = profil_transaction.corr()
        # fig, ax = plt.subplots(figsize=(8,6))
        # sns.heatmap(cor, annot=True, ax=ax, cmap='coolwarm')
        #st.write(fig)
        
        #Relation entre la variable event_addcart et event_transaction
        #checkbox widget
        checkbox0 = st.checkbox("Afficher la corrélation des variables")
        print (checkbox0)        
        if checkbox0:
             st.image("Linearite_events.jpg")
             
        ## Clusterisation avec la méthode du coude
        checkbox1 = st.checkbox(" Afficher le coude optimal calculé")
        print (checkbox1)
        if checkbox1:  
            st.image("coude_optimal_calculé.jpg")
                       
        
        # Normalisation des données
        from sklearn.preprocessing import StandardScaler

        SC = StandardScaler()
        profil_transaction_norm = SC.fit_transform(profil_transaction)
        
        # Création d'un dataframe avec les 2 premières colonnes
        df_kmeans=pd.DataFrame(profil_transaction_norm[:,:2], columns=['event_addtocart','event_transaction'])
        
        cluster_slider = st.select_slider("Simulation calcul du nombre de clusters : ", [2,3,4,5])
        
        # Ajustement du modèle avec n_clusters = 3
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters= cluster_slider)
        kmeans.fit(df_kmeans)
        y_kmeans = kmeans.fit_predict(df_kmeans)

        
        # Visualization
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        for j in range(int(cluster_slider)): 
            lbl_cluster = 'Cluster' + str(j)
            plt.scatter(df_kmeans[y_kmeans==j].iloc[:,0], df_kmeans[y_kmeans==j].iloc[:,1],s=10, label=lbl_cluster)
                
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')   
        plt.title('Représentation des items selon leur cluster')
        plt.ylabel('Transactions')
        plt.legend(loc='best')
        st.pyplot()

                        
        # Assignation des labels au df de départ
        profil_transaction['Cluster_Id'] = kmeans.labels_

        checkbox2=st.checkbox("Afficher le dataframe affecté des clusters")
        print (checkbox2)
        
        if checkbox2:
            st.dataframe(data=profil_transaction.head(30))
            
        # Représentation graphique des clusters plot
        # checkbox widget
        checkbox3=st.checkbox("Afficher le décompte des items par cluster")
        print (checkbox3)
        
        if checkbox3:
            cluster = profil_transaction.groupby(['Cluster_Id']).agg({'event_transaction': 'count'}).reset_index() 
            st.dataframe(data=cluster)
                      
            
        # checkbox widget
        checkbox4=st.checkbox("Afficher le graphique des clusters")
        print (checkbox4)
        
        if checkbox4:
            fig5 = plt.figure()    
            profil_transaction['Cluster_Id'].value_counts(normalize=False).plot(kind='pie',figsize=(8, 8),
                                                                                 colors = ['yellowgreen', 'gold', 'lightcoral', 'blue','green', 'red'],
                                                                                 #explode = (0, 0.5, 0.8),
                                                                                 #shadow = True,
                                                                                 startangle=0,
                                                                                 autopct=lambda x: str(round(x, 2)) + '%')
            plt.title('Représentation  graphique des clusters')
            plt.legend()
            st.write(fig5)
        
        
        # Statitiques descriptives des clusters
        # checkbox widget          
        checkbox5 = st.checkbox("Afficher les Caractéristiques des clusters")
        print (checkbox5)
            
        if checkbox5: 
            groupe = st.radio("Quel groupe voulez-vous analyser ?", range(int(cluster_slider)))  #(0,1,2)
            for index in range(int(cluster_slider)):
                if groupe == index: 
                    st.write(profil_transaction[profil_transaction['Cluster_Id']==index].describe())
                
             
# ----- Fin clustering produits -----------------------------------------------

    elif modele == "Clustering des visiteurs":
        
        st.header("Clustering des visiteurs")
        
        from sklearn import preprocessing
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler        

        st.markdown("Objectif : Regrouper les visiteurs en fonction de leur activité sur le site afin d'orienter le marketing selon le type de client.")

    # Méthode du coude
        checkbox = st.checkbox(" Afficher le coude optimal")
        print (checkbox)
        if checkbox:  
            st.image("coude_vis.jpg", width=500)


        ncluster_vis=[2, 3, 4, 5]
        #selectbox_n_cluster_vis = st.selectbox('Choisir le nombre de clusteurs pour appliquer K-means :', ncluster_vis)        
        
        n = st.select_slider("Choisir le nombre de clusters pour appliquer les K-means : ", ncluster_vis)
        
        #x = st.slider("Taille de l'échantillon", min_value= 10000, max_value= 500000)
        #st.write('Nous prendrons aléatoirement', x , 'lignes du dataframe pour la modélisation.')
        
        #@st.cache
        def load_sample(file_name, nrows):
            data = pd.read_csv(file_name)
            return data.sample(nrows, random_state=888)
        df_kmeans_visiteurs = load_sample('df_kmeans_visiteurs.csv', 500000)
        kmeans_vis = KMeans(n_clusters= n)
        ## Ajustement 
        kmeans_vis.fit(df_kmeans_visiteurs)
        ## Prédictions
        y_kmeans_vis = kmeans_vis.predict(df_kmeans_visiteurs)
        
        for i in range(n) :
            cluster_name_vis = 'Cluster' + str(i)
            plt.scatter(df_kmeans_visiteurs[y_kmeans_vis==i].iloc[:,0], 
                        df_kmeans_visiteurs[y_kmeans_vis==i].iloc[:,1], 
                        s=10, label= cluster_name_vis)
            plt.legend()

        plt.xlabel('Visites')
        plt.ylabel('Transactions')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()        
        
        
        if st.checkbox("Afficher la représentation des centroïdes"):
            plt.scatter(kmeans_vis.cluster_centers_[:,0], kmeans_vis.cluster_centers_[:,1], color="#b967ff");
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        
        
    # Afficher le décompte des visiteurs par cluster         
        checkbox = st.checkbox("Afficher le décompte des visiteurs par cluster")
        print(checkbox)
            
        if checkbox:
            df_kmeans_visiteurs['cluster'] = kmeans_vis.labels_
            cluster_vis = df_kmeans_visiteurs.groupby(['cluster']).agg({'Nombre de transactions': 'count'}).reset_index()
            st.dataframe(data = cluster_vis)


# ------- Fin clustering visiteurs



    elif modele == "Modèle de prédiction d'achat":
        from sklearn import preprocessing
        from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from matplotlib import pyplot
        import plotly.graph_objects as go
        from sklearn import neighbors
        from sklearn import datasets
        
        st.header("Modèle de prédiction d'achat")
        
        @st.cache
        def load_data(file_name, nrows):
            data = pd.read_csv(file_name)
            return data.head(nrows)

        predictions2 = load_data("predictions2.csv", 30)
        st.markdown("Dataframe utilisé pour la modélisation (30 premières lignes) :")
        st.write(predictions2)
        
        st.info(" Features = 'delta_view_add', 'nb_items_vus', 'nb_vues' ")
        st.info(" Target = 'a acheté' ")
        feature_supp = ['Aucune', 'delta_view_add', 'nb_items_vus', 'nb_vues']
        selectbox_feature_supp = st.selectbox('Souhaitez-vous supprimer une feature du modèle :', feature_supp)
        taille = [0.2, 0.25, 0.3]
        selectbox_echantillon_test = st.selectbox('Choisissez la taille de l echantillon test :', taille)
        predictions = pd.read_csv('predictions.csv')

        if selectbox_feature_supp=='Aucune':
            X = predictions.drop(['visitorid', 'a acheté'], axis='columns')
        else:
            X = predictions.drop(['visitorid', 'a acheté', selectbox_feature_supp], axis='columns')

        y = predictions['a acheté']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=selectbox_echantillon_test)
        modele_prediction = ['LogisticRegression', 'K plus proches voisins']
        selectbox_echantillon_test = st.selectbox('Choisissez le modèle souhaité :', modele_prediction)

        if selectbox_echantillon_test == 'LogisticRegression':

            reg = LogisticRegression()
            reg.fit(X_train, y_train)
            train_acc = accuracy_score(y_true=y_train, y_pred=reg.predict(X_train))
            test_acc = accuracy_score(y_true=y_test, y_pred=reg.predict(X_test))
            st.write('- Score sur échantillon entraînement :', train_acc)
            st.write('- Score sur échantillon test :', test_acc)
            
            # Courbe hasard (50%)
            fauxpositif, vraipositif, _ = roc_curve(y_test, [0 for _ in range(len(y_test))])
            # Courbe avec notre modèle
            fauxpositif2, vraipositif2, _ = roc_curve(y_test, reg.predict(X_test))
            roc = go.Figure()
            roc.add_trace(go.Scatter(x=fauxpositif,
                                     y=vraipositif,
                                     name='Hasard',
                                     line=dict(color='orange', width=1, dash='solid')))
            roc.add_trace(go.Scatter(x=fauxpositif2,
                                     y=vraipositif2,
                                     name='Modèle',
                                     line=dict(color='blue', width=1, dash='dot')))
            roc.update_layout(title='Courbe ROC')
            st.plotly_chart(roc)

        elif selectbox_echantillon_test == 'K plus proches voisins':
            st.write('Métrique imposée : minkowski')
            knn = neighbors.KNeighborsClassifier(n_neighbors=30, metric='minkowski')
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            st.write('Score sur échantillon entraînement :', knn.score(X_train, y_train))
            st.write('Score sur échantillon test :', knn.score(X_test, y_test))

# .................................. End of section ......................................




# .................................. Start of Conculsion section ........................
else:
    st.header("Conclusion")
    st.write(
        """
Ce projet nous a permis de mettre en pratique les compétences acquises (principalement avec Python) lors de la formation, à savoir :
- l'exploration des données,
- la visualisation des données,
- le feature engineering,
- la création et l'évaluation de différents modèles de Machine Learning : 
    - prédiction d'achat
    - clustering
    """)

    st.header("Pour aller plus loin")    
    st.write(
        """
Sans certaines contraintes liées au jeu de données que nous avions, nous aurions pu :
- construire un système de recommandation grâce à un algorithme de machine learning, avec des données supplémentaires telles que les notes des produits
- développer un tableau de bord de visualisation pour l'entreprise (avec un chiffre d'affaires, coût des produits, catégories complètes des produits...)
    """)

    
# .................................. End of section ......................................

