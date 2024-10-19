import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Cargar el dataset
df = pd.read_csv('GrammarandProductReviews.csv')

# Layout de la página
st.title('Interactive Dashboard for Product Reviews')

# Sidebar para seleccionar opciones
st.sidebar.title('Options')
selected_visual = st.sidebar.selectbox('Select a visualization', ['Data Overview', 'Word Cloud', 'Ratings Distribution', 'Helpfulness vs Ratings', 'Model Comparison'])

# Sección de exploración de datos
if selected_visual == 'Data Overview':
    st.header('Exploration of the Dataset')
    st.write(df.head())

    # Gráficos interactivos usando Plotly
    fig = px.histogram(df, x='reviews.rating', title="Distribution of Ratings")
    st.plotly_chart(fig)

    fig2 = px.scatter(df, x='reviews.numHelpful', y='reviews.rating', title="Helpfulness vs Rating")
    st.plotly_chart(fig2)

# Visualización del Word Cloud
elif selected_visual == 'Word Cloud':
    st.header('Word Cloud of Reviews')
    # Generar el Word Cloud
    reviews_text = ' '.join(df['reviews.text'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reviews_text)
    
    # Mostrar el Word Cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Distribución de calificaciones con posibilidad de filtrar por marca
elif selected_visual == 'Ratings Distribution':
    st.header('Ratings Distribution by Brand')

    # Filtro para seleccionar marcas
    selected_brand = st.selectbox('Select a Brand', df['brand'].unique())

    filtered_df = df[df['brand'] == selected_brand]
    
    fig = px.histogram(filtered_df, x='reviews.rating', nbins=5, title=f'Distribution of Ratings for {selected_brand}')
    st.plotly_chart(fig)

# Visualización de "Helpfulness vs Ratings"
elif selected_visual == 'Helpfulness vs Ratings':
    st.header('Helpfulness vs Ratings')

    # Dropdown para seleccionar marca
    selected_brand = st.selectbox('Select a Brand', df['brand'].unique())
    filtered_df = df[df['brand'] == selected_brand]

    fig = px.scatter(filtered_df, x='reviews.numHelpful', y='reviews.rating', title=f'Helpfulness vs Ratings for {selected_brand}')
    st.plotly_chart(fig)

# Comparación de modelos
elif selected_visual == 'Model Comparison':
    st.header('Comparison of Classification Models')

    # Preprocesamiento del dataset
    df['sentiment'] = df['reviews.rating'].apply(lambda x: 1 if x >= 4 else 0)  # Sentimiento positivo/negativo

    # Asegúrate de que no haya valores nulos en las columnas usadas para X y y
    df_clean = df[['reviews.numHelpful', 'reviews.rating', 'sentiment']].dropna()

    X = df_clean[['reviews.numHelpful', 'reviews.rating']]
    y = df_clean['sentiment']

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Definir los modelos a comparar
    models = {
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC()
    }

    selected_models = st.multiselect('Select Models to Compare', list(models.keys()), default=['Random Forest'])

    results = {}

    # Entrenar y evaluar los modelos seleccionados
    for model_name in selected_models:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Métricas del modelo
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[model_name] = {'Accuracy': acc, 'Confusion Matrix': cm}

        st.subheader(f'Metrics for {model_name}')
        st.write(f'Accuracy: {acc:.2f}')
        st.write(f'Confusion Matrix:')
        st.write(cm)

    # Gráfico de comparación de precisión
    if len(results) > 1:
        st.subheader('Model Comparison: Accuracy')
        accuracy_fig = px.bar(x=list(results.keys()), y=[results[model]['Accuracy'] for model in results.keys()], 
                              labels={'x': 'Model', 'y': 'Accuracy'}, title='Accuracy Comparison of Models')
        st.plotly_chart(accuracy_fig)
