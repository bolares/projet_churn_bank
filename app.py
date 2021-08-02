import streamlit as st
import numpy as np
import pandas as pd
import fsspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from fsspec.registry import known_implementations


st.write('''#   BIENVENUE SUR VOTRE PORTAIL
            VERIFICATION DE LA FIDELISATION
                  DE VOS CLIENTS''')

def main_entry():
    # slider dans la sidebar pour les variable quantitative:
    Total_Relationship_Count = st.number_input('Total_Relationship_Count')

    # slider dans la sidebar pour les variable quantitative:
    Contacts_Count_12_mon = st.number_input('Contacts_Count_12_mon')

    # slider dans la sidebar pour les variable quantitative:
    Credit_Limit = st.number_input('Credit_Limit')

    # slider dans la sidebar pour les variable quantitative:
    Total_Revolving_Bal = st.number_input('Total_Revolving_Bal')

    # slider dans la sidebar pour les variable quantitative:
    Avg_Open_To_Buy = st.number_input('Avg_Open_To_Buy')

    # slider dans la sidebar pour les variable quantitative:
    Total_Trans_Amt = st.number_input('Total_Trans_Amt')

    # slider dans la sidebar pour les variable quantitative:
    Total_Trans_Ct = st.number_input('Total_Trans_Ct')
    
    data = {'Total_Relationship_Count':Total_Relationship_Count, 'Contacts_Count_12_mon':Contacts_Count_12_mon, 'Credit_Limit':Credit_Limit,
       'Total_Revolving_Bal':Total_Revolving_Bal, 'Avg_Open_To_Buy':Avg_Open_To_Buy, 'Total_Trans_Amt':Total_Trans_Amt,
       'Total_Trans_Ct':Total_Trans_Ct}

    bank_client = pd.DataFrame(data, index = [0] )
    return bank_client 

df = main_entry()

st.subheader('caracteristiques du client')
st.write(df)
#CHARGEMENT DES DONNEES
da = pd.read_csv("Dataset.csv", sep=';')
dat = da.copy()
#TRAITEMENT
dat.replace("Unknown", np.nan, inplace = True)
dat.dropna (axis = 0, how = 'any', inplace = True)
#encodage
encod = {'Married': 2, 'Single': 0, 'Divorced':1,'M': 0, 'F':1,
         'Blue': 4, 'Silver': 3, 'Gold':2, 'Platinum':1,
         'Uneducated': 6, 'College': 5, 'High School':4, 'Graduate':3, 'Post-Graduate':2, 'Doctorate':1,
         'Less than $40K': 5, '$40K - $60K': 4, '$60K - $80K':3, '$80K - $120K':2, '$120K +':1,
         'Existing Customer':1, 'Attrited Customer':0}
for col in dat.select_dtypes("object"):
    dat[col] = dat[col].map(encod)
#division des données
y = dat["Attrition_Flag"]
dat.drop(["Attrition_Flag"], inplace = True, axis = 1)
X = dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
#SELECTION DES VARIABLE PERTINENTES
select_var = SelectKBest(chi2, k=7) # appel de l'algo chi2
select_var.fit_transform(X_train, y_train)
a = select_var.get_support() # resultat
a = pd.DataFrame(a, index=X_train.columns).T
a = a[a==True].dropna(1)
X_train = X_train[a.columns]
#ENTRAINEMENT SUR LE MODEL SELECTIONNE AVEC PREDICTION DU CLIENT SAISI
model = RandomForestClassifier(criterion = "entropy", n_estimators = 100, max_depth = 10,  random_state = 5)
model.fit(X_train, y_train)
prediction = model.predict(df)
#prediction = prediction.tolist()
#st.write(prediction)
#AFFICHAGE DE LA PREDICTION
#submit_button = st.form_submit_button(label='SOUMETTRE')
#if submit_button:
    #st.write(f' {prediction}')
if prediction == 1:
    st.write("CLIENT SATISFAIT DU SERVICE")
if prediction == 0:
    st.write("CLIENT NON-SATISFAIT DU SERVICE")
#st.button("PREDICTION")
#st.write(prediction)

