#!/usr/bin/env python
# coding: utf-8

# # Car Prediction

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[2]:





# In[3]:


#pip install xlrd


# In[4]:


df = pd.read_excel('cars.xls')


# In[5]:


df.info()


# ## Veri Ön İşleme

# In[6]:


X = df.drop('Price', axis=1)
y = df['Price']


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state= 42)


# In[8]:


preprocess = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), ['Mileage', 'Cylinder', 'Liter', 'Doors']),
        ('cat', OneHotEncoder(), ['Make', 'Model', 'Trim', 'Type'])
    ]
)


# In[9]:


my_model = LinearRegression()


# In[10]:


#Pipeline tanımla
pipe = Pipeline(steps=[('preprocessor', preprocess ), ('model', my_model)])


# In[11]:


pipe.fit(X_train, y_train)


# In[12]:


y_pred = pipe.predict(X_test)
print('RMSE', mean_squared_error(y_test, y_pred)**0.5)
print('R2', r2_score(y_test, y_pred))


# In[13]:


#pip install streamlit


# In[16]:


import streamlit as st
def price(make, model, trim, mileage, car_type, cylinder, liter, doors, cruise, spund, sound, leather):
    input_data=pd.DataFrame({'Make':[make],
                             'Model':[model],
                             'Trim':[trim],
                             'Mileage':[mileage],
                             'Type':[car_type],
                             'Cylinder':[cylinder],
                             'Liter':[liter],
                             'Doors':[doors],
                             'Cruise':[cruise],
                             'Sound':[sound],
                             'Leather':[leather]})
    prediction=pipe.predict(input_data)[0]
    return prediction
st.title('2.El Otomobil Fiyat Tahmin @berkay_dun')
st.write('Arabanın özelliklerini seçiniz')
make = st.selectbox('Marka', df['Make'].unique())
model = st.selectbox('Model', df[df['Make'] == make]['Model'].unique())
trim=st.selectbox('Trim',df[(df['Make']==make) &(df['Model']==model)]['Trim'].unique())
mileage=st.number_input('Kilometre',100,200000)
car_type=st.selectbox('Araç Tipi',df[(df['Make']==make) &(df['Model']==model)&(df['Trim']==trim)]['Type'].unique())
cylinder=st.selectbox('Cylinder',df['Cylinder'].unique())
liter=st.number_input('Yakıt hacmi',1,10)
doors=st.selectbox('Kapı sayısı',df['Doors'].unique())
cruise=st.radio('Hız Sbt.',[True,False])
sound=st.radio('Ses Sis.',[True,False])
leather=st.radio('Deri döşeme.',[True,False])
if st.button('Tahmin'):
    pred=price(make,model,trim,mileage,car_type,cylinder,liter,doors,cruise,sound,leather)
    st.write('Fiyat:$', round(pred[0],2))


# In[ ]:




