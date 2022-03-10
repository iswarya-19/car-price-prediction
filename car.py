#Importing the Dependencies
from google.protobuf.symbol_database import Default
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
import streamlit as st
import plotly.express as px

#Data Collection and Processing
def main():
    st.title("Application")
    menu=["Main","About"]
    choice=st.sidebar.selectbox("Menu",menu)
    if choice=="Main":
        st.subheader("Car Price Prediction")
        data_file=st.file_uploader("Upload Your Dataset",type=["xlsx","csv"])
        if data_file is not None:
            car_dataset=pd.read_csv(data_file)
# inspecting the first 5 rows of the dataframe
            car_dataset.head()

# checking the number of rows and columns
            car_dataset.shape

# getting some information about the dataset
            car_dataset.info()

# checking the number of missing values
            car_dataset.isnull().sum()

# checking the distribution of categorical data
            print(car_dataset.Fuel_Type.value_counts())
            print(car_dataset.Seller_Type.value_counts())
            print(car_dataset.Transmission.value_counts())

#Encoding the Categorical Data

# encoding "Fuel_Type" Column
            car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# encoding "Seller_Type" Column
            car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# encoding "Transmission" Column
            car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

            car_dataset.head()

#Splitting the data and Target

            X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
            Y = car_dataset['Selling_Price']

            print(X)


            print(Y)



#Splitting Training and Test data

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

#Model Training

#1.Linear Regression

# loading the linear regression model
            lin_reg_model = LinearRegression()

            lin_reg_model.fit(X_train,Y_train)

#Model Evaluation

# prediction on Training data
            training_data_prediction = lin_reg_model.predict(X_train)

# R squared Error
            error_score = metrics.r2_score(Y_train, training_data_prediction)
            print("R squared Error : ", error_score)

#Visualize the actual prices and Predicted prices
            if st.button("linear Regression training graph"):
              def graph1():
                fig = px.scatter(
                x=Y_train,
                y=training_data_prediction,
                )
                fig.update_layout(
                  xaxis_title="Actual Price",
                  yaxis_title="Predicted Price",
                  )
                st.write(fig)
              graph1()
            


# prediction on Training data
            test_data_prediction = lin_reg_model.predict(X_test)

# R squared Error
            error_score = metrics.r2_score(Y_test, test_data_prediction)
            print("R squared Error : ", error_score)
            if st.button("linear Regression testing graph"):
              def graph2():
                fig1 = px.scatter(
                x=Y_test,
                y=test_data_prediction,
                )
                fig1.update_layout(
                  xaxis_title="Actual Price",
                  yaxis_title="Predicted Price",
                  )
                st.write(fig1)
              graph2()

            

#2.Lasso Regression

# loading the linear regression model
            lass_reg_model = Lasso()

            lass_reg_model.fit(X_train,Y_train)

#Model Evaluation

# prediction on Training data
            training_data_prediction = lass_reg_model.predict(X_train)

# R squared Error
            error_score = metrics.r2_score(Y_train, training_data_prediction)
            print("R squared Error : ", error_score)

#Visualize the actual prices and Predicted prices

            if st.button("lasso Regression training graph"):
              def graph3():
                fig2 = px.scatter(
                x=Y_train,
                y=training_data_prediction,
                )
                fig2.update_layout(
                  xaxis_title="Actual Price",
                  yaxis_title="Predicted Price",
                  )
                st.write(fig2)
              graph3()
            

# prediction on Training data
            test_data_prediction = lass_reg_model.predict(X_test)

# R squared Error
            error_score = metrics.r2_score(Y_test, test_data_prediction)
            print("R squared Error : ", error_score)

            if st.button("lasso Regression testing graph"):
              def graph4():
                fig3 = px.scatter(
                x=Y_test,
                y=test_data_prediction,
                )
                fig3.update_layout(
                  xaxis_title="Actual Price",
                  yaxis_title="Predicted Price",
                  )
                st.write(fig3)
              graph4()
            index_CarName=car_dataset.columns.get_loc('Car_Name')
            index_Year=car_dataset.columns.get_loc('Year')
            index_Sellingprice=car_dataset.columns.get_loc('Selling_Price')
            
#Searching 
#Search Using CarName
            
            def printing():
              carname=[]
              user_input=st.text_input("Enter The Car Name:","Eg.swift")
              carname.append(user_input)
              det= car_dataset.loc[car_dataset['Car_Name'].isin(carname),['Car_Name','Year','Selling_Price','Present_Price','Kms_Driven']]
              if det is not None:
                st.write(det)
              else:
                st.write("Try Again!!!")
            printing()
                 
    else:
        print("Thank You")
    
if __name__=='__main__':
    main()