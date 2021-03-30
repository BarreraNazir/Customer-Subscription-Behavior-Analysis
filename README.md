# Customer-Subscription-Behavior-Analysis-Using_Machine-Learning

**Problem Identification**

   This project analyze the behavior of **Fin-Tech** customers. Its a financial app, having two version namely free, and premium. The free version contain basic features and advance features the user has to enroll. The aim of this project is to sell the premium version in low budget. But, how to find the customers to purchase the subscription of financial app? Offers will be given to those who are interested in premium version. And for that, customers behavior will be analyzed for 24 hours.
    
**Goal**    
    
   The main goal of the company is to sell the premium version app with low advertisement cost. 
 
**Lets Get Started**

Importing Libraries and Data
Data Visualization
Feature Selection
Building Model
Cross Validation

**Importing Libraries and Data**

   * Importing Libraries
            
        The task is performed in Python Language with its libraries NumPy, Pandas, Matplotlib, Seaborn, and Sklearn.    
   
   * Data

        The dataset contains 50,000 customers information with 12 features. All columns have 50,000 non-null values except enrolled_date. It has 31,074 non-null. There is a total of 8 columns that contain integer 64 bit (int64) values and the remaining 4 are object data type.
      
**Data Visualization**

   * Heatmap using the correlation matrix:

        Heatmap used to find the correlation between each and every features using the correlation matrix.
   
   * Pair plot:
   
        The pair plot helps to visualize the distribution of data and scatter plot.
        
   * Histogram:

        Here we visualize in the histogram to understand each feature more easily.
   
   * Correlation barplot:
   
        Visualizing the features to know which one is strongly correlated with ‘enrolled’ through barplot.
         
**Feature Selection**

   Considering those customers who have enrolled after 48 hours as 0 plus dropping those features which are not correlated with each other. 
 
**Building Model**
   
   I trained and test the dataset with multiple Machine Learning algorithms to find the best model such as Decision Tree, K-NN, Naive Bayes, Random Forest, Logistic Regression, Support Vector Machine, and XGBoost. Among all classifiers, XGBoost gave 78 % accuracy.
 
**Cross Validation**
   
   Performing cross-validation to find out that the ML model is overfitted, under fitted or generalized which results that the model is generalized.  
