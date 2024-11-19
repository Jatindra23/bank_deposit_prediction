# BANK TERM DEPOSIT PREDICTION
 -  **Problem Statement**: Predict whether Term deposit will be taken by the customer or not  based on the given features.

 -  **Dataset**: The dataset contains information about customers who have been called for a bank's term deposit.

 -  **Features**:  The dataset contains the following features
        - **Age**: The age of the customer
        - **Job**: The job of the customer
        - **Marital**: The marital status of the customer
        - **Education**: The education level of the customer
        - **Default**: Whether the customer has defaulted on a loan or not
        - **Balance**: The balance in the customer's account
        - **Income**: The income of the customer
        - **Phone**: Whether the customer has a phone or not
        - **Contact**: Whether the customer has been contacted or not
        - **Campaign**: The number of times the customer has been contacted
        - **Pdays**: The number of days since the customer was last contacted
        - **Previous**: Whether the customer has had a previous contact or not
        - **Duration**: The duration of the call in seconds
        - **Poutcome**: The outcome of the previous contact
        - **y**: Whether the customer took the term deposit or not
        -  **Target Variable**: The target variable is **y** which is a binary variable indicating whether


## Project Overview
The goal of this project is to predict the likelihood of clients subscribing to a term deposit based on client demographics and call interaction details. This solution helps financial institutions optimize their telephonic marketing campaigns by targeting potential customers more effectively, reducing operational costs, and increasing conversion rates.

---

## Features
- **Data Processing**: Handles missing values, encodes categorical features, and performs scaling.
- **Exploratory Data Analysis**: Visualizes relationships between features and target variables using Seaborn, Matplotlib, and Plotly.
- **Modeling**: Employs advanced machine learning algorithms like CatBoost and XGBoost, alongside traditional ones for comparison.
- **Model Evaluation**: Uses precision, recall, F1-score, and AUC-ROC for evaluation.
- **Deployment**: Model is deployed via a FastAPI web service, integrated with MongoDB for data storage, and monitored using Evidently.
- **Scalable Deployment**: Deployed on AWS and containerized with Docker for seamless scalability.

## Project Workflows

1. Data Collection: Import data from sources like MongoDB.
2. Data Cleaning and Preprocessing: 
   
Tasks:
Handle missing values.
Remove duplicates.
Convert data types (e.g., dates).
Normalize or scale features.
Categorical Data: Encode using one-hot or label encoding.
Imbalanced Data: Use SMOTE or similar techniques to balance the dataset.
Tools: pandas, numpy, imblearn.

3. Exploratory Data Analysis: Understand patterns and correlations in data.
   ![ From the above box plot, we observe outliers in the data. Further analysis using bar charts reveals that the overall subscription rate is only 11%. However, when filtering for durations greater than 2000 seconds i.e 33 mins, the subscription rate is slightly higher or equal. Therefore, these findings are meaningful and should be considered in the machine learning model.](https://github.com/Jatindra23/bank_deposit_prediction/blob/main/output1.png)


4. Model Training: Train models using algorithms like XGBoost and CatBoost with hyperparameter tuning.
5. Model Deployment: Deploy the trained model on AWS using FastAPI and Docker.
6. Monitoring and Feedback: Use Evidently to monitor model performance post-deployment.

7. Feature Engineering
Transformations: Create new features or modify existing ones to improve model accuracy.
Selection: Choose the most relevant features using statistical tests or feature importance from models like XGBoost.
Tools: pandas, scikit-learn.

## Requirements
Below is a list of libraries and tools required for the project:

```plaintext
pymongo==4.8.0
numpy
pandas
seaborn
statistics
matplotlib
scikit_learn
catboost
xgboost
imblearn
pydotplus
graphviz
python-dotenv==0.19.0
dnspython==2.6.0
PyYAML
dill
plotly
evidently==0.2.8
scipy
fastapi
httptools
mypy-boto3-s3==1.24.76
pip-chill==1.0.1
types-s3transfer==0.6.0.post4
uvicorn
watchfiles==0.17.0
websockets==10.3
wincertstore==0.2
neuro-mf==0.0.5
docker
Jinja2
python-multipart
-e .


## Installations

git clone https://github.com/your-username/bank-term-deposit-prediction.git
cd bank-term-deposit-prediction

-Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

-Install the required dependencies:
pip install -r requirements.txt

## Usage
# Run the FastAPI Server
1. Start the FastAPI server:
  uvicorn app.main:app --reload

2. Access the API documentation:

Open your browser and navigate to http://127.0.0.1:8000/docs to interact with the API.


## Prediction
Send a POST request to the /predict endpoint with client data to get predictions.


## Results

1. The API will return a JSON response with the predicted probability of a client taking a term deposit.

2. Improved model accuracy by 15% compared to baseline models.
3. Deployed model can predict client subscription with a improved precision and recall, helping banks optimize their campaigns.


## License

License
This project is licensed under the MIT License. See the LICENSE file for more details.
