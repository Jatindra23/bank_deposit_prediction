# BANK TERM DEPOSIT PREDICTION
 -  **Problem Statement**: Predict wether term deposit will be taken by the customer or not  based on the given features.

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







-  Git commandline
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Jatindra23/bank_term_deposit_prediction.git
git push -u origin main


-  to remove file from the github repository (for files which are in .gitignore file)
 git rm -r --cached <filename>
 git commit -m "message"
 git push u origin main


- Error in the setup file 
  1. if you get the error in the setup file which says that the  package version is not supported then you can use the following command to install the required version of python the use the  following command
        pip install -r requirements.txt --user
    or python setup.py install --user

