'''
["metadata":{ 
"author": "Nadia Zaheer",
"gitlink": "https://github.com/nadiazk/assessments.git",
"version": "19092022v1",
"department": "Data and Decision Science" 
}]
'''



**Congratulations** for making it to the Challenge stage of the application process! 

The goal of the challenge is to give you a small taste of what it's like to work with our data as well as an opportunity for us to understand your process! **Understanding your process is just as important to us as your answers.**

*Before continuing, please check to make sure that the exercise is in your preferred language and that the filename is labeled with your name (e.g., if your name is Lucy, the filename should read "PFG-VBL ML Exercise - Python [Lucy].ipynb")*.

Please spend no more than 3 hours total on the exercise.

**About this notebook:**

This notebook from the current team of data scientists at PFG-VBL creates an ML model that predicts the probability that our customer will experience financial distress (delinquency) in the next two years. That is, they won't be able to pay back the credit/loan. This notebook uses a mock dataset of credit repayment difficulty rates among customers. Please note that the topic, data and problem are reflective of cases we have solved in the past.

---

**Background:**
VBL-PFG plays a crucial role in market economies. We, as a subprime lender, decide who can get finance and on what terms and can make or break investment decisions. We empathise with our customers. For markets and society to function, individuals and companies need access to credit. Credit scoring algorithms, which make a guess at the probability of default, are the method we use to determine whether or not a loan or credit should be granted. This model is within the domain of credit scoring decisions i.e. it predicts the probability that somebody will experience financial distress in the next two years.

Historical data are provided on 250,000 borrowers.

---
**About the data:**
Data is in two csv files:
cs-training.csv,
cs-test.csv

The variables are the following: SeriousDlqin2yrs is the target variable

**SeriousDlqin2yrs** Person experienced 90 days past due delinquency or worse (Target variable / label)

**RevolvingUtilizationOfUnsecuredLines**: Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits

**age** Age of borrower in years

**NumberOfTime30-59DaysPastDueNotWorse**: Number of times borrower has been 30-59 days past due but no worse in the last 2 years.

**DebtRatio**: Monthly debt payments, alimony,living costs divided by monthy gross income

**MonthlyIncome**: Monthly income

**NumberOfOpenCreditLinesAndLoans**: Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)

**NumberOfTimes90DaysLate**: Number of times borrower has been 90 days or more past due.

**NumberRealEstateLoansOrLines**: Number of mortgage and real estate loans including home equity lines of credit

**NumberOfTime60-89DaysPastDueNotWorse**: Number of times borrower has been 60-89 days past due but no worse in the last 2 years.

**NumberOfDependents**: Number of dependents in family excluding themselves (spouse, children etc.)

---
NDA: By undertaking this challenge, you are automatically bound by the non-disclosure agreement of PFG-VBL.



**Objective of your challenge:**
PFG-VBL data scientists have developed this scoring model and handed over this notebook to you. The goal of this tech challenge is to deploy and productionize this ML scoring model. The model predicts the probability that somebody will experience financial distress in the next two years. 

REMEMBER: Task is not to build the best model but to take it the next step i.e. properly ship and deploy this machine learning model so that we can continuously improve our models and ship out experiments faster.

**Your Deliverable:** 

*   Your submission is an API that is a train-offline and predict-realtime via rest API POST request e.g.
```@app.post("/predict", response_model=PredictResponse)```
* input_json: 
```
[
    {
        "ID": "1",
        "RevolvingUtilizationOfUnsecuredLines": "0.766126609",
        "age": "45",
        "NumberOfTime30-59DaysPastDueNotWorse": "2",
        "DebtRatio": "0.802982129",
        "MonthlyIncome": "9120",
        "NumberOfOpenCreditLinesAndLoans": "13",
        "NumberOfTimes90DaysLate": "0",
        "NumberRealEstateLoansOrLines": "6",
        "NumberOfTime60-89DaysPastDueNotWorse": "0",
        "NumberOfDependents": "2",
        "SeriousDlqin2yrs": "1"
    },
    {
        "ID": "2",
        "RevolvingUtilizationOfUnsecuredLines": "0.957151019",
        "age": "40",
        "NumberOfTime30-59DaysPastDueNotWorse": "0",
        "DebtRatio": "0.121876201",
        "MonthlyIncome": "2600",
        "NumberOfOpenCreditLinesAndLoans": "4",
        "NumberOfTimes90DaysLate": "0",
        "NumberRealEstateLoansOrLines": "0",
        "NumberOfTime60-89DaysPastDueNotWorse": "0",
        "NumberOfDependents": "1",
        "SeriousDlqin2yrs": "0"
    }]
```
* PredictResponse in json:
```
[{
    "Id": "1",
    "Probability": 0.29
  },
  {
    "Id": "2",
    "Probability": 0.03
  }]
```
Ensure to have the following modules for submission. Please email the git repo link, which sould be private. Oh! but don't forget to give us the access when you email the submission link! :)

1.   00_preprocessing_featurepipeline.py <-- should be the pipeline to preprocess the data into what model accepts
2.   01_model_gen.py [optional if you have time]
3.   02_model_prediction.py
4.   wrap the model in an API, dockerise and containerize this application so that we can readily test the rest API once we run your docker container using a post request in json.
5.  We love postman to test. 
6.  Make an output folder. It should attach a snapshot of the post request prediction output from model
7. Ensure to stick to OOP and make your code modular and reusable.
8. We like staying organized so we want to see how you organize your directory structure
9. readme.md should be self-explantory

---

Data Scientists at PFG-VBL used a random forest classifier for two reasons: firstly, because it would allow to quickly and easily change the output to a simple binary classification problem. Secondly, because the predict_proba functionality allows to output a probability score (probability of 1), this score is what is used for predicting the probability of 90 days past due delinquency or worse in 2 years time by the stakeholder teams and is built within product.

Furthermore, data scienitists have predominantly adopted a quantiles based approach in order to streamline the process as much as possible so that hypothetical credit checks can be returned as easily and as quickly as possible.

**Deadline** 5 days **Time Cap:** [3 hours MAX] We do NOT want you to spend more than 3 hours time on this. In case time runs out, please make a list of things you would like to do that we can discuss in the tech interview. This task is designed to give you insight into the kind of problems we work on. This is a take home task, where we want to see how you solve problems in real life when you have ALL the resources at your hand.

**Notes and Tips:**
* This notebook can be run on google colab and takes ~2 mins. 
* Feel free to run and save the model pickle/joblib from colab to save time
* Not looking for perfectionism as we want to ship things fast, ideally it should be a working piece of application.
* Wrap it in some API web framework (Hint: Fast API or Flask API). No need for a front end HTML. Keep it simple! 
* If you want to deploy on some cloud, feel free to do so. Modify your run instructions accordingly
* Feel free to ask questions
* If in doubt, feel free to make an assumption and the reasoning behind it. Ideally, I would prefer if you take note of it in your readme.md

**Preparing for the Tech interview:**

Some design considerations to keep on top of your mind for the next stage of tech interview:
* How would you test for incorrect user input?
* If you had more time what would you do? 
* How would you scale your API?
* How would you improve the architecture and your submitted design?
* How would you build monitoring around this? 
* Comment on this modelling approach taken by the data scientists. 
* What would you improve?

