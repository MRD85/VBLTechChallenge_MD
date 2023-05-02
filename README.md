ML Exercise - Model Deployment

(Quoted from Interview challenge!)
        The goal of this tech challenge is to deploy and productionize this ML scoring model. The model predicts the probability that somebody will experience financial distress in the next two years. 

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


My Approach

The excercise suggested running the notebook in GoogleColab to save time, I did try in VS Code initially but encountered issues as some fo the packages had since been deprecated. Same issue in Google Colab, so reverted back to VS Code and worked through them.

Once done, I needed to create a Pickle file to enable me to load the model in the API. The output was initially 1.6GB which was problematic due to Github storage limitations. As I only needed the model weights/Thetas to provide the prediction, this didn't seem right either. After multiple attempts i ended up with a Joblib file that was a more manageable ~250mb. As this was still over Github's 100mb limit, I had to enable Git Large File Storage (Git LFS) to allow syncing the repo.

Next came feature engineering, or more specifically applying the same steps applied within the provided workbook. So I simply removed all the EDA work, pyplots etc, df.head() or df.corr and other exploratory analysis until the model build section of the workbook. With the stripped down version, there were multiple similar steps that could be combined into more concise functions. I then created a class to apply each of the engineering steps sequentially. This became 00_preprossing_featurepipeline.py. 

With the _02_model prediction, the first issue I encountered was python not liking the Preprocessing file starting with a numeric Character, hence the "_" at the begining. I called the Preprocessor class to apply to the JSON input, declared the input structure as per the sample provided, pulled in the pickle model file. At a later stage I hit the issue of the API call not matching the expected model, so I called the expected variables from the model and added steps to add those columns on and impute the missing values as 0. (I believe this may be why I'm getting different predictions - but more on that in the suggested improvements section!)

With the app.py file, there was an issue with having "-" in the variable names, so I needed to create aliases using Pydantic "  NumberOfTime30_59DaysPastDueNotWorse: int = Field(alias="NumberOfTime30-59DaysPastDueNotWorse")" to allow the initial submission, and then update the DF to replace "-" with "_" to align to the rest of the model. (I did initially get passed the issue change the JSON input however circled back on myself as thought the closer I align to the initial request, the better! )

Then came the dockerfile, testing, trouble shooting. While I've not coverd everything here, it gives a flavour for the approach and some of the issues I encountered. I'm also sure there are better approaches out there, but we have a working API. 


## Running it on Docker 

- Clone this repository https://github.com/MRD85/VBLTechChallenge_MD.git

- Build the Docker image:
```commandline
    docker build --no-cache -t myimage:latest .
```
 - Run the docker image:
```commandline
 docker run -p 5000:5000 myimage:latest     
```

-Open POSTMAN

On Postman, once logged in:
 create a new request by clicking the + sign. 
 change the address to "http://0.0.0.0:5000/predict"
 on the left of the address, change the request type to "POST"
 click on the RAW selector
 click on the body section
 Paste:

```JSON
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
  }
]

```
- and click POST
- You then recieve the following response:

 ```JSON
[
    {
        "Id": "1",
        "Probability": 0.4161666666666667
    },
    {
        "Id": "2",
        "Probability": 0.46194308755760377
    }
]
```

 ## Ways to Improve

1. If I had more time, I would figure out what I did wrong. The responses did not match the suggested responses in the explanation. I suspect this is to do with missing a preprocessing step, scaler or I've imputed variables incorrectly. The other issue which is likely is the binning function, as this probably puts everything in the same bin, it would be better to map the thresholds and use them as a guide when applying the bins? Not sure...
2. Build a front end. This is purely a backend API however in a real-world scenario you'd want to simplify the user experience and perhaps build to accept CSVs. There's also the question of model monitoring, so kkep track of prdictions and assessing them to see if they did in fact default/enter delinquency over time. As with any model, as solid as your prediction is at the time of building, the world changes, and so too should your models. 
3. While my approach called functions from other files, resulting in the need to rename, it perhaps may have been better to complete and output the dataframe and then call it separately. This goes against the instruction to make this modular/reusable as they rely on eachother. 
