>To execute the project - 
>1. First - Run the stream.py from streamlit environment.
>2. Second - Upload the Dataset.csv in the interface

***THIS README FILE EXPLAINS THE PROJECT STRUCTURE***

**4 .ipynb files with 1 additional .py file for deployment**
**1 Project Presentation file .ppt**
**2 .csv dataset files**
**1 .pdf file showing the deployed model in streamlit**
**1 folder containing 8 files for dataset creation, including a .txt file explaining the features**

#### Initial_Raw_EDA.ipynb (Optional Run)
> Here we do the initial EDA on the raw data

#### Covid_Handling_Cross-sectional_EDA.ipynb (Optional Run)
> Here we look at how we handle the covid data which we considered as a big outlier

#### WTI_VAR_Model.ipynb + New-Var.csv (Optional Run)
> Here we independently test the VAR model with statistics and it's predictions.

#### Dataset_Build_and_Model.ipynb (Required Run)
> Here we build the dataset initially with Total 6 features excluding Date and Close Price of Oil. In the code,
* 1. Date[1] and Close Price[2] is scrapped through yahoo finance website
* 2. Features[3 to 8] is scrapped in .csv and .xls files from the following websites:
F[3] U.S. Percent Utilization of Refinery Operable Capacity: https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WPULEUS3&f=W,
F[4] Last Day's Price is shift of Close Price[2],
F[5 & 6] Production and Demand: https://www.iea.org/data-and-statistics/charts/world-oil-supply-and-demand-1971-2020,
F[7] NYMEX Futures Prices: https://www.eia.gov/dnav/pet/pet_pri_fut_s1_d.htm,
F[8] Global Electronic vehicle Sales: https://www.iea.org/data-and-statistics/charts/global-electric-car-sales-by-key-markets-2015-2020

> Then we do data pre-processing and handle the COVID-outlier and generate the Dataset.csv file, the input to streamlit deployment. Further, we build and run the model to generate the forecast price. Note that the statistical analysis and EDA is not in this file. This is post EDA and analysis, just to generate the dataset and test the working model. The .csv and .xls files used to make the dataset is inside the folder 'feature_datasets'


#### stream.py + Dataset.csv + Deployment_Screen.pdf (Required Run)
> The stream.py file contains the deployment code using VAR model for this project. The Dataset.csv is the updated dataset file (last updated 27/12/2021) of oil data needed for the deployment to run. You can generate the updated dataset via the .ipynb files above. The online cloud platform could have been used to get the data automatically from scrapping sites, but it involves a chargee/premium fee. Thus we generate our own dataset using yfinance for each streamlit run. The Deployment_Screen.pdf has the screenshot of the deployed program for reference.




