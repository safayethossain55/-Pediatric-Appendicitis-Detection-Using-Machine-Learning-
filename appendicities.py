
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
# Read the dataset into a Pandas DataFrame
df = pd.read_csv(r"/content/drive/MyDrive/appendicities.csv")
cols = len(df.axes[1])

# Replace all occurrences of "NA" or "na" with NaN
df.replace({"NA": float("nan"), "NaN": float("nan")}, inplace=True)
#df.replace({"NA": float("nan"), "NaN": float("nan"),"+": float("nan"),"++": float("nan"),"+++": float("nan")}, inplace=True)
df=df.drop(['Age'],axis=1)
df=df.drop(['Height'],axis=1)
df=df.drop(['Weight'],axis=1)
df=df.drop(['Sex'],axis=1)
df=df.drop(['TissuePerfusion'],axis=1)
df=df.drop(['BowelWallThick'],axis=1)
df=df.drop(['Ileus'],axis=1)
df=df.drop(['FecalImpaction'],axis=1)
df=df.drop(['Meteorism'],axis=1)
df=df.drop(['Enteritis'],axis=1)
#dff=df.drop([''],axis=1)
#df.BodyTemp.fillna(0, inplace=True)
#df.NeutrophilPerc.fillna(0, inplace=True)
#df.WBCCount.fillna(0, inplace=True)
#df.CRPEntry.fillna(0, inplace=True)


# Fill missing values with interpolation
#df = df.apply(pd.to_numeric, errors='coerce')
for column in df.columns:
    if df[column].dtype == 'object':
        if df[column].notnull().any():
            df[column] = df[column].fillna(df[column].mode()[0])
# Fill missing values with interpolation
#df.interpolate(method='linear', inplace=True)
df.interpolate(method='linear', inplace=True)
#print(df.iloc[2])

# computing number of rows
rows = len(df.axes[0])
# computing number of columns
cols = len(df.axes[1])
print(df)
print(df.describe())








names= ['Age','BMI','Sex','Height','Weight','AlvaradoScore','PediatricAppendicitisScore','AppendixOnSono','AppendixDiameter','MigratoryPain','LowerAbdominalPainRight','ReboundTenderness','CoughingPain','PsoasSign','Nausea','AppetiteLoss','BodyTemp','WBCCount','NeutrophilPerc','KetonesInUrine',
        'ErythrocytesInUrine','WBCInUrine','CRPEntry','Dysuria','Stool','Peritonitis','FreeFluids','AppendixWallLayers','TissuePerfusion','SurroundingTissueReaction','PathLymphNodes','MesentricLymphadenitis','BowelWallThick','Ileus','FecalImpaction','Meteorism','Enteritis','DiagnosisByCriteria',
        'TreatmentGroupBinar','AppendicitisComplications']
plt.hist(names, bins=50)
plt.show()









df['AppendixOnSonoF'] = pd.factorize(df['AppendixOnSono'])[0]
df['MigratoryPainF'] = pd.factorize(df['MigratoryPain'])[0]
df['LowerAbdominalPainRightF'] = pd.factorize(df['LowerAbdominalPainRight'])[0]
df['ReboundTendernessF'] = pd.factorize(df['ReboundTenderness'])[0]
df['CoughingPainF'] = pd.factorize(df['CoughingPain'])[0]
df['PsoasSignF'] = pd.factorize(df['PsoasSign'])[0]
df['NauseaF'] = pd.factorize(df['Nausea'])[0]
df['AppetiteLossF'] = pd.factorize(df['AppetiteLoss'])[0]
df['KetonesInUrineF'] = pd.factorize(df['KetonesInUrine'])[0]
df['ErythrocytesInUrineF'] = pd.factorize(df['ErythrocytesInUrine'])[0]
df['WBCInUrineF'] = pd.factorize(df['WBCInUrine'])[0]
df['DysuriaF'] = pd.factorize(df['Dysuria'])[0]
df['StoolF'] = pd.factorize(df['Stool'])[0]
df['PeritonitisF'] = pd.factorize(df['Peritonitis'])[0]
df['FreeFluidsF'] = pd.factorize(df['FreeFluids'])[0]
df['AppendixWallLayersF'] = pd.factorize(df['AppendixWallLayers'])[0]
df['KokardeF'] = pd.factorize(df['Kokarde'])[0]
df['SurroundingTissueReactionF'] = pd.factorize(df['SurroundingTissueReaction'])[0]
df['PathLymphNodesF'] = pd.factorize(df['PathLymphNodes'])[0]
df['MesentricLymphadenitisF'] = pd.factorize(df['MesentricLymphadenitis'])[0]
df['DiagnosisByCriteriaF'] = pd.factorize(df['DiagnosisByCriteria'])[0]
df['TreatmentGroupBinarF'] = pd.factorize(df['TreatmentGroupBinar'])[0]
df['AppendicitisComplicationsF'] = pd.factorize(df['AppendicitisComplications'])[0]
print(df.iloc[2])







df=df.drop(['AppendixOnSono'],axis=1)
df=df.drop(['MigratoryPain'],axis=1)
df=df.drop(['LowerAbdominalPainRight'],axis=1)
df=df.drop(['ReboundTenderness'],axis=1)
df=df.drop(['CoughingPain'],axis=1)
df=df.drop(['PsoasSign'],axis=1)
df=df.drop(['Nausea'],axis=1)
df=df.drop(['AppetiteLoss'],axis=1)
df=df.drop(['KetonesInUrine'],axis=1)
df=df.drop(['ErythrocytesInUrine'],axis=1)
df=df.drop(['WBCInUrine'],axis=1)
df=df.drop(['Dysuria'],axis=1)
df=df.drop(['Stool'],axis=1)
df=df.drop(['Peritonitis'],axis=1)
df=df.drop(['FreeFluids'],axis=1)
df=df.drop(['AppendixWallLayers'],axis=1)
df=df.drop(['Kokarde'],axis=1)
df=df.drop(['SurroundingTissueReaction'],axis=1)
df=df.drop(['PathLymphNodes'],axis=1)
df=df.drop(['MesentricLymphadenitis'],axis=1)

df=df.drop(['DiagnosisByCriteria'],axis=1)
df=df.drop(['TreatmentGroupBinar'],axis=1)
df=df.drop(['AppendicitisComplications'],axis=1)




X = df.drop('DiagnosisByCriteriaF', axis=1)
y = df['DiagnosisByCriteriaF']



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the logistic regression model
log_reg = LogisticRegression()

# Fit the model to the training data
log_reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = log_reg.predict(X_test)

# Calculate the accuracy of the model
acc = accuracy_score(y_test, y_pred)

# Print the accuracy of the model
print("Accuracy:", acc)





# Define the base models
log_reg = LogisticRegression()
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the base models to the training data
log_reg.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Make predictions on the test data using the individual models
y_pred_log_reg = log_reg.predict(X_test)
y_pred_random_forest = random_forest.predict(X_test)

# Define weights for the models
weight_log_reg = 0.6
weight_random_forest = 0.4

# Calculate the weighted ensemble predictions
weighted_ensemble_pred = (weight_log_reg * y_pred_log_reg + weight_random_forest * y_pred_random_forest).round()

# Calculate the accuracy of the weighted ensemble model
acc_weighted_ensemble = accuracy_score(y_test, weighted_ensemble_pred)
print("Weighted Ensemble Model Accuracy:", acc_weighted_ensemble)


mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
print("The model performance for testing set")
print("--------------------------------------")
print('MAE is %.2f'% mae)
print('MSE is %.2f'% mse)
print('R2 score is %.2f'% r2)