# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income.csv",na_values=[ " ?"])
data

```
<img width="1783" height="853" alt="Screenshot 2025-10-07 103046" src="https://github.com/user-attachments/assets/c8a0a6e8-c59d-480a-8adc-99afdd6f923d" />

```
data.isnull().sum()

```
<img width="1678" height="666" alt="Screenshot 2025-10-07 103055" src="https://github.com/user-attachments/assets/c6f454c2-07b9-42ae-9d7b-53e0514ceeca" />

```
missing=data[data.isnull().any(axis=1)]
missing

```

<img width="1753" height="604" alt="Screenshot 2025-10-07 103108" src="https://github.com/user-attachments/assets/4aaae76e-7cb4-4921-8426-561f707bc6fb" />

```
data2=data.dropna(axis=0)
data2


```
<img width="1754" height="798" alt="Screenshot 2025-10-07 103118" src="https://github.com/user-attachments/assets/284f6530-f5ef-4021-9f2d-fe12a04f5e7e" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs

```

<img width="1065" height="624" alt="Screenshot 2025-10-07 103138" src="https://github.com/user-attachments/assets/27e32e23-29c7-4741-b991-607b8128feaf" />


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

<img width="1797" height="673" alt="Screenshot 2025-10-07 103149" src="https://github.com/user-attachments/assets/356f4a24-cc77-46e9-aa53-d2a3e485ef7d" />

```
columns_list=list(new_data.columns)
print(columns_list)
```

<img width="1804" height="137" alt="Screenshot 2025-10-07 103158" src="https://github.com/user-attachments/assets/df3603ed-d378-4ea0-9079-49b4c8875f82" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
<img width="1789" height="136" alt="Screenshot 2025-10-07 103206" src="https://github.com/user-attachments/assets/d60710ba-f5a3-463e-8c8a-9360bc1eac77" />

```
y=new_data['SalStat'].values
print(y)
```

<img width="1124" height="123" alt="Screenshot 2025-10-07 103213" src="https://github.com/user-attachments/assets/ab1c032f-3d44-4fe2-bc95-e890890b9c83" />

```
x=new_data[features].values
print(x)

```
<img width="979" height="246" alt="Screenshot 2025-10-07 103221" src="https://github.com/user-attachments/assets/51152d29-680a-42d1-b255-4a805a15eae5" />


```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)

```

<img width="931" height="226" alt="Screenshot 2025-10-07 103227" src="https://github.com/user-attachments/assets/a2a226d6-ccf1-4505-a437-2a4cd5c4c6c2" />

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
<img width="661" height="168" alt="Screenshot 2025-10-07 103234" src="https://github.com/user-attachments/assets/7f8b0c54-7e61-4f54-9b49-2a4c368cc73e" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

```
<img width="994" height="115" alt="Screenshot 2025-10-07 103242" src="https://github.com/user-attachments/assets/f2d57059-aef9-4423-baed-912bfd56e743" />

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="867" height="87" alt="Screenshot 2025-10-07 103247" src="https://github.com/user-attachments/assets/8bb8fafd-a568-4501-8ac1-2eabad1559ea" />

```
data.shape
```

<img width="900" height="102" alt="Screenshot 2025-10-07 103252" src="https://github.com/user-attachments/assets/18fe1b1c-cbdf-4e32-8de1-e23547579fbc" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
````
<img width="1829" height="539" alt="Screenshot 2025-10-07 103300" src="https://github.com/user-attachments/assets/ab4679e1-eadb-4fe6-972c-3ba30607b39d" />


```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="1765" height="430" alt="Screenshot 2025-10-07 103314" src="https://github.com/user-attachments/assets/9a42830f-400a-4c96-a1f6-87d30bc7d4b4" />


```
tips.time.unique()

```
<img width="696" height="125" alt="Screenshot 2025-10-07 103321" src="https://github.com/user-attachments/assets/48afffee-6cfc-4c0f-9589-2b6b8a83d3ab" />


```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

```

<img width="946" height="174" alt="Screenshot 2025-10-07 103334" src="https://github.com/user-attachments/assets/8af0ca00-b586-4641-9ef9-e0d8fff84b9e" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")

```
<img width="769" height="167" alt="Screenshot 2025-10-07 103339" src="https://github.com/user-attachments/assets/d95f5f9d-026d-40a9-8ae6-63a0b0512ea0" />














# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
