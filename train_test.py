import pandas as pd
df=pd.read_csv("Teen_Mental_Health_Dataset.csv")
df.head()
from sklearn.preprocessing import LabelEncoder
lc=LabelEncoder()
for col in df.columns:
    df[col]=lc.fit_transform(df[col])
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib



x = df.drop("depression_label", axis = 1)
y = df["depression_label"]

x_train , x_test , y_train , y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(n_estimators=200,
                               
                               random_state=42)

model.fit(x_train,y_train)

prediction = model.predict(x_test)
accuracy=accuracy_score(prediction,y_test)
print("testing accuracy:",accuracy)
pred_train=model.predict(x_train)
accuracy_train=accuracy_score(pred_train,y_train)
print("training accuracy:",accuracy_train)
joblib.dump(model , "model.pkl")


