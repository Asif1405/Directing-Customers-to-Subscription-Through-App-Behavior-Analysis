import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from dateutil import parser
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


df = pd.read_csv("appdata10.csv")
print(df.columns)

df.hour.str.slice(1,3).astype(int)
print(df["hour"].head())

df2 = df.copy()

visualizing
df2 = df2.drop(columns = ["user", "screen_list", "enrolled_date", "first_open", "enrolled"])

plt.suptitle("Histograms", fontsize = 20)
for i in range(1, df2.shape[1]+1):
    plt.subplot(3,3,i)
    f = plt.gca()
    f.set_title(df2.columns.values[i-1])

    vals = np.size(df2.iloc[:,i-1].unique())

    plt.hist(df2.iloc[:,i-1], bins = vals, color = "#3F5D7D")


df2.corrwith(df.enrolled).plot.bar(figsize = (20,10), title = "correlation",
                                fontsize = 15, rot = 45, grid = True)


sn.set(style = "white")
corr = df2.corr()

mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize = (18,15))

cmap = sn.diverging_palette(220,10,as_cmap = True)

sn.heatmap(corr,mask = mask, cmap = cmap, vmax = 3, center = 0,
           square = True, linewidth = 5, cbar_kws = {"shrink":5})

plt.show()


Feature Engg
df["first_open"] = [parser.parse(row_data) for row_data in df["first_open"]]
df["enrolled_date"] = [parser.parse(row_data) if isinstance(row_data, str) else row_data  for row_data in df["enrolled_date"]]

df["difference"] = (df.enrolled_date - df.first_open).astype("timedelta64[h]")

plt.hist(df["difference"].dropna(), color = "blue")
plt.title("Distribution of Time")
plt.show()

df.loc[df.difference > 48, "enrolled"] = 0
df = df.drop(columns = ["difference", "enrolled_date", "first_open"])

top_screens = pd.read_csv("top_screens.csv").top_screens.values
df["screen_list"] = df.screen_list.astype(str) + ","

for sc in top_screens:
    df[sc] = df.screen_list.str.contains(sc).astype(int)
    df["screen_list"] = df.screen_list.str.replace(sc+",", "")

df["Other"] = df.screen_list.str.count(",")
df = df.drop(columns = ["screen_list"])


savings_screens = ["Saving1",
                  "Saving2",
                  "Saving2Amount",
                  "Saving4",
                  "Saving5",
                  "Saving6",
                  "Saving7",
                  "Saving8",
                  "Saving9",
                  "Saving10"]
df["SavingsCount"] = df[savings_screens].sum(axis = 1)
df = df.drop(columns = savings_screens)

cm_screens = ["Credit1",
             "Credit2",
             "Credit3",
             "Credit3Container",
             "Credit3Dashboard"]
df["CMCount"] = df[cm_screens].sum(axis = 1)
df = df.drop(columns = cm_screens)

cc_screens = ["CC1",
             "CC1Category",
             "CC3"]
df["CCCount"] = df[cc_screens].sum(axis = 1)
df = df.drop(columns = cc_screens)

loan_screens = ["Loan",
                "Loan2",
                "Loan3",
                "Loan4"]
df["LoanCount"] = df[loan_screens].sum(axis = 1)
df = df.drop(columns = loan_screens)

df.to_csv("new_appdata10.csv", index = False)


Data Preprocessing
data = pd.read_csv("new_appdata10.csv")
data = data.drop(columns = "hour")

y = data["enrolled"]
x = data.drop(columns = ["enrolled"])
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size = 0.25, random_state  = 0)

train_identifier = x_train["user"]
x_train = x_train.drop(columns = "user")
test_identifier = x_test["user"]
x_test = x_test.drop(columns = "user")

sc_x = ss()
x_train2 = pd.DataFrame(sc_x.fit_transform(x_train))
x_test2 = pd.DataFrame(sc_x.fit_transform(x_test))
x_train2.columns = x_train.columns
x_test2.columns = x_test.columns
print(x_train2)
x_train = x_train2
x_test = x_test2


MODEL
Logistice Regression
LR = LogisticRegression(random_state = 0, penalty = "l1")
LR.fit(x_train, y_train)

y_pred = LR.predict(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

result = pd.DataFrame([["Linear Regression", acc, prec, rec, f1]],
             columns  = ["Model", "Accuracy", "precision", "Recall", "F1 score"])

print(result)
