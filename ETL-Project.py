

import requests
url="https://www.ssa.gov/oact/babynames/names.zip"
with requests.get(url) as response:
  with open("names.zip","wb") as temp_file:
    temp_file.write(response.content)

data_list =[["year","name","gender","count"]]

import csv

import zipfile
data_list =[["year","name","gender","count"]]
with zipfile.ZipFile("names.zip", 'r') as temp_zip:
  for file_names in temp_zip.namelist():
    if ".txt" in file_names:
      with temp_zip.open(file_names) as temp_file:
        for line in temp_file.read().decode("utf-8").splitlines():
          line_chunks=line.split(",")
          year=file_names[3:7]
          name=line_chunks[0]
          gender=line_chunks[1]
          count=line_chunks[2]
          data_list.append([year,name,gender,count])
csv.writer(open("data.csv","w",newline="",encoding="utf-8")).writerows(data_list)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df=pd.read_csv("data.csv")

df.head()

df.tail()

df=pd.read_csv("data.csv")
df=df[df["gender"] == "M"]
df=df[["name","count"]]
df=df.groupby("name")
df=df.sum()
df=df.sort_values("count",ascending=False)
df.head(10)

import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("data.csv")
df=df.pivot_table(index="name", columns="gender",
 values="count", aggfunc=np.sum).dropna()

df = df[(df["M"] >= 50000) & (df["F"] >= 50000)]
df.head(20)

df=pd.read_csv("data.csv")
df=df[df["gender"] == "F"]
df=df[["name","count"]]
df=df.groupby("name")
df=df.sum()
df=df.sort_values("count",ascending=False)
df.head(10)

import seaborn as sns

sns.set(style="ticks",
        rc={
            "figure.figsize": [12, 7],
            "text.color": "white",
            "axes.labelcolor": "white",
            "axes.edgecolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "axes.facecolor": "#443941",
            "figure.facecolor": "#443941"}
        )

import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("data.csv")

both_df = df.groupby("year").sum()
male_df = df[df["gender"] == "M"].groupby("year").sum()
female_df = df[df["gender"] == "F"].groupby("year").sum()

plt.plot(both_df, label="Both", color="yellow")
plt.plot(male_df, label="Male", color="lightblue")
plt.plot(female_df, label="Female", color="pink")
