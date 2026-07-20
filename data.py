import pandas as pd
df=pd.read_csv("Teen_Mental_Health_Dataset.csv")
df.head()
print(df.describe())
#• Missing Value Analysis
df.isnull().sum()
#df["age"].fillna(df["age"].mean(),inplace=True)
#Duplicate Record Analysis
#df.dropna()
df.duplicated().sum()
#Removing Unnecessary Columns
#df.drop("depression_label",axis=1,inplace=True)
#Seaborn bar plot sns.barplot() shows the average value of a numeric column for each category, with error bars for confidence intervals.
import seaborn as sns
import matplotlib.pyplot as plt

avg_data = df.groupby('anxiety_level')[['daily_social_media_hours', 'sleep_hours']].mean()

avg_data.plot(kind='bar', figsize=(8,5))
plt.title('Avg Social Media Hours & Sleep by Depression Label')
plt.ylabel('Hours')
plt.show()
# To spot trends, increases, decreases, and patterns over a sequence.
sns.lineplot(
data=df,
x="age",
y="daily_social_media_hours",
hue="platform_usage")
plt.show()
#scatterplot
#We use scatter plots in matplotlib to see the relationship between 2 numeric variables.
plt.scatter(df['daily_social_media_hours'], df['stress_level'],marker="o",color="red")
plt.title('Social Media Hours vs Stress Level')
plt.xlabel('Daily Social Media Hours')
plt.ylabel('Stress Level')
plt.show()
# Histogram shows the distribution of a single numeric variable by grouping values into bins and plotting how many data points fall in each bin.
plt.hist(df['daily_social_media_hours'], bins=10,color="green")
plt.title('Daily Social Media Hours')
plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.show()
# Box Plot shows the distribution, median, quartiles, and outliers for a numeric column.

plt.boxplot(df['academic_performance'],patch_artist=True,
            boxprops=dict(facecolor='cyan'))
plt.title('Academic Performance')
plt.show()
#A pie chart shows proportions of a whole
df['anxiety_level'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)
plt.title('anxiety level Distribution')
plt.ylabel('')
plt.show()
#correlation 
#Correlation is a statistical measure that describes the strength and direction of a relationship between two variables
corr=df.corr(numeric_only=True)
sns.heatmap(
corr,cmap="coolwarm")
plt.show()


