# Load libraries
import pandas
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import seaborn as sns

# Load dataset
url = "data_folder/modified_HR_data.csv"
names = ['satisfaction_level','last_evaluation','number_project',
         'average_montly_hours','time_spend_company','Work_accident',
         'promotion_last_5years','salary','left']
dataset = pandas.read_csv(url)

# correlation analysis
correlation = dataset.corr()
print(correlation)
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title('Correlation between different fearures')

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('left').size())

# histograms
dataset.hist()
plt.show()

