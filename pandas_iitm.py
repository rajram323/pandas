########### week 1 ###########
gfg - https://www.geeksforgeeks.org/c/output-of-a-program-set-1/

### 6_w1_6 ###
#  Apply Statistics using Pandas # 
# pandas, mean, median, mode, describe #
import pandas as pd

cities = ['Mumbai', 'Chennai', 'Pune', 'Ahmedabad', 'Kolkata', 'Kanpur', 'Delhi']
city_df = pd.DataFrame(cities)    # list to dataframe  ***

city_df.columns = ['City_Name']   # change column name

condition_met = city_df.City_Name == 'Mumbai'  # ***
type(condition_met)    # series
condition_met          # true false list
city_df[condition_met] # give data

city_df[city_df.City_Name == 'Pune']    # ***

  # Aggregation and grouping
random_state = np.random.RandomState(100)         # ***
random_series = pd.Series(random_state.rand(10))  # ***

random_series.mean()
random_series.std()
random_series.sum()

df = pd.DataFrame({'A': random_state.rand(5),
                   'B': random_state.rand(5)})

df.sum()
df.std()
df.mean()

df.mean(axis=1)  # operations row-wise
df.sum(axis=1)

#Groupby
# Three stages

# Split - we split dataframe into multiple smaller dataframe based on the values of keys
# Apply - we apply desired aggregation/transformation on each dataframe.
# Combine - we combine results from apply state into a dataframe

df = pd.DataFrame({'key': ['A','B','C']*2, #list("ABCABC"), ['A','B','C','A','B','C']
                   'data': range(6)})

df.groupby("key")  # <pandas.core.groupby.generic.DataFrameGroupBy object at 0x78e73c1c8040>
df.groupby("key").sum()  # ***
