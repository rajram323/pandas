########### week 1 ###########


### 6_w1_6 ###
#  Apply Statistics using Pandas # 
# pandas, mean, median, mode, describe #
# yt - #
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

### 7_w1_7 ###                
# Pandas Resources
# https://github.com/ajcr/100-pandas-puzzles.git
# https://github.com/guipsamora/pandas_exercises.git
# https://colab.research.google.com/drive/1oFNFTzDe84WhQuTWzw8GqN-CBwC9i-wb?usp=sharing   


### 8_w1_8 ###                     
# Apply and Map - Tutorial #
# yt - https://www.youtube.com/watch?v=n8Nd5qDit50 #
# The apply() method is one of the most common methods of data preprocessing. It simplifies applying a function on each element in a pandas Series and each row or column in a pandas DataFrame.
# series form the basis of pandas. They are just one-dimensional arrays with axis labels called indices.                   
# There are different ways of creating a Series object (e.g., we can initialize a Series with lists or dictionaries). Let’s define a Series object with two lists containing student names as indices and their heights in centimeters as data:

from IPython.display import display  # ***

students = pd.Series(data=[180, 175, 168, 190],      # ***
                     index=['A', 'B', 'C', 'D'])

display(students)      # decorated view
print(type(students))  # series
print(students)

# The data type of the students object is Series, so we can apply any functions on its data using the apply() method. 
def cm_to_feet(h):
    return np.round(h/30.48, 2)

print(students.apply(cm_to_feet))    # for series directly apply ***

# Ex - 1
data1 = pd.DataFrame({'EmployeeName': ['Callen Dunkley', 'Sarah Rayner', 'Jeanette Sloan', 'Kaycee Acosta', 'Henri Conroy', 'Emma Peralta', 'Martin Butt', 'Alex Jensen', 'Kim Howarth', 'Jane Burnett'],
                    'Department': ['Accounting', 'Engineering', 'Engineering', 'HR', 'HR', 'HR', 'Data Science', 'Data Science', 'Accounting', 'Data Science'],
                    'HireDate': [2010, 2018, 2012, 2014, 2014, 2018, 2020, 2018, 2020, 2012],
                    'Sex': ['M', 'F', 'F', 'F', 'M', 'F', 'M', 'M', 'M', 'F'],
                    'Birthdate': ['04/09/1982', '14/04/1981', '06/05/1997', '08/01/1986', '10/10/1988', '12/11/1992', '10/04/1991', '16/07/1995', '08/10/1992', '11/10/1979'],
                    'Weight': [78, 80, 66, 67, 90, 57, 115, 87, 95, 57],
                    'Height': [176, 160, 169, 157, 185, 164, 195, 180, 174, 165],
                    'Kids': [2, 1, 0, 1, 1, 0, 2, 0, 3, 1]
                    })
display(data1)

data1['FirstName'] = data1['EmployeeName'].apply(lambda x : x.split()[0])    # lambda func and create new column   ***
data1['LastName'] = data1['EmployeeName'].apply(lambda x : x.split()[1])
display(data1)

# Ex - 2
# Define data as Series
airline_series = pd.Series(['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Vistara'])
duration_series = pd.Series(['2h 5m', '2h', '50m', '1h 30m', '3h 20m'])
arrival_time_series = pd.Series(['21:45', '10:00', '00:50', '15:30', '19:15'])
departure_time_series = pd.Series(['19:40', '07:55', '00:44', '14:00', '15:55'])
distance_series = pd.Series([2050, 2000, 220, 800, 3000])

# Create DataFrame
df = pd.DataFrame({
    'Airline': airline_series,
    'Duration': duration_series,
    'Arrival Time': arrival_time_series,
    'Departure Time': departure_time_series,
    'Distance (km)': distance_series
})
# Create DataFrame
data = pd.DataFrame(df)    # may be not required  ***

# Display DataFrame
data          # decorated view  ***
print(df)     # normal view 
display(df)   # decorated view

def min_sec(s):
  sec = 0
  a = s.split() # ["2h", "30m"]
  for i in range(len(a)):
    if "h" in a[i]:                # use of in **
     sec += int(a[i][:-1])*60*60   # use of int ** 
    if "m" in a[i]:
      sec += int(a[i][:-1])*60
  return sec

data["new_duration"] = data['Duration'].apply(min_sec)

def hr(time):
  time = time.split(':')      # split by ***
  hour = int(time[0])
  if 5 <= hour and hour < 12:
    return 'Morning'
  if 12 <= hour and hour < 17:
    return 'Afternoon'
  if 17 <= hour and hour < 20:
    return 'Evening'
  if 20 <= hour or hour < 5:
    return 'Night'

data['Arrival Time'] = data['Arrival Time'].apply(hr)    # same column ***

# Define a function to map ages to age groups
def map_age(age):
    if age < 30:
        return 'Young'
    elif age >= 30 and age < 40:
        return 'Middle-aged'
    else:
        return 'Senior'

# Map values in 'Age' column using the function
df_map['Age Group'] = df_map['Age'].map(map_age)    # use of map not apply ***

### 9_w1_9 ###                     
# Concatenate - Tutorial #
# yt - https://www.youtube.com/watch?v=9AYVGYF4j_s #

# Create two sample DataFrames
temp_data = pd.DataFrame({'Date': ['12-02-2023', '13-02-2023', '14-02-2023', '15-02-2023', '16-02-2023'],
                    'TempMax': [24.3, 26.9, 23.4, 15.5, 16.1 ] })

rainfall_data = pd.DataFrame({'Date': ['12-02-2023', '13-02-2023', '14-02-2023', '15-02-2023', '16-02-2023'],
                    'Rainfall': [0, 3.6, 3.6, 39.8, 2.8 ] })

result_col = pd.concat([temp_data, rainfall_data])    # .concat take list ***
print("\nConcatenate along rows:")
print(result_col)

# Concatenate along columns (axis=1)
result_col = pd.concat([temp_data, rainfall_data], axis=1)  # axis = 1 horizontallly merge ***
print("\nConcatenate along columns:")
print(result_col)

# ignore_index
# In pandas.concat, the ignore_index parameter is a boolean value that determines whether to ignore the index labels along the concatenation axis or not.
# When ignore_index is set to True, the resulting concatenated DataFrame will have a new RangeIndex along the concatenation axis, effectively ignoring the original index labels of the input DataFrames.

# Concatenate along columns (axis=0) (ignore_index=False)
result_col_notIgnore = pd.concat([temp_data, rainfall_data], axis=0,ignore_index=False)  # *** ignore index = false
print("\nConcatenate along columns with ignore_index = False:")
print(result_col_notIgnore)

# Concatenate along columns (axis=0) (ignore_index=True)
result_col_ignore = pd.concat([temp_data, rainfall_data], axis=0, ignore_index=True)  # *** ignore index = true
print("\nConcatenate along columns with ignore_index = True:")
print(result_col_ignore)                                                              # *** notice the index of result_col dataframe new index

result_col_inner = pd.concat([temp_data, rainfall_data], axis=0, join='inner')       # *** inner common column
print("\nConcatenate with inner join:")
print(result_col_inner)

# Concatenate with outer join
result_col_outer = pd.concat([temp_data, rainfall_data], axis=0, join='outer')      # *** outer all column
print("\nConcatenate with outer join:")
print(result_col_outer)

# Creating the first DataFrame
data1 = {'Name': ['Alice', 'Bob', 'Charlie'],
         'Age': [25, 30, 35],
         'Score': [85, 90, 88]}
df1 = pd.DataFrame(data1)

# Creating the second DataFrame
data2 = {'Name': ['David', 'Eve', 'Charlie'],
         'Age': [27, 32, 35],
         'Score': [82, 88, 88],
         "extra":[100,100,100]}
df2 = pd.DataFrame(data2)

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

result_col_inner2 = pd.concat([df1, df2], axis=0, join='inner',ignore_index=True)    # inner with new index ***
result_col_inner2

result_col_inner2_1 = pd.concat([df1, df2], axis=1,ignore_index=True)        #  with new index , horizontally ***
result_col_inner2_1

result_col_key1 = pd.concat([temp_data, rainfall_data], axis=0, keys=('Delhi','Chennai'))    # *** 
print("\nConcatenate along columns with keys, axis = 0:")
print(result_col_key1)

result_col_key1.loc["Delhi"]
result_col_key1.loc["Delhi"]['TempMax'][4]

result_col_key2 = pd.concat([temp_data, rainfall_data], axis=1, keys='AB')      # *** 
print("\nConcatenate along columns with keys, axis = 1:")
result_col_key2

ax_temp_A = result_col_key2['A']['TempMax'].max()      # ***
print("Maximum temperature in section A:", max_temp_A)

result_col_key3 = pd.concat([df1, df2], axis=1, keys='AB')  # ***
print("\nConcatenate along columns with keys, axis = 1:")
result_col_key3

min_score_A3 = result_col_key3['A']['Score'].min()
print("Maximum temperature in section A:", min_score_A3)

min_score_B3 = result_col_key3['B']['Score'].min()
print("Maximum temperature in section B:", min_score_B3)

### 10_w1_10 ###                     
# Tutorial Compare #
# yt - https://www.youtube.com/watch?v=1qGFXfdEjeo #

# Pandas, the compare() function provides a way to compare two DataFrame objects and generate a DataFrame highlighting the differences between them. 
# This can be particularly useful when you have two datasets and want to identify discrepancies or changes between them

# DataFrame.compare(other, align_axis=1, keep_shape=False, keep_equal=False)
# So, let’s understand each of its parameters –

# other : This is the first parameter which actually takes the DataFrame object to be compared with the present DataFrame.

# align_axis : It deals with the axis(vertical / horizontal) where the comparison is to be made(by default False).0 or index : Here the output of the differences are presented vertically, 1 or columns : The output of the differences are displayed horizontally.

# keep_shape : It means that whether we want all the data values to be displayed in the output or only the ones with distinct value. It is of bool type and the default value for it is “false”, i.e. it displays all the values in the table by default.

# keep_equal : This is mainly for displaying same or equal values in the output when set to True. If it is made false then it will display the equal values as NANs.


df = pd.DataFrame(
    {
        "col1": ["a", "a", "b", "b", "a"],
        "col2": [1.0, 2.0, 3.0, np.nan, 5.0],
        "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
    },
    columns=["col1", "col2", "col3"],
)

print(df)

df2 = df.copy()
df.compare(df2)                # ***

df2.loc[0, "col1"] = "c"        # ***
df2.loc[2, "col3"] = 4.0

df.compare(df2)

df.compare(df2, align_axis = 0)    # ***

df3 = df.copy()
df3.loc[0,"col1"] = "c"
df3.loc[1,"col2"] = 100
df3.loc[2,"col3"] = 4.0

df.compare(df3)

# Comparing various columns instead of whole dataframe
df['col2'].equals(df2['col2'])    # ***

# Comparing elements of two different columns
output = pd.Series(df['col2'] == df2['col2'])

df.loc[output]



