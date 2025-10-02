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

### 11_w1_11 ###                     
# Pivot Table - Tutorial #
# yt - https://www.youtube.com/watch?v=Pcwf-IIgbu0 #

# A pivot table is a powerful data summarization tool used in data analysis. It allows you to reorganize and summarize tabular data in a flexible manner, providing a compact representation of complex data relationships. 
# Pivot tables enable you to aggregate and visualize data based on one or more key variables, making it easier to identify patterns and trends within your dataset.

# Create a DataFrame
data = {
    'Date': ['2022-01-01', '2022-01-01', '2022-01-02', '2022-01-02'],
    'Category': ['A', 'B', 'A', 'B'],
    'Value': [10, 20, 30, 40]
}
df = pd.DataFrame(data)

# Create a pivot table
pivot_table = df.pivot_table(values='Value', index='Date', columns='Category', aggfunc='sum')  # ***
print(pivot_table)

data = {}
np.random.seed(2)
for i in [chr(x) for x in range(65,70)]:
  data['col'+i] = np.random.randint(1,100,10)
data['orderID'] = np.random.choice(['A', 'B', 'C'], 10)
data['product'] = np.random.choice(['Product1', 'Product2', 'Product3'], 10)
data['customer'] = np.random.choice(['Customer1', 'Customer2', 'Customer3', 'Customer4'], 10)
df = pd.DataFrame(data)


pivot_table2 = df.pivot_table(values='colA', index='orderID', columns='product', aggfunc=np.mean)  # ***
print(pivot_table2.to_string())

# Aggregation Functions You can specify different aggregation functions when creating a pivot table. Common aggregation functions include 'sum', 'mean', 'count', 'max', and 'min'. For example:

pivot_table3 = df.pivot_table(values='colA', index='orderID', columns='product', aggfunc='max')
pivot_table3
pivot_table = df.pivot_table(values='Value', index='Date', columns='Category', aggfunc='max')
pivot_table

# Multi-level Pivot Tables
pivot_table = df.pivot_table(values='Value', index=['Date', 'Category'], aggfunc='sum')    # *** no columns check 
pivot_table
pivot_table4 = df.pivot_table(values='colA', index=['orderID','customer'], columns='product', aggfunc='max')   # *** index is list 
pivot_table4

# Handling Missing Values. You can specify how missing values are handled using the fill_value parameter:
pivot_table5 = df.pivot_table(values='colA', index='orderID', columns='product', aggfunc=np.mean, fill_value=0)    # ***
pivot_table5
pivot_table6 = df.pivot_table(values='colA', index=['orderID','customer'], columns='product', aggfunc='max', fill_value = 0)
pivot_table6

### 12_w1_12 ###                     
# Merge and join Methods - Tutorial #
# yt - https://www.youtube.com/watch?v=qx2QZWlEuFk #

data1 = {
    "ID" : [10001,20002,30003,40004,50005],
    "Numbers": [10,20,20,40,50],
    "Letters":["A","B","C","D","E"]
}

df1 = pd.DataFrame(data1)

data2 = {
    "ID" : [10001,20002,30003,60006,70007],
    "Numbers": [10,20,30,40,60],
    "City":["Lucknow","Munnar","Chennai","Delhi","Jaipur"]
}

df2 = pd.DataFrame(data2)

merge_data = df1.merge(df2, how="inner")      # ***

merge_data2 = df1.merge(df2, how="inner",on="ID")  # ***

merge_data2 = df2.merge(df1, how="inner",on="ID")
merge_data2

merge_data2 = df1.merge(df2, how="inner",on=["ID","Numbers"])      # ***
merge_data2

merge_data3 = df1.merge(df2, how='outer')  # ***

merge_data3 = df1.merge(df2,how='outer', on='ID') #  ***
merge_data3

left_merge = pd.merge(df1,df2,how="left")  # ***

right_merge = pd.merge(df1,df2,how="right")  # ***
right_merge

cross_merge = df1.merge(df2, how='cross')  # ***

# Questions to practice #
# Q1) Merge two tables given in below code(employees and departments) so you only get employees whose departments exist in the department list.
# Which employee(s) get left out? Why might that happen in a real company database?
# Run the below code to get the two dataframes called employees and departments
# Use these two dataframes to perform actions asked in below questions

# this given below is the code to generate two dataframes which is to use in exercise, PLEASE DO NOT CHANGE THIS CODE
employees = pd.DataFrame({
    "emp_id": [101, 102, 103, 104],
    "name": ["Alice", "Bob", "Charlie", "David"],
    "dept_id": [10, 20, 30, 40]
})

departments = pd.DataFrame({
    "dept_id": [10, 20, 30],
    "dept_name": ["HR", "Finance", "IT"]
})

# Q2) A university maintains two records: one for students and another for courses offered. Some students mistakenly got registered for courses that don’t exist.
# Merge so that all students appear, regardless of whether their course exists.
# Show the students whose course_id doesn’t match any real course.
# Replace missing courses with "Not Assigned" so it looks cleaner for reporting.

#--------USE THESE TWO DATAFRAMES TWO ANSWER QUESTIONS 2---------
students = pd.DataFrame({
    "roll_no": [1, 2, 3, 4],
    "student_name": ["John", "Emma", "Liam", "Sophia"],
    "course_id": [101, 102, 103, 104]
})

courses = pd.DataFrame({
    "course_id": [101, 102, 105],
    "course_name": ["Math", "Physics", "Biology"]
})


# Q3) An e-commerce store has sales records for two years. The manager wants to check which orders belong to:
# only 2024,
# only 2025,
# both years.
# Perform a full outer join on order_id.
# Add an indicator column to see which year(s) each order belongs to.
# Identify the loyal customers whose orders appear in both years.

#--------USE THESE TWO DATAFRAMES TWO ANSWER QUESTIONS 3---------
sales_2024 = pd.DataFrame({
    "order_id": [1, 2, 3],
    "product": ["Laptop", "Phone", "Tablet"]
})

sales_2025 = pd.DataFrame({
    "order_id": [3, 4, 5],
    "product": ["Tablet", "Monitor", "Headphones"]
})

# Q4) A store keeps two logs: orders placed and payments received. Sometimes payments don’t match the right customer due to ID mismatches.
# Merge using both order_id and customer_id.
# Find which order(s) failed to match correctly due to customer_id mismatches.
# What happens if you merge only on order_id? Why could that be risky in real financial systems?

#--------USE THESE TWO DATAFRAMES TWO ANSWER QUESTIONS 4---------
orders = pd.DataFrame({
    "order_id": [1, 2, 3, 4],
    "customer_id": [11, 12, 11, 13],
    "product": ["Book", "Pen", "Notebook", "Pencil"]
})

payments = pd.DataFrame({
    "order_id": [1, 2, 3, 4],
    "customer_id": [11, 12, 14, 13],
    "payment_status": ["Paid", "Pending", "Paid", "Pending"]
})

# Q5)A teacher has two exam score sheets for the same group of students. Some students only appeared for one exam.
# Merge on id using an outer join.
# Use suffixes=('_exam1', '_exam2') to clearly distinguish the two scores.
# Identify which students improved in the second exam.

#--------USE THESE TWO DATAFRAMES TWO ANSWER QUESTIONS 4---------
df1 = pd.DataFrame({
    "id": [1, 2, 3],
    "score": [85, 90, 78]
})

df2 = pd.DataFrame({
    "id": [2, 3, 4],
    "score": [88, 92, 80]
})

### 13_w1_13 ###                     
# Matplotlib - Tutorial #
# yt - https://www.youtube.com/watch?v=d8OfmTzLSAY #
# colab - https://colab.research.google.com/drive/1UAuz1BniInjFXOAV6av7rzL3mXQfUlE2?usp=sharing#scrollTo=JdkcKninkhTz #

# Matplotlib is a powerful and widely-used library in Python for creating static, interactive, and animated visualizations.

# This video extensively covers :

# Line Plots
# Scatter Plots
# Histogram
# Bar Chart
# Pie Chart
# Box Plot
# supplementary content :

# 3D Plotting
# Animating Plots

# Line Plots

from matplotlib import pyplot as plt

# sample_data_points : (2,1),(5,6),(10,8)
X=[2, 5, 10]
Y=[1, 6, 8]

#create the Plot
plt.plot(X,Y,'ko--')  # ***

#Display the Plot
plt.show()

import matplotlib.pyplot as plt

# DataSet of a quadratic function y1=x^2
x = [0, 1, 2, 3, 4, 5]
y1 = [0, 1, 4, 9, 16, 25]

# DataSet of a Cubic function y2=x^3
y2=[0, 1, 8, 27, 64, 125]


# Create a line plot
plt.plot(x, y1, marker='o', linestyle='--', color='b', label='y = x^2')  # ***
plt.plot(x, y2, marker='o', linestyle='--', color='k', label='y = x^3')

# labels and a title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot')

# Add a legend
plt.legend()  # ***

#Add Gridlines
plt.grid(True,c='k')  # ***

# Display the plot
plt.show()  # ***

# Scatter Plot

import numpy as np
x=np.random.randint(100,size=100)
print(x)

import numpy as np

#random generation of Dataset
x=np.random.randint(100,size=100)
y=np.random.randint(100,size=100)

# Creating a Scatter plot
plt.scatter(x,y,marker='o',s=30)

#Add labels and Title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')

#Show the plot
plt.show()

# Histogram

x=np.random.normal(20,1,100)
print(x)

#Random normal distribution with mean=20 and Std=5
x=np.random.normal(20,5,1000)

#plotting the histogram
plt.hist(x,bins=13)      # ***

# Add title
plt.title('Histogram',fontsize=25)  # ***

# Display the plot
plt.show()

# Bar Chart

# Sample data
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(100,size=5)  # ***

# Create a bar chart
plt.bar(categories, values, color='skyblue',width=0.7)  # ***

# Add labels and title
plt.xlabel('Categories')  # ***
plt.ylabel('Values')      # ***
plt.title('Bar Chart')    # ***

# Display the plot
plt.show()

# Pie chart
# Sample data
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(100,size=5)

# Create a Pie chart
plt.pie(values, labels=categories,explode=[0,0.1,0,0,0],autopct="%.1f%%",startangle=90,shadow=True)  # ***

# Add Title
plt.title('Bar Chart')

# Display the plot
plt.show()

# Box Plot
import matplotlib.pyplot as plt

# Sample data
x=np.random.normal(200, 8, 1000)  # ***

#Five Number summary  # ***
IQR=np.quantile(x, 0.75) - np.quantile(x, 0.25)
print('Lower Fence : ', np.quantile(x, 0.25)-(1.5*IQR))
print('Q1 : ',np.quantile(x, 0.25))
print('Median : ',np.median(x))
print('Q3 : ',np.quantile(x, 0.75))
print('Upper Fence : ',np.quantile(x, 0.75)+(1.5*IQR))
print('\n')


# Create a Pie chart
plt.boxplot(x)  # ***

# Add Title
plt.title('BOX Plot')
plt.yticks([np.quantile(x, 0.25)-(1.5*IQR), np.quantile(x, 0.25), np.median(x), np.quantile(x, 0.75), np.quantile(x, 0.75)+(1.5*IQR)])    # *** 

# Display the plot
plt.show()

# Subploting

#Creating the dataset
x=np.arange(100)

P1=np.sin(x)
P2=np.cos(x)

#1st subplot
plt.subplot(221)
plt.plot(P1)
plt.title('sine function')

#2nd Subplot
plt.subplot(222)
plt.plot(P2)
plt.title('cosine function')

# Display the subplot
plt.show()

#spliting the figure into subparts
fig , axis = plt.subplots(2,2)

# Plotting through axis
axis[0, 0].plot(np.sin(np.arange(100)))
axis[0, 1].hist(np.random.normal(20,5,1000))
axis[1, 0].bar(categories, values)
axis[1, 1].plot(np.log(np.arange(100)))

# Display the plots
plt.show()

# Subploting
#Creating the dataset
x=np.arange(100)  # ***

P1=np.sin(x)
P2=np.cos(x)

#1st subplot
plt.subplot(221)  # ***
plt.plot(P1)  # ***
plt.title('sine function')

#2nd Subplot
plt.subplot(222)  # ***
plt.plot(P2)  # ***
plt.title('cosine function')

# Display the subplot
plt.show()

#spliting the figure into subparts
fig , axis = plt.subplots(2,2)  # ***

# Plotting through axis  # ***
axis[0, 0].plot(np.sin(np.arange(100)))
axis[0, 1].hist(np.random.normal(20,5,1000))
axis[1, 0].bar(categories, values)
axis[1, 1].plot(np.log(np.arange(100)))

# Display the plots
plt.show()

# 3D PLOTTING
# Defining the axis as 3D
ax=plt.axes(projection='3d')  # ***

# Dataset generation
X=np.random.randint(100,size=100)
Y=np.random.randint(100,size=100)
Z=np.random.randint(100,size=100)

#Plotting the dataset
ax.scatter(X,Y,Z)  # ***

#Display the plot
plt.show()

# Animating the plots
# Number of tosses
n_tosses = 500
# Initialize the counts of heads and total tosses
heads_count = 0
tosses = np.arange(1, n_tosses + 1)  # ***

# Create a figure for plotting
fig, ax = plt.subplots()   # ***
ax.set_xlim(1, n_tosses)   # ***
ax.set_ylim(0, 1)          # ***
line, = ax.plot([], [], color='blue')  # ***

# Labels and title
ax.set_xlabel('Number of Tosses')
ax.set_ylabel('Proportion of Heads')
ax.set_title('Coin Toss Simulation')

# List to store the proportion of heads over time
proportion_heads = []

# Simulate the coin toss and update the plot
for i in range(1, n_tosses + 1):
    # Simulate a coin toss (1 = heads, 0 = tails)
    toss = np.random.randint(0, 2)  # ***
    if toss == 1:
        heads_count += 1

    # Calculate the proportion of heads
    proportion_heads.append(heads_count / i)  # ***

    # Update the line data
    line.set_data(tosses[:i], proportion_heads)  # ***

    # Redraw the plot
    plt.pause(0.01)  # ***

# Show the final plot
plt.show()

### 1_w1_ ###                     
# Seaborn - Tutorial #
# yt - https://www.youtube.com/watch?v=gD3x3-XTuXQ #
# colab - https://colab.research.google.com/drive/1V6sCQPC3jW0fSZ-kdXV58eNNKRmu9bqO?usp=sharing #

import seaborn as sns
from copy import deepcopy  # ***

#reading the  data
Data=pd.read_csv('/content/train.csv')

# Extracting the columns
exploration_set=deepcopy(Data[['ID','RecipeNumber','RecipeCode','UserReputation','ThumbsUpCount', 'ThumbsDownCount','BestScore','Rating']])  # ***

# Creating a correlation matrix
corr_matrix=exploration_set.corr(method='pearson')  # ***

# Creating  a heatmap
sns.heatmap(corr_matrix,annot=True)  # ***

sns.scatterplot(x=Data['RecipeNumber'],y=Data['RecipeCode'], hue=Data['Rating'])  # ***

sns.scatterplot(x=Data['ThumbsUpCount'],y=Data['BestScore'], hue=Data['Rating'])  # ***

Data.hist(bins=50,figsize=(15,15))  # ***

sns.pairplot(Data, diag_kind="hist") # ***





















