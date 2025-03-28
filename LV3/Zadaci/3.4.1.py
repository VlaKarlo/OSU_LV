import pandas as pd
import numpy as np
data = pd.read_csv('data_C02_emission.csv')

# ---------------------------------------- A ----------------------------------------

print(len(data))
print(data.dtypes)
print(data.duplicated().any())
data.drop_duplicates()
print(data.isnull().any().any())
data.dropna()

for col in ['Make','Model','Vehicle Class','Transmission','Fuel Type']:
    data[col] = data[col].astype('category')

print(data.dtypes)

# ---------------------------------------- B ----------------------------------------

sortedData = data.sort_values('Fuel Consumption City (L/100km)',ascending=False)
print('\n')
print(data.sort_values('Fuel Consumption City (L/100km)',ascending=False)[['Make','Model','Fuel Consumption City (L/100km)']].head(3))
print('\n')
print(data.sort_values('Fuel Consumption City (L/100km)',ascending=True)[['Make','Model','Fuel Consumption City (L/100km)']].head(3))

# ---------------------------------------- C ----------------------------------------

print('\n')
print(len(data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]))
print('\n')
print(data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]['CO2 Emissions (g/km)'].mean())

# ---------------------------------------- D ----------------------------------------

print('\n')
print(len(data[(data['Make'] == 'Audi')]))
print('\n')
print(data[(data['Make'] == 'Audi') & (data['Cylinders'] == 4)]['CO2 Emissions (g/km)'].mean())

# ---------------------------------------- E ----------------------------------------

numberOfCylinders = [3,4,5,6,8,10,12,16]

print('\n')
for number in numberOfCylinders:
    print(len(data[(data['Cylinders'] == number)]))
    print(data[(data['Cylinders'] == number)]['CO2 Emissions (g/km)'].mean())
    print('\n')

# ---------------------------------------- F ----------------------------------------

print(data[(data['Fuel Type'] == 'Z')]['Fuel Consumption City (L/100km)'].mean())
print(data[(data['Fuel Type'] == 'X')]['Fuel Consumption City (L/100km)'].mean())
print('\n')
print(data[(data['Fuel Type'] == 'Z')]['Fuel Consumption City (L/100km)'].median())
print(data[(data['Fuel Type'] == 'X')]['Fuel Consumption City (L/100km)'].median())

# ---------------------------------------- G ----------------------------------------

print('\n')
print(data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'Z')].sort_values('Fuel Consumption City (L/100km)',ascending=False).head(1))

# ---------------------------------------- H ----------------------------------------

print('\n')
print(len(data[data['Transmission'].str.startswith('M')]))

# ---------------------------------------- I ----------------------------------------

print('\n')
print(data[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','CO2 Emissions (g/km)']].corr())