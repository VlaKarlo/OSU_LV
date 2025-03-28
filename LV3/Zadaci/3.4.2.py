import pandas as pd
import numpy as np
import matplotlib . pyplot as plt
data = pd.read_csv('data_C02_emission.csv')

# ---------------------------------------- A ----------------------------------------

plt.figure()
data['Fuel Consumption City (L/100km)'].plot(kind='hist', bins=20)
plt.show()

# ---------------------------------------- B ----------------------------------------

plt.figure()
data.plot.scatter(
    x='Fuel Consumption City (L/100km)',
    y='CO2 Emissions (g/km)',
    )
plt.show()

# ---------------------------------------- C ----------------------------------------

plt.figure()
data.groupby('Fuel Type').boxplot(column='CO2 Emissions (g/km)', by='Fuel Type')
plt.show()


# ---------------------------------------- D ----------------------------------------

plt.figure()
data.groupby('Fuel Type').size().plot.bar()
plt.xticks(rotation=0)
plt.show()

# ---------------------------------------- E ----------------------------------------

plt.figure()
data.groupby(['Cylinders'])['CO2 Emissions (g/km)'].mean().plot.bar()
plt.xticks(rotation=0)
plt.show()