import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv',delimiter=',')
data = np.delete(data,(0),axis=0)

# -------------------- a --------------------

print(f"Na {len(data)} osoba je izvršeno mjerenje") 

# -------------------- b --------------------

x = data[:,1]
y = data[:,2]

plt.scatter (x , y , marker = "."  )
plt.xlabel ( 'Visina')
plt.ylabel ( 'Masa' )
plt.title ( 'Odnos visine i mase')
plt.show ()

# -------------------- c --------------------

plt.scatter (x[::50] , y[::50] , marker = ".", )
plt.xlabel ( 'Visina')
plt.ylabel ( 'Masa' )
plt.title ( 'Odnos visine i mase')
plt.show ()

# -------------------- d --------------------

print(f"Maksimalna vrijednost visine: {np.max(x)}")
print(f"Minimalna vrijednost visine: {np.min(x)}")
print(f"Prosjecna vrijednost visine: {np.mean(x)}")

# -------------------- e --------------------

visinaMuskaraca = []
visinaZena = []

for row in data:
    if row[0] == 1.0:
        visinaMuskaraca.append(row[1])
    else:
        visinaZena.append(row[1])

# visinaMuskaraca = [row[1] for row in data if row[0] == 1.0]
# visinaZena = [row[1] for row in data if row[0] == 0.0]

print(f"Maksimalna vrijednost visine muskaraca: {np.max(visinaMuskaraca)}")
print(f"Minimalna vrijednost visine muskaraca: {np.min(visinaMuskaraca)}")
print(f"Prosjecna vrijednost visine muskaraca: {np.mean(visinaMuskaraca)}")

print(f"Maksimalna vrijednost visine zena: {np.max(visinaZena)}")
print(f"Minimalna vrijednost visine zena: {np.min(visinaZena)}")
print(f"Prosjecna vrijednost visine zena: {np.mean(visinaZena)}")