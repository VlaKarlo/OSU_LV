import pandas as pd
data = pd.read_csv('data_C02_emission.csv')
# provjera koliko je izostalih vrijednosti po svakom stupcu DataFramea
print ( data . isnull () . sum () )
# brisanje redova gdje barem vrijednost jedne velicine nedosta je
data . dropna ( axis = 0 )
# brisanje stupaca gdje barem jedna vrijednost nedostaje
data . dropna ( axis = 1 )
# brisanje dupliciranih redova
data . drop_duplicates ()
# kada se obrisu pojedini redovi potrebno je resetirati indekse retka
data = data . reset_index ( drop = True )