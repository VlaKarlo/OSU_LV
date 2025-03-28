import pandas as pd
import numpy as np
data = pd.read_csv('data_C02_emission.csv')
# izdvajanje pojedinog stupca
print ( data [ 'Cylinders'] )
print ( data . Cylinders )
# izdvajanje vise stupaca
print ( data [ [ 'Model' , 'Cylinders'] ] )
# izdvajanje redaka koristenjem iloc metode
print ( data . iloc [ 2 :6 , 2 : 7 ] )
print ( data . iloc [ : , 2 : 5 ] )
print ( data . iloc [ : , [0 ,4 , 7 ] ] )
# logicki uvjeti na pojedine stupce
print ( data . Cylinders > 6 )
print ( data [ data . Cylinders > 6 ] )
print ( data [ ( data ['Cylinders'] == 4 ) & ( data ['Engine Size (L)'] > 2.4 ) ] . Model)
# dodavanje novih stupaca
data [ 'jedinice'] = np . ones ( len ( data ) )
data [ 'large'] = ( data ['Cylinders'] > 10 )