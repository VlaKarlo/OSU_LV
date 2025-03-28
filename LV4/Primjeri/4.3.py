from sklearn . preprocessing import OneHotEncoder
ohe = OneHotEncoder ()
X_encoded = ohe . fit_transform ( data[[ 'Fuel Type']]) . toarray ()