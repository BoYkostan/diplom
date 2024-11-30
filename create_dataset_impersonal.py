import pandas as pd
from sklearn.preprocessing import LabelEncoder


loaded_data = pd.read_excel('dataset_noname.xlsx')
print(loaded_data['full_name'])

labelencoder = LabelEncoder()
loaded_data_new = labelencoder.fit_transform(loaded_data['full_name'])
loaded_data['full_name'] = loaded_data_new
print(loaded_data['full_name'])

loaded_data.to_excel('new_dataset_impersonal.xlsx', index=False)