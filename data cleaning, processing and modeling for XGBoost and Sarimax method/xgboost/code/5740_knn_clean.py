import pandas as pd

# Load the uploaded CSV file
file_path = 'uscounties.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head()
from sklearn.neighbors import NearestNeighbors

# Extract relevant columns for the calculation (latitude and longitude)
coordinates = data[['lat', 'lng']]

# Fit the NearestNeighbors model to find the nearest 4 neighbors
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')  # 5 includes the point itself
knn.fit(coordinates)

# Find the 4 nearest neighbors for each row
distances, indices = knn.kneighbors(coordinates)

# Indices of nearest neighbors (excluding the row itself)
nearest_4_indices = indices[:, 1:5]

# Append nearest 4 neighbors to the dataset
for i in range(4):
    data[f'neighbor_{i+1}'] = [data.iloc[idx]['county'] for idx in nearest_4_indices[:, i]]


data.to_csv('Nearest Neighbors Added',index=False)
