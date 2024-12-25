#add neibour population
import pandas as pd

# Load the uploaded files
nearest_neighbors_file = 'Nearest_Neighbors_Added.csv'
covid_data_file = 'clean_covid_case.csv'

# Load data into dataframes
nearest_neighbors_df = pd.read_csv(nearest_neighbors_file)
covid_data_df = pd.read_csv(covid_data_file)

# Display the first few rows of both datasets to understand their structure
nearest_neighbors_df.head(), covid_data_df.head()




# Merge neighbor population information into covid_data_df
# First, prepare a mapping of county populations for quick lookup
fips_population_mapping = nearest_neighbors_df.set_index('county_ascii')['population'].to_dict()

# Function to calculate the sum of neighbor populations for a row
def calculate_neighbor_population(row, neighbor_columns, population_mapping):
    total_population = 0
    for neighbor in neighbor_columns:
        neighbor_name = row[neighbor]
        if pd.notna(neighbor_name):  # Check if neighbor name is valid
            total_population += population_mapping.get(neighbor_name, 0)  # Add neighbor population if it exists
    return total_population

# Apply the function to the nearest_neighbors_df to calculate the total neighbor population
neighbor_columns = ['neighbor_1', 'neighbor_2', 'neighbor_3', 'neighbor_4']
nearest_neighbors_df['neighbor_population_sum'] = nearest_neighbors_df.apply(
    calculate_neighbor_population, axis=1, neighbor_columns=neighbor_columns, population_mapping=fips_population_mapping
)

# Merge the neighbor_population_sum into the COVID data using county name
covid_data_df = covid_data_df.merge(
    nearest_neighbors_df[['county_ascii', 'neighbor_population_sum']],
    left_on='county_x',
    right_on='county_ascii',
    how='left'
)

# Drop unnecessary duplicate column after merge
covid_data_df.drop(columns=['county_ascii'], inplace=True)

# Drop the specified columns
columns_to_drop = ['state', 'lat', 'lng']
covid_data_df_cleaned = covid_data_df.drop(columns=columns_to_drop)


output_file = 'covid_data_augmented.csv'
covid_data_df_cleaned.to_csv(output_file, index=False)