import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder


# turned it into a class so its easier to use and read
class WeatherPredictionModel:
    def __init__(self, data_path):
        # preloading and setting up our dataset
        self.data = pd.read_csv(data_path)
        self._prepare_data()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _prepare_data(self):

        self.data['date'] = pd.to_datetime(self.data['date'], format='%Y%m%d')
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month

        # setting up months to seasons
        self.data['season'] = self.data['month'].apply(self._get_season)

        # error handling (many missing values in the data)
        self.data = self.data.dropna(subset=['mean_temp'])
        self.data = self.data.dropna()

        # features and target (we want to predict the mean temperature)
        self.features = ['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 
                         'min_temp', 'precipitation', 'pressure', 'snow_depth', 'season']
        self.target = 'mean_temp'

        # catgory endocding for season
        encoder = OneHotEncoder(sparse_output=False)
        encoded_seasons = encoder.fit_transform(self.data[['season']])
        encoded_season_df = pd.DataFrame(encoded_seasons, columns=encoder.get_feature_names_out(['season']))
        self.data = pd.concat([self.data, encoded_season_df], axis=1)
        self.features.extend(encoder.get_feature_names_out(['season']))
        self.features.remove('season')

        # spliting data
        X = self.data[self.features]
        y = self.data[self.target]

        # error handling (many missing y values in the data)
        X = X[~y.isna()]
        y = y.dropna()

        # taining sets 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def _get_season(self, month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    # using random forest regressor 
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        # Evaluate the model
        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"Model trained. Mean Absolute Error: {mae:.2f}, R2 Score: {r2:.2f}")

    # syntehtic data creator!
    def generate_synthetic_data(self, year):
        synthetic_data = []
        for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
            season_data = self.data[self.data['season'] == season]['mean_temp']
            if season_data.empty:
                print(f"No data available for season {season}. Skipping.")
                continue

            try:
                kde = sns.kdeplot(season_data, bw_adjust=0.5)
                kde_data = kde.get_lines()[0].get_data()
                plt.close()

                x = kde_data[0]
                y = kde_data[1]
                probabilities = y / y.sum()
                synthetic_temps = np.random.choice(x, size=365, p=probabilities)

                # generates the synthetic data for whatever year is inputted
                synthetic_season_data = pd.DataFrame({
                    'cloud_cover': self.data[self.data['season'] == season]['cloud_cover'].sample(365, replace=True).values,
                    'sunshine': self.data[self.data['season'] == season]['sunshine'].sample(365, replace=True).values,
                    'global_radiation': self.data[self.data['season'] == season]['global_radiation'].sample(365, replace=True).values,
                    'max_temp': self.data[self.data['season'] == season]['max_temp'].sample(365, replace=True).values,
                    'min_temp': self.data[self.data['season'] == season]['min_temp'].sample(365, replace=True).values,
                    'precipitation': self.data[self.data['season'] == season]['precipitation'].sample(365, replace=True).values,
                    'pressure': self.data[self.data['season'] == season]['pressure'].sample(365, replace=True).values,
                    'snow_depth': self.data[self.data['season'] == season]['snow_depth'].sample(365, replace=True).values,
                    'mean_temp': synthetic_temps,
                    'season': season
                })
                synthetic_season_data['year'] = year
                synthetic_data.append(synthetic_season_data)
            except IndexError:
                print(f"KDE failed for season {season}. Skipping.")
                continue

        if synthetic_data:
            synthetic_data = pd.concat(synthetic_data, ignore_index=True)
            return synthetic_data
        else:
            print(f"No synthetic data generated for year {year}.")
            return pd.DataFrame()

    def plot_kde(self, data, year, data_type):
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=data, x='mean_temp', hue='season', fill=True, common_norm=False, palette='Set2')
        plt.title(f'Seasonal KDE of Mean Temperature for {year} ({data_type} Data)')
        plt.xlabel('Mean Temperature (°C)')
        plt.ylabel('Density')
        plt.legend(title='Season')
        plt.grid(True)
        plt.show()

#RUN DA CLASS

# Initialize the model
model = WeatherPredictionModel('london_weather.csv')

# Train the model
model.train_model()

# Loop to allow the user to interact with the model
while True:
    user_input = input("Enter a year to generate synthetic data or type 'quit' to exit: ")
    
    if user_input.lower() == 'quit':
        break
    
    try:
        year = int(user_input)
        
        # historcial or synthetic data?
        data_option = input(f"Do you want to use historical data or generate synthetic data for {year}? (type 'historical' or 'synthetic'): ").strip().lower()
        
        if data_option == 'historical':
            # data filtering
            historical_data = model.data[model.data['year'] == year]
            
            if historical_data.empty:
                print(f"No data found for {year}.")
            else:
                print(f"Showing historical data for {year}:")
                print(historical_data.head())
                
                #KDE for historical data
                model.plot_kde(historical_data, year, 'Historical')

        elif data_option == 'synthetic':
            synthetic_data = model.generate_synthetic_data(year)
            
            if not synthetic_data.empty:
                print(f"Synthetic data generated for {year}:")
                print(synthetic_data.head())
                
                #KDE for synthetic data
                model.plot_kde(synthetic_data, year, 'Synthetic')

        else:
            print("Invalid")
    except ValueError:
        print("Invalid, enter a valid year.")
