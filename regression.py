import pandas as pd
from nba_api.stats.endpoints import leagueleaders
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Get Data: Top 50 NBA scorers this season
data = leagueleaders.LeagueLeaders().get_data_frames()[0].head(50)

# 2. Setup Variables (Predict Points based on Minutes)
X = data[['MIN']].values  # Independent variable
y = data['PTS'].values    # Dependent variable

# 3. Run the Regression
model = LinearRegression()
model.fit(X, y)

# 4. Print the "Secret Sauce" (The Coefficients)
print(f"For every 1 extra minute played, a player is predicted to score {model.coef_[0]:.2f} more points.")

# 5. Visualize it
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title('Minutes vs Points (Top 50 Scorers)')
plt.xlabel('Minutes Played')
plt.ylabel('Points Scored')
plt.savefig('regression_plot.png') # This saves the chart to your folder
print("Success! Your plot is saved as regression_plot.png")