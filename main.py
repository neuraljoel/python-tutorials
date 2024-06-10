import pandas as pd
import mlbstatsapi
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Get the data
mlb = mlbstatsapi.Mlb()
params = {'season': 2023}
type = ['season']
groups = ['hitting']

data = []
teams = mlb.get_teams(sportid=1)

for team in teams:
    stats = mlb.get_team_stats(team_id=team.id,
                               stats=type,
                               groups=groups)['hitting']['season']
    
    for split in stats.splits:
        data.append({
            'Team': team.name,
            'Runs': split.stat.runs,
            'Avg': float(split.stat.avg)
        })

df = pd.DataFrame(data)
X = df[['Avg']]
y = df['Runs']

# Create the model
linreg = LinearRegression()
linreg.fit(X, y)

y_hat = linreg.predict(X)
alpha = linreg.intercept_
beta = linreg.coef_[0]

x_values = np.linspace( X.min().values[0], X.max().values[0], len(df) )
y_values = alpha + beta * x_values

# Create the plot
plt.scatter(df['Avg'], y, color='blue')

for i, team in enumerate(df['Team']):
    plt.annotate(team,
                 (df['Avg'][i], y[i]),
                 textcoords="offset points",
                 xytext=(0,5),
                 fontsize=6,
                 rotation=35)
    
plt.plot(x_values,
         y_values,
         color='orange',
         label=f'$runs = {alpha:.2f} + {beta:.2f} bavg$')

plt.xlabel('Batting average', fontsize=16)
plt.ylabel('Home Runs', fontsize=16)
plt.title('Regressing Home Runs on Batting Average', fontsize=18)

plt.grid()
plt.legend()
plt.show()