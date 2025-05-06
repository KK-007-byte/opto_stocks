#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("batting_stats_for_icc_mens_t20_world_cup_2024.csv")
df.head()


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

top_scorers=df.sort_values('Runs',ascending=False).head(20)
sns.barplot(x='Runs',y='Player',data=top_scorers)

plt.show("Top 20 Run Scorers")

# Correlation heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

features = [ 'SR', 'Mat', 'Ave', 'NO']
target = 'Runs'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("R2 Score:", r2_score(y_test, preds))


# In[5]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

features=['Runs','Ave','NO','SR']
X=df[features]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

sse = []
for k in range(1, 20):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    sse.append(km.inertia_)

plt.plot(range(1, 20), sse, marker='o')
plt.title("Elbow Method for Optimal k")


# In[6]:


df=pd.read_csv("batting_stats_for_icc_mens_t20_world_cup_2024.csv")
df.head()


# In[7]:


pip install plotly


# In[8]:


import matplotlib.pyplot as plt
import pandas as pd

# Load data
df = pd.read_csv('batting_stats_for_icc_mens_t20_world_cup_2024.csv')

# Pick a few players to compare
players_to_compare = ['Rahmanullah Gurbaz', 'H Klaasen', 'JC Buttler']
features = ['Runs', 'SR','HS','NO']

# Normalize data (optional but helps when features are on different scales)
df_norm = df.copy()
# Convert the 'HS' and any other relevant columns to numeric, handling errors
df_norm['HS'] = pd.to_numeric(df_norm['HS'], errors='coerce')
df_norm['Runs'] = pd.to_numeric(df_norm['Runs'], errors='coerce')
df_norm['SR'] = pd.to_numeric(df_norm['SR'], errors='coerce')
df_norm['NO'] = pd.to_numeric(df_norm['NO'], errors='coerce')
    
    # Now apply the normalization
df_norm[features] = df_norm[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Radar plot function
def radar_plot(df, players, features):
    labels = np.array(features)
    num_vars = len(labels)

    


# In[9]:


import numpy as np
def radar_plot(df, players, features):
    labels = np.array(features)
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for player in players:
        values = df[df['Player'] == player][features].values.flatten().tolist()
        values += values[:1]  # repeat first value to close the plot
        ax.plot(angles, values, label=player)
        ax.fill(angles, values, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("T20 WC 2024 Batsmen Comparison", size=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()


# In[10]:


import plotly.graph_objects as go

def interactive_radar(df, players, features):
    fig = go.Figure()

    for player in players:
        row = df[df['Player'] == player][features].values.flatten().tolist()
        fig.add_trace(go.Scatterpolar(
            r=row + [row[0]],
            theta=features + [features[0]],
            fill='toself',
            name=player
        ))

    fig.update_layout(
      polar=dict(radialaxis=dict(visible=True)),
      showlegend=True,
      title='Interactive Radar Plot of Batsmen'
    )

    fig.show()

interactive_radar(df_norm, players_to_compare, features)


# In[11]:


pip install fastapi uvicorn pandas matplotlib plotly scikit-learn


# In[31]:


# app/utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

df = pd.read_csv("batting_stats_for_icc_mens_t20_world_cup_2024.csv")

# Normalize for radar
def normalize_df(features):
    df_norm = df.copy()
    df_norm[features] = df[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df_norm

def generate_radar_image(players, features):
    df_norm = normalize_df(features)

    labels = np.array(features)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for player in players:
        values = df_norm[df_norm['Player'] == player][features].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, label=player)
        ax.fill(angles, values, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Radar Comparison")

    # Save to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_bytes = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_bytes


# In[35]:


pip install app.utils


# In[37]:


# app/main.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse


app = FastAPI(title="T20 World Cup 2024 Batting API")

@app.get("/")
def root():
    return {"message": "Welcome to the T20 World Cup Batting Stats API"}

@app.get("/players")
def get_players():
    players = df['Player'].unique().tolist()
    return {"players": players}

@app.get("/player-stats")
def player_stats(player: str):
    player_data = df[df['Player'] == player]
    if player_data.empty:
        return JSONResponse(status_code=404, content={"error": "Player not found"})
    return player_data.to_dict(orient="records")[0]


    


# In[ ]:




