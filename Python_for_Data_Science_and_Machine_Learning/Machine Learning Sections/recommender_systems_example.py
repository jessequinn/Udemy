import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# turn on edges
plt.rcParams["patch.force_edgecolor"] = True

column_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('./Recommender-Systems/u.data', sep='\t', names=column_names)

# print(df.head())

movie_titles = pd.read_csv('./Recommender-Systems/Movie_Id_Titles')

# print(movie_titles.head())

df = pd.merge(df, movie_titles, on='item_id')

# print(df.head())

sns.set_style('white')

# print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())

# print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

# print(ratings.head())

plt.figure(1, figsize=(10, 6))
ratings['num of ratings'].hist(bins=70)
plt.xlabel('Number of Ratings')

plt.figure(2, figsize=(10, 6))
ratings['rating'].hist(bins=70)
plt.xlabel('Ratings')

sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)

moviemat = df.pivot_table(index='user_id', columns='title', values='rating')

# print(moviemat.head())

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars = corr_starwars.join(ratings['num of ratings'])

print(corr_starwars[corr_starwars['num of ratings'] > 100].sort_values(
    'Correlation', ascending=False).head())

corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])

print(corr_liarliar[corr_liarliar['num of ratings'] > 100].sort_values(
    'Correlation', ascending=False).head())

# save plots
for i in range(1, plt.gcf().number + 1):
    plt.figure(i)
    # tighten up layout
    plt.tight_layout()
    plt.savefig('recommender_systems_figure_' + str(i) + '.png')
    plt.close()
