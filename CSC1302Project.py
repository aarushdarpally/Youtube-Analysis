print('1. We chose Spotify and Youtube dataset.')       
# Link: https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube
print()

print('2. Data preprocessing.')
print()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Spotify_Youtube.csv')     # You will need to download the csv file from the link above first.
print('This is the dataset BEFORE the cleaning part.')
print()
print(df)
print()

print('This is the information of the dataset BEFORE the cleaning part.')
print()
print(df.info())
print()
print(df.describe())
print()

# Remove unused columns.
df = df.drop(['Unnamed: 0', 'Url_spotify', 'Uri', 'Url_youtube', 'Description'], axis=1)
# Fill missing values with the mean of the columns.
df = df.fillna(df.mean())
# Remove duplicates.
df= df.drop_duplicates()

print('This is the dataset AFTER the cleaning part.')
print()
print(df)
print()

print('This is the information of the dataset AFTER the cleaning part.')
print()
print(df.info())
print()
print(df.describe())
print()

print('3. Calculate some statistics.')
print()


'''
    Calculate and print mean, median, variance, and standard deviation of Energy column. 
    Then, describe what you find.
'''

print('Energy:')

mean2 = df['Energy'].mean()
median2 = df['Energy'].median()
variance2 = df['Energy'].var()
std_deviation2 = df['Energy'].std()

print(f'Mean of Energy column is: {mean2: .4f}')
print(f'Median of Energy column is: {median2: .4f}')
print(f'Variance of Energy column is: {variance2: .4f}')
print(f'Standard deviation of Energy column is: {std_deviation2: .4f}')
print()


'''
    Calculate and print mean, median, variance, and standard deviation of Key column. 
    Then, describe what you find.
'''

print('Key:')

mean3 = df['Key'].mean()
median3 = df['Key'].median()
variance3 = df['Key'].var()
std_deviation3 = df['Key'].std()

print(f'Mean of Key column is: {mean3: .4f}')
print(f'Median of Key column is: {median3: .4f}')
print(f'Variance of Key column is: {variance3: .4f}')
print(f'Standard deviation of Key column is: {std_deviation3: .4f}')
print()

print('4. Visualize the data.')
print()

'''
    1st chart: Bar chart (only 1 horizontal bar at 1 item): 
    Using Views and Stream columns: 
    Top Ten Songs By Most Views on YouTube and Stream on Spotify.
'''
print('Top Ten Songs By Most Views on YouTube and Stream on Spotify:')

grouped = df.groupby(['Views','Stream'], as_index=False)['Track'].sum()

# Sort top ten songs by views and stream.
views = grouped.sort_values('Views', ascending=False)[:10]
stream = grouped.sort_values('Stream', ascending=False)[:10]

# Create two subplots.
fig, axs = plt.subplots(2, 1, figsize=(12,6))

x1 = views['Track'].to_numpy()
y1 = views['Views'].to_numpy()

x2 = stream['Track'].to_numpy()
y2 = stream['Stream'].to_numpy()

# Create horizontal bars.
axs[0].barh(x1, y1, color = 'g', height=0.7)
axs[1].barh(x2, y2, color = 'b', height=0.7)

# Set titles for each subplot.
axs[0].set_xlabel('Views (billions) on YouTube', fontdict={'fontsize': 12, 'fontweight': 'bold'})
axs[1].set_xlabel('Stream (billions) on Spotify', fontdict={'fontsize': 12, 'fontweight': 'bold'})

# Invert y-axis on each subplot.
axs[0].invert_yaxis()
axs[1].invert_yaxis()

# Adjust the spacing between the subplots.
plt.subplots_adjust(hspace=0.5)

fig.suptitle('Top Ten Songs By Most Views on YouTube and Stream on Spotify', fontsize=16, fontweight='bold')

plt.show()
print()

'''
    2nd chart: Bar chart (3 vertical bars at 1 item): 
    Using Danceability, Energy, and Valence columns: 
    Danceability, Energy, and Valence of Top Ten Songs By Stream.
'''
print('Danceability, Energy, and Valence of Top Ten Songs By Stream:')

grouped = df.groupby(['Danceability', 'Energy', 'Valence', 'Track'], as_index=False)['Stream'].sum()

# Sort top ten songs by stream.
stream = grouped.sort_values('Stream', ascending=False, ignore_index=True)[:10]

# Create a vertical bar chart.
fig, ax = plt.subplots(figsize=(12,6))
ax.bar(stream.index, stream['Danceability'], width=0.3, label='Danceability')
ax.bar(stream.index+0.3, stream['Energy'], width=0.3, label='Energy')
ax.bar(stream.index+0.6, stream['Valence'], width=0.3, label='Valence')

ax.set_title('Danceability, Energy, and Valence of Top Ten Songs By Stream', fontsize=18, fontweight='bold')
ax.set_xlabel('Track', fontsize=14, fontweight='bold')
ax.set_ylabel('Value', fontsize=14, fontweight='bold')

# Set the x-tick labels.
ax.set_xticks(stream.index+0.3)
ax.set_xticklabels(stream['Track'])
plt.xticks(rotation=45, ha='right')

# Add a legend.
ax.legend()

plt.show()
print()

'''
    3rd chart: 3 Pie charts: 
    Using Album_type, Licensed, and official_video columns: 
    Types of Albums, The Video is Licensed, and The Video is Official.
'''
print('Types of Albums, The Video is Licensed, and The Video is Official:')

# Make a copy of the original data frame.
df_copy = df.copy()

# Filter to keep only 'True' and 'False' values
licensed_filtered = df_copy[df_copy['Licensed'].isin([True, False])]
official_video_filtered = df_copy[df_copy['official_video'].isin([True, False])]

album_type_count = df['Album_type'].value_counts()
licensed_count = licensed_filtered['Licensed'].value_counts()
official_video_count = official_video_filtered['official_video'].value_counts()

labels1 = album_type_count.index.tolist()
sizes1 = album_type_count.values.tolist()
labels2 = licensed_count.index.tolist()
sizes2 = licensed_count.values.tolist()
labels3 = official_video_count.index.tolist()
sizes3 = official_video_count.values.tolist()

# Create two subplots.
fig, axs = plt.subplots(1, 3, figsize=(20,15))

# 'autopct=%1.1f%%' means to display the percentage value rounded to one decimal place.
# 'startangle=90' means that all the slices are rotated counter-clockwise by 90 degrees.
axs[0].pie(sizes1, labels=labels1, autopct='%1.1f%%', startangle=90)
axs[1].pie(sizes2, labels=labels2, autopct='%1.1f%%', startangle=90)
axs[2].pie(sizes3, labels=labels3, autopct='%1.1f%%', startangle=90)

# Set titles for each subplot.
axs[0].set_xlabel('Types of Albums', fontdict={'fontsize': 12, 'fontweight': 'bold'})
axs[1].set_xlabel('The Video is Licensed', fontdict={'fontsize': 12, 'fontweight': 'bold'})
axs[2].set_xlabel('The Video is Official', fontdict={'fontsize': 12, 'fontweight': 'bold'})

# Add a legend and set it to lower right
axs[0].legend(labels1, loc="lower right")
axs[1].legend(labels2, loc="lower right")
axs[2].legend(labels3, loc="lower right")

plt.show()
print()

'''
    4th chart: Scatter chart: 
    Using Loudness and Acousticness columns: 
    Loudness and Acousticness.
'''
print('Loudness and Acousticness:')

grouped = df.groupby(['Loudness'], as_index=False)['Acousticness'].sum()

# Filter the Acousticness column from 0 to 1.
acousticness = grouped['Acousticness'].tolist()
acousticness_filtered = [x for x in acousticness if 0 <= x <= 1]

# Get the loudness values for the filtered acousticness values.
loudness = grouped.loc[grouped['Acousticness'].isin(acousticness_filtered), 'Loudness'].tolist()

# Create the scatter chart with custom settings.
fig, ax = plt.subplots(figsize=(15, 8))
ax.scatter(loudness, acousticness_filtered, s=20)

plt.title('Loudness and Acousticness', fontsize=18, fontweight='bold')
plt.xlabel('Loudness (db)', fontsize=14, fontweight='bold')
plt.ylabel('Acousticness', fontsize=14, fontweight='bold')

plt.show()
print()

