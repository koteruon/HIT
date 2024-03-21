import pandas as pd

path_ = f'data/table_tennis/annotations/action_timestamp_.csv'

csv_df = pd.read_csv(path_)
new_df = pd.DataFrame(columns=csv_df.columns)

cur = 0
movies = pd.unique(csv_df['video_id'].values.flatten())
for video in movies:
    local_df = csv_df.loc[csv_df['video_id']==video].copy()
    local_df.reset_index(drop=True, inplace=True)
    times = 0
    for idx, row in local_df.iterrows():
        new_df.loc[len(new_df.index)] = [row[0], row[1]+1*times, row[2], row[3], row[4]]
        if len(new_df.iloc[cur:].index) % 10 == 0:
            new_df.loc[len(new_df.index)] = [row[0], row[1]+1+1*times, 2, 1, row[4]+30]
            new_df.loc[len(new_df.index)] = [row[0], row[1]+1+1*times, 2, 2, row[4]+30]
            cur = len(new_df.index)
            times +=1
new_df.to_csv(f'data/table_tennis/annotations/action_timestamp_.csv', index=False)
# print(new_df)