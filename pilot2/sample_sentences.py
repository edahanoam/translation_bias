import pandas as pd

def sample():
    df = pd.read_csv("updated_unamb.csv")
    selected_rows_take = df[df['take'] == 1]

    remaining_df_after_take = df[~df.index.isin(selected_rows_take.index)]

    if remaining_df_after_take[remaining_df_after_take['gender'] == 'F'].shape[0] >= 3:
        selected_rows_female = remaining_df_after_take[remaining_df_after_take['gender'] == 'F'].sample(3)
    else:
        selected_rows_female = remaining_df_after_take[remaining_df_after_take['gender'] == 'F']

    combined_selected = pd.concat([selected_rows_take, selected_rows_female])

    remaining_df = df[~df['original_id'].isin(combined_selected['original_id'])]

    if remaining_df.shape[0] >= 10:
        selected_final_rows = remaining_df.sample(10)
    else:
        selected_final_rows = remaining_df  # If less than 10, take all available

    final_df = pd.concat([selected_final_rows, combined_selected])

    print(final_df)
    final_df.to_csv("sampled_unamb.csv")


if __name__ == '__main__':
    print("startd")
    sample()