from ast import literal_eval
import pandas as pd

meta = pd.read_csv(f"../data/data.csv")

new_meta = []
for key, row in meta.iterrows():

    new_rows = []
    old_name = row['file name']
    for segment in literal_eval(row.segments):
        tmp_row = row.copy()
        tmp_row['file name'] = f"{old_name}__segment{segment}"
        new_rows.append(tmp_row)
    new_meta.extend(new_rows)

new_meta = pd.DataFrame(new_meta, columns=meta.columns)
new_meta.to_csv("../data/flattened_data.csv")
