import pandas as pd
import os

meta = pd.read_csv('../data/flattened_data.csv')

classes = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
}

cwd = os.getcwd()

for key, data in meta.iterrows():
    c = data['queen status'] 
    image = f"{cwd}/../data/resized_image_files/{data['file name']}.png"
    # dest = f"{cwd}/../data/classes/class_{c}/{c}_{data['file name']}_{classes[c]:0>4}.png"
    dest = f"{cwd}/../data/classes/class_{c}/{data['file name']}.png"

    # Remove if exists
    try:
        os.remove(dest)
    except:
        pass
    os.symlink(image, dest)

    classes[c] += 1


