# BeehiveMachineLearning

## Goal

Identification of Queen Status, given a Bee Audio dataset and some ancilliary metadata on the audio files. More information can be found in the `notebooks/`. Some example architectures (before hyper-parameter optimisation) can also be found in this sub-directory.

## Information

### Notebooks
1. Audio Data Feature Extraction
   - Extracts the features into mel spectrograms, to be used in the machine learning model.
2. Post Image Gen
   - Processes the image data
   - Create deep learning models to train, hyper-parameter optimise and evaluate.

### Scripts
There are several auxilliary scripts to do some data processing into:
1. Mel Spectrograms.
2. Directory and folder structure expected by Tensorflow.


### Resources
There are a multitude of resources that I have used and are available on request.
The dataset is from https://www.kaggle.com/datasets/annajyang/beehive-sounds.

I am also not going to upload all the images, as it is too large to realistically upload to GitHub.


### Models:
There were 3 general models I wanted to try:
1. Simple CNN model, with only the Mel Spectrograms as inputs.
2. Simple CNN model, with both the Mel Spectrograms and metadata as inputs.
3. DenseNet CNN model.

Unfortunately, I could not compile (3) due to hardware constraints and (1) and (2) were also limited during the hyper-parameter optimisation stage.

The models were evaluated with some small number of parameters during the hyper-parameter stage, using Hyperband optimisation. The models trained using the ADAM optimiser.

The final results were the following, using Categorical Cross Entropy for evaluation:
1. Only Mel Spectrograms (Images) - Accuracy: 80%.
2. Mel Spectrograms (Images) + Metadata features - Accuracy: 86%.

I tested on the test sample set only once, to reduce manual bias introduced due to further model changes.

## Notable Obstacles Faced
1. Theory. Learning about Fourier Transforms, Audio Data Classification, Mel Spectrograms, Mel-frequency Spectral Coefficients (MFFCs) all required a large investment in time to allow me to even begin feature engineering. 
2. Vanishing gradient problem. I think this is a pretty common issue faced on such large neural network models, especially CNNs. I got around this by using Dropouts, Batch Normalisation and normalising the metadata ahead of time.
3. Hardware Limitations. I was bottlenecked by GPU VRAM.
4. Partially loading 7100 images. It turned out that there was Tensorflow API to do this already. However, concatenating this data with the ancilliary metadata required custom Tensorflow `Dataset` generation. There was limited documentation on how to do this meaningfully and was a large obstacle I had to overcome.


## Next Steps
It would be nice to test this on a cloud infrastructure setup, to avoid hardware limitations. 

Another thing I want to try is using less computationally intensive models such as XGBoost Trees.

Finally, I wonder if there is a way to introspect layer-by-layer performance. Maybe there are further optimisations I can do purely on the loss per layer.
