# Self-supervised learning
## Auto-Encoder Based Reconstruction approach followed by a multilayer perceptron classifier

### Repository Structure,
- Gesture
    - Contains the Sequence of TimeSeries data where each sequence denotes a gesture
    - Used this data to fine-tune the encoder and train the classifier.
    - Contains timeseries data of 8 unique gestures. 
- HAR
    - Contains similar type of data as Gesture but with only 5 unique gestures
    - Used this data to train the encoder to learn the latent representation of the gesture data
- src
    - Contains Model, Data, and output folders to store model results etc.
    - The name convention used to name output folders is - '{embeddingsDimension}_x{numOfGRULayersInEncoder}_drop{droputPercentage}_{modelName}'

### Instructions to train the model.
- First ensure you have the following packages of python,
    1. scikit-learn, torch, pandas, seaborn, matplot
- To simply run the script and repo, just run main.py file,
    - For Eg: `python main.py --input-folder ../HAR/ --classifier-input-folder ../Gesture/ --embeddings-dim 128`
    - In this case, the encoder generates an representation of shape (128 , 1)
- To Model folder contains two models, TimeNet and GestureClassifier. TimeNet used in Self-supervised learning phase, and GestureClassifier to classify

