
# DeepBach

  

This is an implementation of DeepBach in Keras. There are a couple steps to training a model to generate music.

All commands will be run from the root folder (or deep_bach/) from the command line and **not in the src folder.**

Before starting, make sure to install requirements:

```
pip -r requirement.txt
```

  

## 1. Serializing Raw Data

  

The first step is to get the raw data from the music21 library and save it to disk for easy access. To do this the following command must be performed in the command line:

  

```
python src/data/get_raw_data.py
```



## 2. Preprocessing Data

Next the data must be preprocessed into a dataframe. This can simply be done be running:
```
python src/preprocess/preprocess.py
```

This creates 2 dataframes called _train_df.csv_ and _test_df.csv_. These include the sequential events of the bach chorales with 4 voices.

 The next step involves creating an object that will create the samples. This is called the data generator.
## 3. Creating The Datagen

This as step is slightly more involved as it take a few parameters. The command that should be ran is the following:

```
python src/models/create_datagen.py train_df.csv test_df.csv <train datagen name> <test datagen name> <delta time> <gap>
```

The first 2 arguments are respectively the names of the train and test dataframes **relative to the data folder**. The next 2 arguments are simply the names of the train and test datagens that are being created and can be whatever you want. Then the final arguments are where things get interesting.

### Delta time

Delta time is the amount of notes the model can see in the past and future at any given time. So a higher _dt_ will result in the model having more memory/context and a smaller _dt_ will result in a more localized model that cannot understand long term structure as much. 

I have not played around too much with this parameter so it would be a good place to explore. I would recommend trying 4, 8, 16, and maybe 32 as your _dt_ and compare the results. Increasing the _dt_ will also increase the complexity of the data as longer sequences will be used as inputs. So using a _dt_ of 32 will create a model that takes longer to train than a _dt_ of 4.

### Gap
Gap is the increment at which samples will be created.  If gap is 5 then every 5 time units a new sample will be created. A lower gap create more samples but may lead to overfitting as the model will have seen more of each piece. I would suggest trying gap values of 1, 5, and 10 and see how they compare.


## 4. Train The Model

Finally it is time to build and train the model. This can be done by entering the following:

```
python src/models/train_model.py <train_datagen_name> <test_datagen_name> <model name> <embedding dimension> <lstm1 dimension> <middle dimension>
```

The train and test datagen are the names of the datagens that were created. The model name can be anything. The next 3 arguments are hyper parameters for the model and will determine how complex the neural network is. A higher number creates a more complex network that can potentially learn more complicated trends. But it also can lead to overfitting so there must be a balance between complexity and simplicity. For baseline, try the following setup:

**Embedding dimension**: 16
**LSTM1 dimension**: 200
**Middle dimension**: 200

There are optional arguments as well. To use those enter the following after positional arguments:
```
--<argument> <value>
```
These options include:

1. LSTM2Dimension - Adds another LSTM layer. Adds more complexity to the network.
