'''
CS5242 Project - Classification of videos actions using breakfast action datasets

-----------------
List of packages (Python 3.5):
Keras                2.3.1
tensorflow           2.1.0
numpy                1.18.2
pandas               0.24.2
sklearn              0.0
scipy                1.4.1
torch                1.4.0
'''

import os
from lib.functions import *

# Choose model to train
from modelConfigs.ModelA import *
model, modelName = createModel()

# Path directory
workDir = '/home/j/jtandi/cs5242/cs5242project'
savePath = os.path.join(workDir, 'results', modelName)

# Train (and validate)
trainedModel = train(model, modelName, savePath, batchSize=50, epochs=50, COMP_PATH=workDir)

# Test 
test(trainedModel, savePath, COMP_PATH=workDir)