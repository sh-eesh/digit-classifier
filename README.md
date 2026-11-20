# Digit Classifier
A handwritten digit classifier built with Python and PyTorch. It is trained on the MNIST dataset and uses a simple feedforward neural network to recognize digits from 0 to 9.
It provides a simple drawing interface to test the model.

## Setup
Clone the repository. <br>
`git clone https://github.com/sh-eesh/digit-classifier` <br>
`cd digit classifier`

Create and activate the virtual environment. <br>
`py -m venv .venv` <br>
`.venv\Scripts\activate`

Install dependencies <br>
`pip install -r requirements.txt`

## Training
Run the training script. <br>
`python src/train.py` <br>
The MNIST dataset is downloaded automatically and saved to the models folder.

## Run the Application
After training is complete run. <br>
`python src/app.py` <br>
Draw a digit on the canvas and click on Predict to see the result.