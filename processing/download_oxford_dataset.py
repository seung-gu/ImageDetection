import gdown
import os

# oxford dataset
url = 'https://drive.google.com/uc?id=1dIR9ANjUsV9dWa0pS9J0c2KUGMfpIRG0'
fname = 'oxford_pet.zip'
path = os.path.join('..', 'dataset', fname)
gdown.download(url, path, quiet=False)
