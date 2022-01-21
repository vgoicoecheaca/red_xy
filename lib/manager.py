'''A Class to brige the different modules of this code'''

from lib.config import Config 
from lib.helper import Helper  
from lib.plotter import Plotter
from lib.cnn import CNN

class Manager():
    def __init__(self):
        self.config  = Config()
        self.helper  = Helper(manager=self)
        self.plotter = Plotter(manager=self)
        self.cnn     = CNN(manager=self)
    