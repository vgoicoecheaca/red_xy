'''A Class to brige the different modules of this code'''

from config import Config 
from helper import Helper  
from plotter import Plotter
from cnn import CNN

class Manager():
    def __init__(self):
        self.config  = Config()
        self.helper  = Helper(manager=self)
        self.plotter = Plotter(manager=self)
        self.cnn     = CNN(manager=self)
    