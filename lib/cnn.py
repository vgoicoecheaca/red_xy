'''CLASS to deal with cnn training and fine tunning'''
from keras.models import load_model 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, UpSampling2D,Input,ZeroPadding2D,Dense,Dropout
from tensorflow.keras.regularizers import l2

class CNN():
    def __init__(self,manager): 
        self.m = manager
        self.trained_model = self.m.config('file','path','str')+self.m.config('cnn','model','str')
    
    def build_model(self):
        model = Sequential([
            Input(shape=(6,4,1),name="Input"),
            ZeroPadding2D(padding=(1,2),name="Padding"),
            UpSampling2D(size=(2,2),data_format="channels_last",interpolation="bilinear"),
            Conv2D(3, (2,2),activation=self.m.config('cnn','act','str'), name="Conv1",padding ="same"),
            Conv2D(6, (2,2),activation=self.m.config('cnn','act','str'),strides = (2,2), name="Conv2",padding ="same"),
            Conv2D(6, (2,2),activation=self.m.config('cnn','act','str'),strides = (1,1), name="Conv3",padding ="same"),
            Conv2D(12,(2,2),activation=self.m.config('cnn','act','str'),strides = (2,2), name="Conv4",padding ="same"),
            Conv2D(12,(2,2),activation=self.m.config('cnn','act','str'),strides = (1,1), name="Conv5",padding ="same"),
            Conv2D(24,(1,1),activation=self.m.config('cnn','act','str'),strides = (2,2), name="Conv6",padding ="same"),
            Conv2D(24,(1,1),activation=self.m.config('cnn','act','str'),strides = (1,1), name="Conv7",padding ="same"),
            Flatten(),
            Dense(96,activation=self.m.config('cnn','act','str'),name = 'Dense1',activity_regularizer=l2(0.05)), 
            Dense(48,activation=self.m.config('cnn','act','str'),name = 'Dense2',activity_regularizer=l2(0.05)),
            Dropout(0.1),
            Dense(24,activation=self.m.config('cnn','act','str'),name = 'Dense3',activity_regularizer=l2(0.05)),
            Dense(2,name="Output")])
        return model

    def train(self,x,y):
        if self.m.config('cnn','fine_tune','bool'):
            base_model = load_model(self.trained_model)
            base_model.trainabale = True
            fine_tune_at = 9
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False 
            self.model = base_model 
        else:
            self.model = self.build_model()
        self.model.compile(optimizer=Adam(learning_rate=self.m.config('cnn','lr','float')),loss ='mae')    
        print(self.model.summary()) 
        history = self.model.fit(x,y,
                       epochs           = self.m.config('cnn','epochs','int'),
                       validation_split = self.m.config('cnn','val_split','float'), 
                       shuffle          = self.m.config('cnn','shuffle','bool'), 
                       callbacks        = [EarlyStopping(monitor="val_loss", patience=self.m.config('cnn','patience','int'), mode="min")])
        self.model.save(self.m.config('file','path','str')+self.m.config('cnn','model_tr','str'))
        return self.m.config('file','path','str')+self.m.config('cnn','model_tr','str'), history

    def predict_pos(self,model,events):  
        return load_model(model).predict(events)

