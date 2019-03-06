from keras.utils import np_utils
from keras.layers import Dropout, Dense, Activation
from keras.models import Sequential
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping  
from keras.utils.np_utils import to_categorical

class FFClassifier:
    ''' Class for ANN Classifier'''
    
    def __init__(self, train_data=None, valid_data=None,
                 target_train=None, target_valid=None, 
                 num_classes=2, dropout=0.5):
        '''Initialize dataset'''
        
        # Training/Validation Data
        self.X_train = train_data
        self.X_valid = valid_data
        self.y_train = target_train
        self.y_valid = target_valid
                
        # Encode target variables
        if target_train is not None and target_valid is not None:
            self.y_train_cat = to_categorical(target_train) 
            self.y_valid_cat = to_categorical(target_valid) 

        self.num_classes = num_classes
        self.dropout = dropout
        self.model = None

        
    def create_model(self, _input_dim):
        ''' Build simple ANN Model'''
        
        model = Sequential()
        model.add(Dense(64, input_dim=_input_dim))
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout))

        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout))

        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        model.summary()
        self.model = model
    
        
    def compile_model(self, _optimizer, _loss, _metrics):
        ''' Set optimization and loss functions'''
        #self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer=_optimizer, 
                           loss=_loss, 
                           metrics= _metrics)
        print("\n\nModel successfully compiled!\n\n")
    

    def train_model(self, checkpoint_file, epochs=30):
        ''' Train the model and save to checkpoint '''
    
        train_tensors = self.X_train 
        train_targets = self.y_train_cat
        valid_tensors = self.X_valid
        valid_targets = self.y_valid_cat
        #print(train_tensors.shape, valid_tensors.shape, train_targets.shape, valid_targets.shape)

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)

        checkpointer = ModelCheckpoint(filepath=checkpoint_file, 
                                       verbose=1, save_best_only=True)

        self.model.fit(train_tensors, train_targets, 
                  validation_data=(valid_tensors, valid_targets),
                  epochs=epochs, batch_size=64, callbacks=[checkpointer, early_stopping], verbose=1)

        return self.model
