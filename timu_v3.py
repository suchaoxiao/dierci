import keras
from keras.layers import Dense,Conv1D,Input,Reshape,Flatten,Dropout  #卷积窗为8 先验
from keras.utils import to_categorical
from keras.optimizers import Adam
import pandas as pd
from keras.models import Model
import numpy as np
from keras.utils import plot_model
df_train_seq=pd.read_csv('train_sequence.csv',header=None).values
df_train_lab=pd.read_csv('train_label.csv',header=None).values
df_test_seq=pd.read_csv('test_sequence.csv',header=None).values
model_input=Input(shape=(12,))
x=Dense(64,activation='relu')(model_input)
x=Dense(128,activation='relu')(x)
x=Dropout(0.3)(x)
x=Dense(64,activation='relu')(x)
model_out=Dense(12,activation='softmax')(x)

model=Model(model_input,model_out)
model.compile(optimizer=Adam(lr=1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
plot_model(model,'v3_model.png',show_shapes=True)
model.fit(df_train_seq,df_train_lab,batch_size=32,epochs=5,validation_split=0.1)
