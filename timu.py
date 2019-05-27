import keras
from keras.layers import Dense,Conv1D,Input,Reshape,Flatten  #卷积窗为8 先验
from keras.utils import to_categorical
from keras.optimizers import Adam
import pandas as pd
from keras.models import Model
import numpy as np
from keras.utils import plot_model
df_train_seq=pd.read_csv('train_sequence.csv').values
df_train_lab=pd.read_csv('train_label.csv').values
df_test_seq=pd.read_csv('test_sequence.csv').values
df_train_seq=df_train_seq.reshape((-1,12,1))
# df_train_lab=df_train_lab.reshape((-1,1,1))
model_input=Input((12,1))
# model_input=Reshape((1,12))(model_input)
x=Conv1D(64,8,activation='tanh')(model_input)
x=Conv1D(32,4,activation='tanh')(x)
# x=Conv1D(16,2,activation='tanh')(x)
x=Flatten()(x)
model_out=Dense(12,activation='softmax')(x)
model=Model(model_input,model_out)
model.compile(optimizer=Adam(lr=1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
plot_model(model,'model.png')
model.fit(df_train_seq,df_train_lab,batch_size=32,epochs=13,validation_split=0.2)
