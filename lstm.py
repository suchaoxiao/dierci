import keras
from keras.layers import Dense,Conv1D,Input,Reshape,Flatten,Dropout,LSTM  #卷积窗为8 先验
from keras.utils import to_categorical
from keras.optimizers import Adam
import pandas as pd
from keras.models import Model
import numpy as np
from keras.utils import plot_model
df_train_seq=pd.read_csv('train_sequence.csv',header=None).values
df_train_lab=pd.read_csv('train_label.csv',header=None).values
df_test_seq=pd.read_csv('test_sequence.csv',header=None).values


df_train_seq=df_train_seq.reshape((-1,12,1))
df_test_seq=df_test_seq.reshape((-1,12,1))
# df_train_lab=df_train_lab.reshape((-1,1,1))
model_input=Input((12,1))
# model_input=Reshape((1,12))(model_input)
x=Conv1D(64,8,activation='relu')(model_input)
x=LSTM(32)(x)
# x=Conv1D(64,4,activation='relu')(x)
# x=Dropout(0.3)(x)
# x=Conv1D(128,2,activation='tanh')(x)
# x=Flatten()(x)
# x=Dense(32,activation='relu')(x)
model_out=Dense(12,activation='softmax')(x)
model=Model(model_input,model_out)
model.compile(optimizer=Adam(lr=1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
plot_model(model,'V4_model.png',show_shapes=True)
model.fit(df_train_seq,df_train_lab,batch_size=32,epochs=30,validation_split=0.1)

# train_test=df_train_seq[31900:,:].reshape((-1,12,1))
# train_test_label=df_train_lab[31900:,:]
# predict=model.predict(train_test)
predict=model.predict(df_test_seq)
predict_label=np.argmax(predict,axis=-1)
print(predict.shape)
print(predict_label)
data1 = pd.DataFrame(predict_label)
data1.to_csv('V4_pred_2.csv',header=0, index=0)

