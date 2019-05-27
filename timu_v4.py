from keras import backend as K
from keras.layers import Dense,Conv1D,Input,Reshape,Flatten,Dropout  #卷积窗为8 先验
from keras.utils import to_categorical
from keras.optimizers import Adam
import pandas as pd
from keras.models import Model
import numpy as np
from keras.utils import plot_model
# from keras.objectives import binary_crossentropy as bce
from keras.objectives import categorical_crossentropy as bce
from keras.activations import softmax
data_dim=12

df_train_seq=pd.read_csv('train_sequence.csv',header=None).values
df_train_lab=pd.read_csv('train_label.csv',header=None).values
df_test_seq=pd.read_csv('test_sequence.csv',header=None).values


df_train_seq=df_train_seq.reshape((-1,12,1))
df_test_seq=df_test_seq.reshape((-1,12,1))
# df_train_lab=df_train_lab.reshape((-1,1,1))
model_input=Input((12,1))
# model_input=Reshape((1,12))(model_input)
x=Conv1D(64,8,activation='relu')(model_input)
x=Conv1D(64,4,activation='relu')(x)
x=Dropout(0.3)(x)
# x=Conv1D(128,2,activation='tanh')(x)
x=Flatten()(x)
model_out=Dense(12,activation='softmax')(x)
label_input=Input((1,))
model_out_= np.argmax(model_out)
def gumbel_loss(label_input, model_out_):
    # q_y = K.reshape(logits_y, (-1, N, M))
    # q_y = K.reshape(model_out, (-1, N, M))
    # q_y = softmax(q_y)
    q_y=model_out
    log_q_y = K.log(q_y + 1e-20)
    kl_tmp = q_y * (log_q_y - K.log(1.0))
    KL = K.sum(kl_tmp, axis=1)
    elbo = data_dim * bce(label_input, model_out_) - KL
    return elbo

model=Model(model_input,model_out)
model.compile(optimizer=Adam(lr=1e-3),loss=gumbel_loss,metrics=['accuracy'])
plot_model(model,'model.png',show_shapes=True)
model.fit(df_train_seq,df_train_lab,batch_size=32,epochs=30,validation_split=0.1)

# train_test=df_train_seq[31900:,:].reshape((-1,12,1))
# train_test_label=df_train_lab[31900:,:]
# predict=model.predict(train_test)
predict=model.predict(df_test_seq)
predict_label=np.argmax(predict,axis=-1)
print(predict.shape)
print(predict_label)
# data1 = pd.DataFrame(predict_label)
# data1.to_csv('pred_2.csv',header=0, index=0)

