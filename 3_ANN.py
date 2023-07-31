import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_fetch import load_homemade_csv
from data_functions import plot_previsions, getBatch, batch_generator, evaluate_simple
import datetime
from sklearn.preprocessing import MinMaxScaler

from matplotlib.ticker import MaxNLocator

from keras.models import Sequential
from keras.layers import Dropout
from keras import layers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau



#TUNABLE PARAMETERS AND CONSTANTS

shift_steps = 24
aq_parameter = 8 #NO2
train_split = 0.9
#this is not 1-train_split, 1-train_split is test_split
validation_split = 0.1

sequence_length = shift_steps * 3
hidden_layer_size = 32

batch_size = 128
steps_per_epoch = 100
epochs = 20

np.random.seed(12345678)


#LOAD THE DATA

df = load_homemade_csv(folderName="RN_data")

parameters_per_station = len(list(df.columns.levels[1][:-1]))

df_time = pd.DataFrame()
df_time['DateTime'] = df.index.copy()
df_time.set_index(['DateTime'], inplace=True)
df_time['Month'] = df_time.index.month
df_time['DayOfWeek'] = df_time.index.dayofweek
df_time['Hour'] = df_time.index.hour

for col in df_time.columns:
    df['Time', col] = df_time[col]

df_targets = df.iloc[:, df.columns.get_level_values(1)==aq_parameter].copy()

df_targets = df_targets.iloc[shift_steps:]
df = df.iloc[:-shift_steps]

x_data = df.values
y_data = df_targets.values

num_data = len(x_data)
num_train = int(train_split * num_data)
num_test = num_data - num_train


df_x_train = df.iloc[0:num_train].copy()
df_x_test = df.iloc[num_train:].copy()

df_y_train = df_targets.iloc[0:num_train].copy()
df_y_test = df_targets.iloc[num_train:].copy()

print(df_y_test.describe())


x_train = df_x_train.values
x_test = df_x_test.values

y_train = df_y_train.values
y_test = df_y_test.values


num_x_signals = x_data.shape[1]
num_y_signals = y_data.shape[1]

print(num_x_signals, num_y_signals)


#SCALING THE DATA

x_scaler = MinMaxScaler()
x_scaler.fit(x_data)

x_train_scaled = x_scaler.transform(x_train)
x_test_scaled = x_scaler.transform(x_test)

y_scaler = MinMaxScaler()
y_scaler.fit(y_data)

y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


all_data_generator = batch_generator(x_source = x_train_scaled, y_source = y_train_scaled, input_signals_count = num_x_signals, output_signal_count = num_y_signals,
                        data_size = num_train, batch_size=(num_train - sequence_length + 1), sequence_length=sequence_length, y_as_sequence=False, shuffle=False)

train_data_x, train_data_y = next(all_data_generator)



#CREATING BATCHES

all_data_generator = batch_generator(x_source = x_train_scaled, y_source = y_train_scaled, input_signals_count = num_x_signals, output_signal_count = num_y_signals,
                        data_size = num_train, batch_size=(num_train - sequence_length + 1), sequence_length=sequence_length, y_as_sequence=False, shuffle=False)

train_data_x, train_data_y = next(all_data_generator) #(4246, 21, 23) ok, (4246, 3) poiché y_as_sequence false
train_data_size = train_data_x.shape[0] #4246


#shuffle train data sequences
shuffle_seed = np.arange(train_data_size)
np.random.shuffle(shuffle_seed)

train_data_x = train_data_x[shuffle_seed]
train_data_y = train_data_y[shuffle_seed]


validation_data_size = int(train_data_size * validation_split)
val_data_x = train_data_x[train_data_size-validation_data_size:]
val_data_y = train_data_y[train_data_size-validation_data_size:]

train_data_x = train_data_x[0:train_data_size-validation_data_size]
train_data_y = train_data_y[0:train_data_size-validation_data_size]
train_data_size = train_data_x.shape[0] #4246-425=3821


train_generator = getBatch(x = train_data_x, y = train_data_y, batch_size=batch_size)



validation_data = (val_data_x, val_data_y)

test_generator = batch_generator(x_source = x_test_scaled, y_source = y_test_scaled, input_signals_count = num_x_signals, output_signal_count = num_y_signals,
                        data_size = num_test, batch_size = (num_test - sequence_length + 1), sequence_length = sequence_length, y_as_sequence=False, shuffle=False)

test_data_x, test_data_y = next(test_generator)


#TESTING DIFFERENT ANN

code_name = {0: "linear", 1 : "ann_base_multi", 2 : "gru_single_layer",  3 : "gru_multi_layer"}
model_type = 3 #CHOOSE BETWEEN MODEL TYPES LISTED ABOVE, SEE THE RESULTS
model = Sequential()

if model_type == 0:
    model.add(layers.Flatten(input_shape=(sequence_length, x_data.shape[1])))
    model.add(layers.Dense(num_y_signals, activation='linear'))

if model_type == 1:

    model.add(layers.Flatten(input_shape=(sequence_length, x_data.shape[1])))
    model.add(layers.Dense(hidden_layer_size, activation='relu'))
    model.add(layers.Dense(hidden_layer_size, activation='relu'))
    model.add(layers.Dense(num_y_signals, activation='sigmoid'))

if model_type == 2:

    model.add(layers.GRU(hidden_layer_size, activation='relu', input_shape=(sequence_length, x_data.shape[1])))
    model.add(layers.Dense(num_y_signals, activation='sigmoid'))

if model_type == 3:

    model.add(layers.GRU(hidden_layer_size*2,
                return_sequences=True,
                input_shape=(sequence_length, x_data.shape[1])))
    model.add(layers.GRU(hidden_layer_size))
    model.add(layers.Dense(num_y_signals, activation='sigmoid'))


model.compile(optimizer=Adam(), loss='mae')
model.summary()

timenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
run_name = '{7}_Nodes_{0}-Units_{1}-shift_steps_{2}-aq_parameter_{3}-train_split_{4}-batch_size_{5}-sequence_length_{6}'.format(code_name[model_type], hidden_layer_size, shift_steps, aq_parameter, train_split, batch_size, sequence_length, timenow)
path_checkpoint = 'checkpoints\\{0}.keras'.format(run_name)

callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

callback_tensorboard = TensorBoard(log_dir='./ann_test_logs/', histogram_freq=0, write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-4, patience=3, verbose=1)

                            
callbacks = [callback_checkpoint, callback_tensorboard, callback_reduce_lr]

history = model.fit(x=train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=validation_data, callbacks = callbacks)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))


ax = plt.figure().gca()
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Mean absolute error')
plt.legend()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig('figures\\loss___{0}.png'.format(run_name), bbox_inches='tight')
#plt.show()

print('Best MAE on validation: ',min(val_loss))

try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

simple_loss, simple_prevs = evaluate_simple(x=test_data_x,
                y=test_data_y, parameters_per_station=parameters_per_station)

print("simple_loss (test-set):", simple_loss)
larr = np.zeros((1,num_y_signals))
for i in range(num_y_signals):
    larr[0,i] = simple_loss
print("simple_loss (test-set rescaled):", y_scaler.inverse_transform(larr)[0,0])


result = model.evaluate(x=test_data_x, y=test_data_y)

print("loss (test-set):", result)
larr = np.zeros((1,num_y_signals))
for i in range(num_y_signals):
    larr[0,i] = result
print("loss (test-set rescaled):", y_scaler.inverse_transform(larr)[0,0])


df_y_test_removed_first = df_y_test.iloc[sequence_length-1:].copy()
plot_previsions(x = test_data_x, df_y_true = df_y_test_removed_first, model = model, start_idx = 9400, y_scaler=y_scaler, run_name=run_name, length=150)
