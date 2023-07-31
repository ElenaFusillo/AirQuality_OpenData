import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error


def plot_previsions(x, df_y_true, model, start_idx, y_scaler, run_name, length=100, aq_parameter=8, y_prevs = None):
    end_idx = start_idx + length
    x = x[start_idx:end_idx]
    y_true = df_y_true.iloc[start_idx:end_idx]

    if y_prevs is None:
        y_pred = model.predict(x, verbose=1)
    else:
        y_pred = y_prevs[start_idx:end_idx]
    y_pred_rescaled = y_scaler.inverse_transform(y_pred)

    print(df_y_true.describe())
    print(y_true.describe())
    station_ids = [i[0] for i in y_true.columns.values]
    print(station_ids)

    # For each output-signal.
    for i, station_id in enumerate(station_ids):
        df_to_plot = pd.DataFrame()
        df_to_plot['DateTime'] = y_true.index.copy()
        df_to_plot['TrueValue'] = y_true[station_id][aq_parameter].values
        df_to_plot['Predicted'] = y_pred_rescaled[:, i]
        df_to_plot.set_index(['DateTime'], inplace=True)

        print(df_to_plot.head())

        ax = df_to_plot.plot(figsize=(12, 8), title="Test set predictions - station: {0}".format(station_id))
        ax.set_ylabel("NO2 [ug/m3]")
        #plt.show()
        plt.savefig('figures\\prevs_{1}___{0}.png'.format(run_name, station_id), bbox_inches='tight')
    plt.show()

def getBatch(x, y, batch_size): #(4246, 21, 23), (4246, 3), 128
    while True:
        random_indexes = np.arange(x.shape[0])
        np.random.shuffle(random_indexes)
        random_indexes = random_indexes[0:batch_size]
        yield (x[random_indexes], y[random_indexes])

def batch_generator(x_source, y_source, input_signals_count, output_signal_count, data_size, batch_size, sequence_length, y_as_sequence=True, shuffle=True):
    """
    Generator function for creating random batches.
    """
    idx = 0
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, input_signals_count)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        if y_as_sequence:
            y_shape = (batch_size, sequence_length, output_signal_count)
        else:
            y_shape = (batch_size, output_signal_count)

        y_batch = np.zeros(shape=y_shape, dtype=np.float16)
        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            if shuffle:
                # Get a random start-index.
                # This points somewhere into the training-data.
                idx = np.random.randint(data_size - sequence_length)
            else: 
                if (idx + sequence_length) >= data_size:
                    idx = 0

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_source[idx:idx+sequence_length]
            if y_as_sequence:
                #idx+sequence_length is exclusive
                y_batch[i] = y_source[idx:idx+sequence_length]
            else:
                y_batch[i] = y_source[idx + sequence_length - 1]
            
            idx += 1
        
        yield (x_batch, y_batch)

def evaluate_simple(x, y, parameters_per_station=6):
    batch_input_size = y.shape[0]
    y_num = y.shape[1]

    previsions = np.zeros((batch_input_size, y_num))
    
    for i in range(0, batch_input_size):
        for j in range(0, y_num):
            previsions[i, j] = x[i, -1, parameters_per_station*j]

    loss = mean_absolute_error(y, previsions)
    return loss, previsions
