import numpy as np
import pandas as pd


#-------------------FUNCTIONS-------------------#

# Function to calculate the mean squared error
def mse(a, b):
    return np.mean((a - b) ** 2)

# Function to calculate ratios between the elements of an array
def relative_differences(arr):
    return np.diff(arr) / arr[:-1]

# Function to compare the differnces between the query and the target arrays
def similarity_search(query, target_arrays, target_results, n=None):
    errors = []
    query_diffs = relative_differences(query)

    for target in target_arrays:
        min_error = float('inf')
        
        for i in range(len(target) - len(query_diffs)):
            target_diffs = relative_differences(target[i:i+len(query_diffs)+1])
            error = mse(query_diffs, target_diffs)
            if error < min_error:
                min_error = error
                
        errors.append(min_error)
    
    
    sorted_indices = np.argsort(errors)
    sorted_target_arrays = [target_arrays[i] for i in sorted_indices]
    sorted_target_results = [target_results[i] for i in sorted_indices]
    sorted_errors = [errors[i] for i in sorted_indices]

    # Return top n results if n is specified
    if n is not None:
        return sorted_indices[:n], sorted_target_arrays[:n],sorted_target_results[:n],sorted_errors[:n]
    # Return all results
    else:
        return sorted_indices, sorted_target_arrays,sorted_target_results,sorted_errors
    
'''

#-------------------PREPARE DATA-------------------#

# Set the number of values in the sequence to predict the next value.
SEQUENCE_SIZE = 5

data_column = 'Data'
date_column = 'Date'

data = pd.read_csv("Timeseries_TSLA.csv")

data = data.dropna(subset=[data_column])

data = data[[date_column,data_column]]

data[date_column]= pd.to_datetime(data[date_column])

data = data.sort_values(date_column, ascending=False)
data = data.reset_index(drop=True)

data_list = data[data_column].tolist()

# Remove the elements from the beginning of the list to make the length a multiple of SEQUENCE_SIZE
data_list = data_list[len(data_list) % SEQUENCE_SIZE:]

# Define a function to convert the data into sequences.
def to_sequences(seq_size, obs):
    x, y = [], []
    for i in range(len(obs)-seq_size):
        # Create a window of data of length seq_size.
        window = obs[i:(i+seq_size)]
        
        # Get the value after the window.
        after_window = obs[i+seq_size]
        
        # Add the window and after_window to the x and y lists.
        x.append(window)
        y.append(after_window)

    # Return x and y numpy arrays and the last value from the y array.
    return np.array(x),np.array(y)

#-------------------SEARCH-------------------#

# Use the to_sequences function to create sequences and predictions.
sequences, predictions = to_sequences(SEQUENCE_SIZE, data_list)
# Convert to integers
sequences = sequences.astype(int)
predictions = predictions.astype(int)

# Print each row of the sequences and predictions
for i in range(len(sequences)):
    print("{} - {}".format(sequences[i],predictions[i]))

query = np.array([185,192,194,207,195])

sorted_indices, sorted_target_arrays,sorted_target_results,sorted_errors = similarity_search(query, sequences, predictions, 5)

print("\nSorted indices:", sorted_indices)
print("\nSorted target arrays (most similar first):", sorted_target_arrays)
print("\nSorted target results (most similar first):", sorted_target_results)
print("\nMSE:", sorted_errors)

# Print 10 most similar sequences,their indexes and the MSE
for i in range(len(sorted_indices)):
    print("{} - {} => {} - {}".format(sorted_indices[i], sorted_target_arrays[i], sorted_target_results[i],sorted_errors[i]))

'''