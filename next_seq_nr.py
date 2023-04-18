import pandas as pd
import numpy as np
import os
import time

#-------------------PREPARE DATA-------------------#

# Set the number of values in the sequence to predict the next value.
SEQUENCE_SIZE = 5

# Set the name of the column containing the data to be predicted.
data_column = 'Data'

# Set the name of the column containing the dates.
date_column = 'Date'

# Read the data from the CSV file.
data = pd.read_csv("Timeseries.csv")

# Drop all rows that have NaN values in the predicted column.
data = data.dropna(subset=[data_column])

# Keep only the date and data columns.
data = data[[date_column,data_column]]

# Convert the date column to a datetime object.
data[date_column]= pd.to_datetime(data[date_column])

# Sort the data by date and reset the index.
data = data.sort_values(date_column, ascending=True)
data = data.reset_index(drop=True)

# Convert the data column to a list.
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
        
    # Convert x and y to numpy arrays.
    return np.array(x),np.array(y)

# Use the to_sequences function to create sequences and predictions.
sequences, predictions = to_sequences(SEQUENCE_SIZE, data_list)
# Convert to integers
sequences = sequences.astype(int)
predictions = predictions.astype(int)

# Print each row of the sequences and predictions
for i in range(len(sequences)):
    print("{} - {}".format(sequences[i],predictions[i]))

#-------------------CREATE TEMPLATE-------------------#

# Define the template.
task = """What is the next number following this sequence:{} ?
The response should be in the format of a single number without context.
Examples: [194 192 185 185 184] => 186 or [233 213 216 237 236] => 235
"""

#-------------------CREATE MODEL-------------------#

import openai

# Openai_API_key
api_key = os.environ.get('OPENAI_API_KEY')
# Configure OpenAI and Pinecone
openai.api_key = api_key

# Function to call the OpenAI API
def llm_call(
    messages: str,
    model: str = 'gpt-3.5-turbo',
    temperature: float = 0,
    max_tokens: int = 100,
):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except openai.error.RateLimitError:
        print(
            "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
        )
        time.sleep(10)  # Wait 10 seconds and try again

    content = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens

    return content, tokens_used

#-------------------RUN MODEL-------------------#

# Set the number of times to run the model. 
MAX_ITERATIONS = len(sequences) # Use the whole timeseries dataset.
# Previous conversations to remember
MAX_MEMORY = 6

total_tokens_used = []
# Initialize the prompt with the first sequence.
prompt = task.format(sequences[0])
# Initialize the messages list with the prompt.
messages = [{"role": "user", "content": prompt}]

iterations = 0

actual_list = []
predicted_list = []

# Main loop to run the model, and modify the prompt based on the model's response.
for i in range(0,MAX_ITERATIONS):
    print("Iteration:{}".format(i+1))
    messages = messages
    next_nr,tokens_used = llm_call(messages)
    # Append tokens_used to total_tokens_used and sum the total tokens used.
    total_tokens_used.append(tokens_used)

    # Append a new dictionary to the messages list.
    messages.append({"role": "assistant", "content": next_nr})

    # If this is the last iteration, modify the sequence that will be used for prediction.
    # This is necessary if predicting on the whole timeseries dataset.
    if i == MAX_ITERATIONS-1:
        # Remove the first element from the sequences[i] and replace the last element with the predictions[i].
        seq4pred = np.delete(sequences[i], 0)
        seq4pred = np.append(seq4pred,predictions[i])
    else:
        seq4pred = sequences[i+1]

    # Compare the next_nr to the actual next number in the sequence. If the prediction is correct change the task template.
    if next_nr == str(predictions[i]):
        print("Actual:{}-Predicted:\033[92m{}\033[0m. Tokens_Used:{}".format(str(predictions[i]),next_nr,tokens_used))
        task = 'Correct! Congratulations! Now try with a new sequence:{}'
        prompt = task.format(seq4pred)
    # If the prediction differs by less than 1.5% of the actual next number in the sequence, change the task template.
    elif abs(int(next_nr)-predictions[i])/predictions[i] < 0.015:
        print("Actual:{}-Predicted:\033[93m{}\033[0m. Tokens_Used:{}".format(str(predictions[i]),next_nr,tokens_used))
        task = 'Close! You are getting better. The number that follows {} is {}. Now try with a new sequence:{}'
        prompt = task.format(sequences[i],predictions[i],seq4pred)
    # If the prediction is incorrect change the task template.
    else:
        print("Actual:{}-Predicted:\033[91m{}\033[0m. Tokens_Used:{}".format(str(predictions[i]),next_nr,tokens_used))
        task = 'Incorrect. The number that follows {} is {}. Now try with a new sequence:{}'
        prompt = task.format(sequences[i],predictions[i],seq4pred)
    
    messages.append({"role": "user", "content": prompt})
    
    # If the number of iterations is greater than MAX_MEMORY, pop the second question:answer pair from the messages list to limit the token use.
    if iterations > MAX_MEMORY*2:
        messages.pop(1)
        messages.pop(1)

    iterations += 1

    # Append the actual and predicted values to the lists.
    actual_list.append(predictions[i])
    predicted_list.append(int(next_nr))

    # If this is the last iteration, run the model for on more time to get the final prediction.
    if i == MAX_ITERATIONS-1:
        messages = messages
        next_nr,tokens_used = llm_call(messages)
        total_tokens_used.append(tokens_used)
        print("\033[94mFinal Prediction: {} => {}\033[0m".format(seq4pred,next_nr))

# Print the total tokens used.
print("Total tokens used: {}".format(sum(total_tokens_used)))

#-------------------EVALUATE MODEL-------------------#

actual_array = np.array(actual_list)
predicted_array = np.array(predicted_list)

mape = np.mean(np.abs((actual_array - predicted_array) / actual_array)) * 100
mape_rounded = round(mape, 2)

print("GPT-3.5-turbo Mean Absolute Percentage Error: {}%".format(mape_rounded))

#-------------------BENCHMARK MODEL-------------------#

# calculate the size of the test set as 20% of the data set
days_back = int(len(data)*0.2)

# Benchmark - Persistence model (predict the next value from the previous value)
lag = 1
values = pd.DataFrame(data[data_column].values)
dataframe = pd.concat([values.shift(lag), values], axis=1)
dataframe.columns = ['t-lag', 't+lag']
# split into train and test sets
X = dataframe.values
train, test = X[1:len(X)-days_back], X[len(X)-days_back:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# persistence model
def model_persistence(x):
    return x

# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
# Evaluate benchmark model
mape = np.mean(np.abs((test_y - predictions) / test_y)) * 100
mape_rounded = round(mape, 2)

print("Benchmark Mean Absolute Percentage Error: {}%".format(mape_rounded))

#-------------------PLOT THE RESULTS-------------------#

import matplotlib.pyplot as plt

# Create a dataframe with the actual and predicted values.
df = pd.DataFrame({'Actual': actual_list, 'Predicted': predicted_list})

# Calculate the difference between actual and predicted values.
df['Difference'] = np.abs(df['Actual'] - df['Predicted'])

# Plot the difference using a scatter plot with a polynomial line of best fit, and the line plot of the actual and the predicted values.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.scatter(df.index, df['Difference'])
ax1.plot(np.poly1d(np.polyfit(df.index, df['Difference'], 1))(df.index),"r--")
ax1.set_title('Difference between Actual and Predicted Values')
ax1.set_xlabel('Index')
ax1.set_ylabel('Error')

ax2.plot(df.index, df['Actual'], color='navy', label='Actual')
ax2.plot(df.index, df['Predicted'], color='orange', label='Predicted')
ax2.set_title('Actual and Predicted Values')
ax2.set_xlabel('Index')
ax2.set_ylabel('Value')
ax2.legend()

plt.show()
