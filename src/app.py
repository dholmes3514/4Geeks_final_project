'''Web app for price and turnaround time prediction'''

import pickle

# Flask stuff (wait for later)


# Load assets (Label encoder, one-hot encoder, price model, TT model)
with open('data/encoder.pkl', 'rb') as input_file:
    encoder = pickle.load(input_file)

with open('data/labeler.pkl', 'rb') as input_file:
    labeler = pickle.load(input_file)


#############
# Functions #
#############

def preprocess_input(state, postal_code, ship_mode, category, subcategory, encoder, labeler):
    '''Does preprocessing of user input, returns single row Pandas df for inference'''

    print('Running the input preprocessing function...')

    result = 'placeholder'

    return result


def predict_price(input_data):
    '''Takes preprocessed input data, runs price prediction, returns predicted price'''

    print('Running the price prediction function...')

    result = 'placeholder'

    return result


def predict_tt(input_data):
    '''Takes preprocessed input data, runs price prediction, returns predicted turnaround time'''

    print('Running the turnaround time prediction function...')

    result = 'placeholder'

    return result


# Test values (fake user input) - pretend these came from flask
order_date = '8/11/2017'
ship_date = '11/11/2017'
ship_mode = 'Second Class'
state = 'Kentucky'
postal_code = '42420'
category = 'Furniture'
sub_category = 'Bookcases'
sales = '261.96'

# 'Main fence'
if __name__ == '__main__':

    print('Running sales & TT app...\n')

    # Preprocess with input preprocessing function
    preprocessed_data = preprocess_input(
        order_date,
        state,
        postal_code,
        category,
        sub_category,
        encoder,
        labeler
    )

    print(f'Result from input preprocessing function: {preprocessed_data}\n')


    # Predict price
    predicted_price = predict_price(preprocessed_data)

    print(f'Result from price prediction function: {predicted_price}\n')


    # Predict turnaround time
    predicted_turnaround_time = predict_tt(preprocessed_data)

    print(f'Result from turnaround time prediction function: {predicted_turnaround_time}\n')

    print('Done.')