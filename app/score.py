import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


data = pd.read_csv('C:/Users/safwa/OneDrive/Documents/EASY-SAPS-SANS/app/SAPS_answer.csv')

responses_df = data.dropna()

# Example item parameters (a, [b1, b2, b3, b4, b5])
items = [{'id': 1, 'a': 1.5097045, 'b': [0.3285361,-0.005906699,0.2939168, 0.6907958,1.285917]},
         {'id': 2, 'a': 1.0320759, 'b': [0.8011340, 1.112417248, 1.2962282, 1.6516471, 2.453920]},
        {'id': 3, 'a': 1.4764997, 'b': [0.4774598, 0.702077625, 0.8356454, 1.0912027, 1.647400]},
        {'id': 4, 'a': 1.8366236, 'b': [1.7410411, 1.831936440 ,2.0211540, 2.2637153, 2.728590]},
        {'id': 5, 'a': 2.0366980, 'b': [1.7604304,1.797083593, 2.1709457,2.9410258,3.802192]},
        {'id': 6, 'a': 1.6193756, 'b': [1.7666369, 1.930214250,2.2633967,2.6677537,3.187222]},
        {'id': 7, 'a': 1.8911487, 'b': [-0.4461850, -0.173258803, 0.1063228,0.6160109,1.325337]},
        {'id': 8, 'a': 1.2516636, 'b': [2.6943297,2.769718505,2.8932336,3.0816022,3.822982]},
        {'id': 9, 'a': 2.8040102, 'b': [2.1472095, 2.247616150, 2.5514400, 2.8258061, 3.314969]},
        {'id': 10, 'a': 1.3596846, 'b': [2.2937446, 2.375191668, 2.5603201, 3.1086584, 3.599424]},
        {'id': 11, 'a': 2.0166317, 'b': [2.4471829,2.555025728,2.9669683,3.2725808,3.852966]},
        {'id': 12, 'a': 1.1768714, 'b': [2.8843484, 2.968298042,3.1069130,3.4564240,3.999298]},
        {'id': 13, 'a': 1.5243322, 'b': [-0.3200680,0.062348016,0.3939837,0.9026866,1.672415]},
        {'id': 14, 'a': 1.4222138, 'b': [1.4393246,1.617008242,1.7528912,2.0561043,2.937124]},
        {'id': 15, 'a': 1.5806995, 'b': [1.7345453,1.857007523,2.0401394,2.5163524,2.934104]},
        {'id': 16, 'a': 1.2947459, 'b': [1.4683481,1.605667735,1.7709595,2.3325678,3.412199]},
        {'id': 17, 'a': 2.2497791, 'b': [1.9739373, 2.193049914,2.3168613,2.6030299,2.965264]},
        {'id': 18, 'a': 2.0932872, 'b': [2.1152534,2.430683446,2.6108303,2.8621323,3.241668]},
        {'id': 19, 'a': 0.7729189, 'b': [1.8998935,2.470574221,3.3116072,4.7130681,6.539770]},
        {'id': 20, 'a': 1.0229372, 'b': [1.2981755,1.622649184,2.3763007,3.7609894,4.956147]},
        {'id': 21, 'a': 1.0286840, 'b': [0.2780794,0.628728710,1.2722875,2.2638927,3.330522]},
        {'id': 22, 'a': 1.2193015, 'b': [1.9322649,2.211082739,2.8399985,4.1210984,4.818710]},
        {'id': 23, 'a': 1.4501642, 'b': [1.7867459,2.011595992,2.3220479,3.0459060,3.836638]},
        {'id': 24, 'a': 1.6642272, 'b': [2.0070053,2.204916001,2.4953573,3.2356780,3.918205]},
        {'id': 25, 'a': 1.7603278, 'b': [1.9154725,2.095092560,2.4739141,3.3495848,4.124891]},
        {'id': 26, 'a': 2.0400808, 'b': [1.7982368,2.102744462,2.3887983,3.1923724,4.547116]},
        {'id': 27, 'a': 1.5583037, 'b': [2.1171632,2.391594467,2.8043904,3.6135243,4.438341]},
        {'id': 28, 'a': 3.0882680, 'b': [2.0510367,2.145899265,2.3289653,2.7553937,3.214954]},
        {'id': 29, 'a': 2.1055838, 'b': [1.9402122,2.092035317,2.5876428,3.6347072,4.509457]},
        {'id': 30, 'a': 4.5576695, 'b': [1.9539791,2.059858979,2.3288670,2.8506053,3.805913]}]


def grm_probability(theta, a, b):
    """Calculate the probabilities of responding in each category for GRM."""
    P = [1 / (1 + np.exp(-a * (theta - bk))) for bk in b]
    P = [0] + P + [1]  # Boundaries
    return [P[i] - P[i + 1] for i in range(len(P) - 1)]

def likelihood(theta, responses, items):
    """Calculate the likelihood of a given ability (theta) for all responses."""
    log_likelihood = 0
    for response, item in zip(responses, items):
        probs = grm_probability(theta, item['a'], item['b'])
        if response >= len(probs) or response < 0:
            print(f"Error: Response {response} is out of range for item {item['id']} with {len(probs)} categories.")
            response = min(max(response, 0), len(probs) - 1)
        log_likelihood += np.log(probs[int(response)])
    return -log_likelihood

def standard_error(total_info):
    return 1 / np.sqrt(total_info)

def select_next_item(theta, administered_items):
    """Select the next item based on current ability estimate (theta)."""
    remaining_items = [item for item in items if item not in administered_items]
    information = [item_information(theta, item) for item in remaining_items]
    max_info = max(information)
    candidates = [item for item, info in zip(remaining_items, information) if info == max_info]
    return np.random.choice(candidates)

def item_information(theta, item):
    """Calculate the information for a single item at ability theta."""
    probs = grm_probability(theta, item['a'], item['b'])
    return item['a'] ** 2 * sum([(p - sum(probs[:k + 1])) ** 2 / p for k, p in enumerate(probs)])

def total_information(theta, administered_items):
    """Calculate the total information for all administered items at ability theta."""
    return sum([item_information(theta, item) for item in administered_items])

def calculate_saps_score(patient_responses):
    # Initialize
    theta = 0
    responses = []
    administered_items = []
    se_threshold = 0.07

    # Simulate test-taking process
    while True:
        item = select_next_item(theta, administered_items)
        administered_items.append(item)
        
        response = patient_responses[item['id'] - 1]
        responses.append(int(response))  # Ensure response is an integer
        
        # Update theta (using MLE)
        result = minimize(lambda t: likelihood(t, responses, administered_items), theta)
        theta = result.x[0]
        total_info = total_information(theta, administered_items)
        se = standard_error(total_info)
        
        if se < se_threshold or len(administered_items) >= 20:
            break

    # Calculate SAPS score
    saps_score = sum(responses)
    num_items_administered = len(administered_items)
    return saps_score, num_items_administered

# Calculate total SAPS score (sum of all responses) for each patient
responses_df['Total_SAPS'] = responses_df.sum(axis=1)

# Calculate SAPS scores using CAT for all patients
saps_scores_cat = []
num_items_administered = []
for index, row in responses_df.iterrows():
    patient_responses = row[:-1].values  # Exclude the 'Total_SAPS' column
    saps_score, num_items = calculate_saps_score(patient_responses)
    saps_scores_cat.append(saps_score)
    num_items_administered.append(num_items)

# Add the CAT SAPS scores and number of items administered to the DataFrame
responses_df['CAT_SAPS'] = saps_scores_cat
responses_df['Num_Items_Administered'] = num_items_administered

# Calculate correlation
correlation, _ = pearsonr(responses_df['Total_SAPS'], responses_df['CAT_SAPS'])
print(f'Correlation: {correlation}')

mean_num_items_administered = np.mean(num_items_administered)
print(f'Mean number of items administered: {mean_num_items_administered}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(responses_df['Total_SAPS'], responses_df['CAT_SAPS']))
print(f'RMSE: {rmse}')

# Save the results to a new CSV file
# responses_df.to_csv('C:/Users/safwa/OneDrive/Documents/SAPS_scores_comparison.csv', index=False)