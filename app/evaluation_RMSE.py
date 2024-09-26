import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

# Load the CSV file
data = pd.read_csv('C:/Users/safwa/OneDrive/Documents/EASY-SAPS-SANS/app/SAPS_answer.csv')
data = data.dropna()

# Convert data to a list of dictionaries for easier processing
patient_responses = data.to_dict('records')

item_mapping = {
    1: 'Pr_SAPS_AH',
    2: 'Pr_SAPS_VCom',
    3: 'Pr_SAPS_VCon',
    4: 'Pr_SAPS_SoH',
    5: 'Pr_SAPS_Olf_H',
    6: 'Pr_SAPS_VisHL',
    7: 'Pr_SAPS_D_Prsctn',
    8: 'Pr_SAPS_D_Jelsy',
    9: 'Pr_SAPS_D_Guilt_Sin',
    10: 'Pr_SAPS_D_Grnd',
    11: 'Pr_SAPS_D_Relgn',
    12: 'Pr_SAPS_D_Somtc',
    13: 'Pr_SAPS_D_Ref',
    14: 'Pr_SAPS_D_Cntrl',
    15: 'Pr_SAPS_D_MindRdn',
    16: 'Pr_SAPS_Thgt_Brd',
    17: 'Pr_SAPS_Thgt_Insrtn',
    18: 'Pr_SAPS_Thgt_Wthdrl',
    19: 'Pr_SAPS_BB_Clthg_Appr',
    20: 'Pr_SAPS_BB_Scl_Sxl_Bhvr',
    21: 'Pr_SAPS_BB_Aggr_Agitn',
    22: 'Pr_SAPS_BB_Reptv_Strtypd',
    23: 'Pr_SAPS_FTD_Drlmnt',
    24: 'Pr_SAPS_FTD_Tngtly',
    25: 'Pr_SAPS_FTD_Inchrnce',
    26: 'Pr_SAPS_FTD_Illgclty',
    27: 'Pr_SAPS_FTD_Crcmtntly',
    28: 'Pr_SAPS_FTD_PrSpch',
    29: 'Pr_SAPS_FTD_DsSpch',
    30: 'Pr_SAPS_FTD_Clng'
}


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
    P_diff = [max(P[i] - P[i + 1], 1e-10) for i in range(len(P) - 1)]  # Ensure non-zero probabilities
    return P_diff

def likelihood(theta, responses, items):
    """Calculate the likelihood of a given ability (theta) for all responses."""
    log_likelihood = 0
    for response, item in zip(responses, items):
        probs = grm_probability(theta, item['a'], item['b'])
        log_likelihood += np.log(probs[response])
    return -log_likelihood

def standard_error(total_info, scale_factor=100000):
    """Calculate the standard error of the ability estimate, scaled to be more user-friendly."""
    if total_info > 0:
        return (1 / np.sqrt(total_info)) * scale_factor
    else:
        return float('inf')

def select_next_item(theta, administered_items):
    """Select the next item based on current ability estimate (theta)."""
    remaining_items = [item for item in items if item not in administered_items]
    if not remaining_items:
        return None  # No more items to administer
    
    information = [item_information(theta, item) for item in remaining_items]
    max_info = max(information)
    
    if max_info == 0:
        # If no item provides information, choose randomly
        return np.random.choice(remaining_items)
    
    candidates = [item for item, info in zip(remaining_items, information) if info == max_info]
    return np.random.choice(candidates)

def item_information(theta, item):
    """Calculate the information for a single item at ability theta."""
    probs = grm_probability(theta, item['a'], item['b'])
    return item['a'] ** 2 * sum([(p - sum(probs[:k + 1])) ** 2 / max(p, 1e-10) for k, p in enumerate(probs)])

def total_information(theta, administered_items):
    """Calculate the total information for all administered items at ability theta."""
    return sum([item_information(theta, item) for item in administered_items])

def calculate_saps_score(responses):
    return sum(responses)

def run_cat(patient_response, se_threshold, max_items, min_items):
    theta = 0
    responses = []
    administered_items = []

    while True:
        item = select_next_item(theta, administered_items)
        if item is None or len(administered_items) >= max_items:
            break  # No more items to administer or max items reached
        
        administered_items.append(item)
        
        # Use actual patient response with the mapping
        column_name = item_mapping[item['id']]
        response = int(patient_response[column_name])
        responses.append(response)
        
        # Update theta (using MLE)
        result = minimize_scalar(lambda t: likelihood(t, responses, administered_items))
        theta = result.x
        total_info = total_information(theta, administered_items)
        se = standard_error(total_info)
        
        if len(administered_items) >= min_items and se < se_threshold:
            break

    return theta, administered_items, responses

# Gather evaluation criteria from user
se_threshold = float(input("Enter the standard error threshold (e.g., 0.1): "))
max_items = int(input("Enter the maximum number of items to be administered (e.g., 20): "))
min_items = int(input("Enter the minimum number of items to be administered (e.g., 5): "))

cat_scores = []
full_scores = []
individual_correlations = []
patient_rmses = []

for i, patient in enumerate(patient_responses):
    try:
        # Run CAT
        theta, administered_items, cat_responses = run_cat(patient, se_threshold, max_items, min_items)
        
        # Calculate CAT SAPS score
        cat_score = calculate_saps_score(cat_responses)
        cat_scores.append(cat_score)
        
        # Calculate full SAPS score
        full_responses = [int(patient[column]) for column in item_mapping.values()]
        full_score = calculate_saps_score(full_responses)
        full_scores.append(full_score)
        
        # Calculate individual RMSE
        rmse = np.sqrt(np.mean((np.array(cat_responses) - np.array(full_responses[:len(cat_responses)]))**2))
        patient_rmses.append(rmse)
        
        # Calculate individual correlation
        administered_item_ids = [item['id'] for item in administered_items]
        cat_full_responses = [full_responses[id-1] for id in administered_item_ids]
        individual_corr = np.corrcoef(cat_responses, cat_full_responses)[0, 1]
        individual_correlations.append(individual_corr)
        
        print(f"Patient {i+1}:")
        print(f"  CAT Score: {cat_score}")
        print(f"  Full Score: {full_score}")
        print(f"  Items administered: {len(administered_items)}")
        print(f"  RMSE: {rmse}")
        print()
        
    except Exception as e:
        print(f"Error processing patient {i+1}: {e}")
        continue

# Calculate overall metrics
if cat_scores and full_scores:
    overall_correlation = np.corrcoef(cat_scores, full_scores)[0, 1]
    mae = np.mean(np.abs(np.array(cat_scores) - np.array(full_scores)))
    rmse = np.sqrt(np.mean((np.array(cat_scores) - np.array(full_scores))**2))
    
    print("Overall Results:")
    print(f"Overall Correlation between CAT scores and full scores: {overall_correlation}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Total patients processed: {len(cat_scores)}")
else:
    print("No valid scores were calculated.")

# Gather rater's required RMSE
required_rmse = float(input("Enter the required RMSE threshold to filter patients: "))

# Filter patients based on required RMSE
filtered_patients = [(i+1, cat_scores[i], full_scores[i], patient_rmses[i]) for i in range(len(patient_rmses)) if patient_rmses[i] <= required_rmse]

print(f"\nPatients with RMSE <= {required_rmse}:")
for patient in filtered_patients:
    print(f"Patient {patient[0]}: CAT Score: {patient[1]}, Full Score: {patient[2]}, RMSE: {patient[3]}")