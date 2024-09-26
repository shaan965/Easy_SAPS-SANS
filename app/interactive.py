import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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


# def select_next_item(theta, administered_items):
#     """Select the next item based on current ability estimate (theta)."""
#     remaining_items = [item for item in items if item not in administered_items]
#     if not remaining_items:
#         return None  # No more items to administer
    
#     information = [item_information(theta, item) for item in remaining_items]
#     max_info = max(information)
    
#     if max_info == 0:
#         # If no item provides information, choose randomly
#         return np.random.choice(remaining_items)
    
#     candidates = [item for item, info in zip(remaining_items, information) if info == max_info]
#     return np.random.choice(candidates)

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
    remaining_items = [item for item in items if item not in administered_items]
    if not remaining_items:
        return None  # No more items to administer

    information = [item_information(theta, item) for item in remaining_items]
    max_info = max(information)
    
    # Select item with max information but consider a range around it to avoid overfitting
    candidates = [item for item, info in zip(remaining_items, information) if info >= max_info * 0.9]
    
    # Add some randomness to avoid consistently overestimating
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


def calculate_accuracy(cat_score, full_score, threshold_score):
    if full_score >= threshold_score:
        return 1 if cat_score >= threshold_score else 0
    else:
        return 1 if cat_score < threshold_score else 0
    
def run_cat(patient_response, max_items, min_items, threshold_score, required_accuracy):
    theta = 0  # Initial guess for theta
    responses = []
    administered_items = []

    for k in range(1, max_items + 1):
        item = select_next_item(theta, administered_items)
        if item is None:
            print(f"No more items to administer at iteration {k}.")
            break

        administered_items.append(item)
        column_name = item_mapping[item['id']]
        response = int(patient_response[column_name])
        responses.append(response)

        # Update theta estimate with more conservative step sizes
        result = minimize_scalar(lambda t: likelihood(t, responses, administered_items), bounds=(-6, 6), method='bounded')
        theta = result.x

        # Calculate predicted score
        v = calculate_saps_score(responses)

        # Find matching patients in training data
        matching_patients = [
            p for p in patient_responses
            if all(int(p[item_mapping[item['id']]]) == resp for item, resp in zip(administered_items, responses))
        ]

        if len(matching_patients) > 0:
            correct_predictions = sum(
                1 for p in matching_patients 
                if (v >= threshold_score) == (calculate_saps_score([int(p[col]) for col in item_mapping.values()]) >= threshold_score)
            )
            accuracy = correct_predictions / len(matching_patients)
            
            # Adjust condition to ensure more items are administered
            if accuracy >= required_accuracy and k >= min_items:
                return v, administered_items, responses

    return v, administered_items, responses


def run_cat_with_new_stopping_rule(patient_response, max_items, min_items, threshold_score=None, allowed_worst_error=None, required_accuracy=None):
    theta = 0
    responses = []
    administered_items = []
    
    for k in range(1, max_items + 1):
        item = select_next_item(theta, administered_items)
        if item is None:
            break
        
        administered_items.append(item)
        column_name = item_mapping[item['id']]
        response = int(patient_response[column_name])
        responses.append(response)
        
        result = minimize_scalar(lambda t: likelihood(t, responses, administered_items), bounds=(-4, 4), method='bounded')
        theta = result.x
        
        # Calculate predicted score
        v = calculate_saps_score(responses)
        
        # Find matching patients in training data
        matching_patients = [p for p in patient_responses if all(int(p[item_mapping[item['id']]]) == resp for item, resp in zip(administered_items, responses))]
        
        if len(matching_patients) > 0:
            if threshold_score is not None:
                # Option 2: Threshold of Score
                correct_predictions = sum(1 for p in matching_patients if (v >= threshold_score) == (calculate_saps_score([int(p[col]) for col in item_mapping.values()]) >= threshold_score))
                accuracy = correct_predictions / len(matching_patients)
                
                if accuracy >= required_accuracy and k >= min_items:
                    return v, administered_items, responses
            
            elif allowed_worst_error is not None:
                # Option 3: Worst Error
                good_predictions = sum(1 for p in matching_patients if abs(v - calculate_saps_score([int(p[col]) for col in item_mapping.values()])) <= allowed_worst_error)
                accuracy = good_predictions / len(matching_patients)
                
                if accuracy >= required_accuracy and k >= min_items:
                    return v, administered_items, responses
    
    return v, administered_items, responses

# Step 1: Present options to the user
print("Choose an option:")
print("1. Number of Questions")
print("2. Threshold of Score")
print("3. Worst Error")
option = int(input("Enter the option number (1/2/3): "))

if option == 1:
    # Option 1: Number of Questions
    max_items = int(input("Enter the maximum number of items to be administered (e.g., 20): "))
    min_items = 1

    irt_rmse_data = []
    random_rmse_data = []

    for num_questions in range(min_items, max_items + 1):
        irt_scores = []
        random_scores = []
        full_scores = []

        for patient in patient_responses:
            # IRT-based selection
            theta, administered_items, cat_responses = run_cat_with_new_stopping_rule(patient, num_questions, num_questions)
            irt_score = calculate_saps_score(cat_responses)
            irt_scores.append(irt_score)

            # Random selection
            random_items = np.random.choice(items, num_questions, replace=False)
            random_responses = [int(patient[item_mapping[item['id']]]) for item in random_items]
            random_score = calculate_saps_score(random_responses)
            random_scores.append(random_score)

            # Full score
            full_responses = [int(patient[column]) for column in item_mapping.values()]
            full_score = calculate_saps_score(full_responses)
            full_scores.append(full_score)

        irt_rmse = np.sqrt(np.mean((np.array(irt_scores) - np.array(full_scores))**2))
        random_rmse = np.sqrt(np.mean((np.array(random_scores) - np.array(full_scores))**2))

        irt_rmse_data.append(irt_rmse)
        random_rmse_data.append(random_rmse)

    # Calculate average RMSE for both methods
    avg_irt_rmse = np.mean(irt_rmse_data)
    avg_random_rmse = np.mean(random_rmse_data)

    # Visualization
    question_counts = list(range(min_items, max_items + 1))

    # Create smooth curves using interpolation
    x_smooth = np.linspace(min_items, max_items, 300)
    irt_smooth = interp1d(question_counts, irt_rmse_data, kind='cubic')(x_smooth)
    random_smooth = interp1d(question_counts, random_rmse_data, kind='cubic')(x_smooth)

    plt.figure(figsize=(10, 6))
    plt.plot(x_smooth, irt_smooth, label='IRT-based selection', color='blue')
    plt.plot(x_smooth, random_smooth, label='Random selection', color='red')
    plt.scatter(question_counts, irt_rmse_data, color='blue', alpha=0.5)
    plt.scatter(question_counts, random_rmse_data, color='red', alpha=0.5)

    plt.title('RMSE vs Number of Questions')
    plt.xlabel('Number of Questions')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    print(f"\nDone: Final RMSE (IRT-based) = {irt_rmse_data[-1]:.2f}")
    print(f"Final RMSE (Random) = {random_rmse_data[-1]:.2f}")
    print(f"Average RMSE (IRT-based) = {avg_irt_rmse:.2f}")
    print(f"Average RMSE (Random) = {avg_random_rmse:.2f}")

elif option == 2:
    # Implementing the Threshold of Score Option
    cat_scores = []
    full_scores = []
    question_count = []
    threshold_score = float(input("Enter the threshold score: "))
    required_accuracy = float(input("Enter the required accuracy (e.g., 0.95 for 95%): "))
    max_items = 30
    min_items = 1  # Setting a minimum number of items

    for i, patient in enumerate(patient_responses):
        try:
            cat_score, administered_items, cat_responses = run_cat(
                patient, max_items, min_items, threshold_score, required_accuracy
            )

            cat_scores.append(cat_score)
            full_responses = [int(patient[column]) for column in item_mapping.values() if column in patient]
            full_score = calculate_saps_score(full_responses)
            full_scores.append(full_score)
            question_count.append(len(administered_items))

        except Exception as e:
            print(f"Error processing patient {i + 1}: {str(e)}")

    # Calculate accuracy
    accuracy = sum(1 for cat, full in zip(cat_scores, full_scores) if (cat >= threshold_score) == (full >= threshold_score)) / len(cat_scores)

    print(f"\nDone: Accuracy obtained = {accuracy:.2f}")
    print(f"Average # of questions: {np.mean(question_count):.2f}")

    # Histogram of questions asked
    plt.hist(question_count, bins=range(1, max(question_count) + 2), edgecolor='black')
    plt.title('Histogram of # of Questions Asked')
    plt.xlabel('Number of Questions Asked')
    plt.ylabel('Number of Patients')
    plt.show()

    # Scatter plot of predicted vs true score
    plt.scatter(cat_scores, full_scores, alpha=0.6)
    plt.plot([min(cat_scores), max(cat_scores)], [min(cat_scores), max(cat_scores)], color='red', linestyle='--')
    plt.title('Scatter Plot of Predicted vs True Score')
    plt.xlabel('Predicted Score')
    plt.ylabel('True Score')
    plt.show()

elif option == 3:
    # Option 3: Worst Error
    cat_scores = []
    full_scores = []
    worst_errors = []
    question_count = []
    allowed_worst_error = float(input("What is allowed worst error? "))
    required_accuracy = float(input("Required accuracy (e.g., 0.9 for 90%): "))
    max_items = 25
    min_items = 1

    for i, patient in enumerate(patient_responses):
        try:
            cat_score, administered_items, cat_responses = run_cat_with_new_stopping_rule(
                patient, max_items, min_items, allowed_worst_error=allowed_worst_error, required_accuracy=required_accuracy
            )
            
            cat_scores.append(cat_score)
            full_responses = [int(patient[column]) for column in item_mapping.values()]
            full_score = calculate_saps_score(full_responses)
            full_scores.append(full_score)
            worst_error = abs(cat_score - full_score)
            worst_errors.append(worst_error)
            question_count.append(len(administered_items))
        
        except Exception as e:
            print(f"Error processing patient {i + 1}: {str(e)}")
    
    # Calculate accuracy and RMSE
    accuracy = sum(1 for error in worst_errors if error <= allowed_worst_error) / len(worst_errors)
    rmse = np.sqrt(np.mean((np.array(cat_scores) - np.array(full_scores))**2))
    
    print(f"\nDone: Accuracy obtained = {accuracy:.2f}")
    print(f"RMSE = {rmse:.2f}")
    print(f"Average # of questions: {np.mean(question_count):.2f}")
    # Scatter plot of predicted vs true score
    plt.scatter(cat_scores, full_scores, alpha=0.6)
    plt.plot([min(cat_scores), max(cat_scores)], [min(cat_scores), max(cat_scores)], color='red', linestyle='--')
    plt.title('Scatter Plot of Predicted vs True Score')
    plt.xlabel('Predicted Score')
    plt.ylabel('True Score')
    plt.show()

else:
    print("Invalid option selected.")