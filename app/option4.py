import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

# Load the CSV file
data = pd.read_csv('C:/Users/safwa/OneDrive/Documents/EASY-SAPS-SANS/app/SAPS_answer.csv')
data = data.dropna()

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=51/457, random_state=21)

# Item mapping
item_mapping = {
    1: 'Pr_SAPS_AH', 2: 'Pr_SAPS_VCom', 3: 'Pr_SAPS_VCon', 4: 'Pr_SAPS_SoH',
    5: 'Pr_SAPS_Olf_H', 6: 'Pr_SAPS_VisHL', 7: 'Pr_SAPS_D_Prsctn', 8: 'Pr_SAPS_D_Jelsy',
    9: 'Pr_SAPS_D_Guilt_Sin', 10: 'Pr_SAPS_D_Grnd', 11: 'Pr_SAPS_D_Relgn', 12: 'Pr_SAPS_D_Somtc',
    13: 'Pr_SAPS_D_Ref', 14: 'Pr_SAPS_D_Cntrl', 15: 'Pr_SAPS_D_MindRdn', 16: 'Pr_SAPS_Thgt_Brd',
    17: 'Pr_SAPS_Thgt_Insrtn', 18: 'Pr_SAPS_Thgt_Wthdrl', 19: 'Pr_SAPS_BB_Clthg_Appr',
    20: 'Pr_SAPS_BB_Scl_Sxl_Bhvr', 21: 'Pr_SAPS_BB_Aggr_Agitn', 22: 'Pr_SAPS_BB_Reptv_Strtypd',
    23: 'Pr_SAPS_FTD_Drlmnt', 24: 'Pr_SAPS_FTD_Tngtly', 25: 'Pr_SAPS_FTD_Inchrnce',
    26: 'Pr_SAPS_FTD_Illgclty', 27: 'Pr_SAPS_FTD_Crcmtntly', 28: 'Pr_SAPS_FTD_PrSpch',
    29: 'Pr_SAPS_FTD_DsSpch', 30: 'Pr_SAPS_FTD_Clng'
}

def grm_probability(theta, a, b):
    """Calculate the probabilities of responding in each category for GRM."""
    P = [1 / (1 + np.exp(-a * (theta - bk))) for bk in b]
    P = [1] + P + [0]  # Boundaries
    return [P[i] - P[i+1] for i in range(len(P) - 1)]

def likelihood(theta, responses, items):
    """Calculate the likelihood of a given ability (theta) for all responses."""
    log_likelihood = 0
    epsilon = 1e-10  # Small value to avoid log(0)
    for response, item in zip(responses, items):
        probs = grm_probability(theta, item['a'], item['b'])
        if int(response) >= len(probs):
            prob = epsilon
        else:
            prob = np.clip(probs[int(response)], epsilon, 1 - epsilon)  # Avoid log(0)
        log_likelihood += np.log(prob)
    return -log_likelihood

def standard_error(total_info):
    return 1 / np.sqrt(max(total_info, 1e-10))

def select_next_item(theta, administered_items, all_items):
    """Select the next item based on maximum information at current theta."""
    remaining_items = [item for item in all_items if item not in administered_items]
    information = [item_information(theta, item) for item in remaining_items]
    return remaining_items[np.argmax(information)]

def item_information(theta, item):
    """Calculate the information for a single item at ability theta."""
    probs = grm_probability(theta, item['a'], item['b'])
    info = item['a'] ** 2 * sum([(p - sum(probs[:k + 1])) ** 2 / max(p, 1e-10) for k, p in enumerate(probs)])
    return max(info, 1e-10)

def total_information(theta, administered_items):
    """Calculate the total information for all administered items at ability theta."""
    return sum([item_information(theta, item) for item in administered_items])

def train_model(train_data):
    items = []
    for i in range(1, 31):
        column = item_mapping[i]
        responses = train_data[column].values
        a = np.random.uniform(0.5, 2.0)  # Random discrimination parameter
        max_response = int(responses.max())
        b = np.sort(np.random.uniform(-3, 3, max_response))  # Random difficulty parameters
        items.append({'id': i, 'a': a, 'b': b})
    return items

# Initialize
items = train_model(train_data)
se_threshold = 0.45
max_items = 30
min_items = 5

# Simulate test-taking process using the test data
test_patient_responses = test_data.values.tolist()

# Store the results
results = []

for patient_id, patient_responses in enumerate(test_patient_responses, 1):
    responses = []
    administered_items = []
    theta = 0  # Initial theta estimate
    se = float('inf')
    
    print(f"\nPatient {patient_id}:")
    
    for item_count in range(1, max_items + 1):
        item = select_next_item(theta, administered_items, items)
        administered_items.append(item)
        
        # Get the actual response from the test data
        response = int(patient_responses[item['id'] - 1])  # Ensure response is an integer
        responses.append(response)
        
        # Update theta (using MLE)
        result = minimize_scalar(
            lambda t: likelihood(t, responses, administered_items),
            method='bounded',
            bounds=(-4, 4)
        )
        new_theta = result.x
        total_info = total_information(new_theta, administered_items)
        new_se = standard_error(total_info)
        
        print(f"  Item {item_count}: id={item['id']}, response={response}, theta={new_theta:.2f}, se={new_se:.2f}")
        
        if item_count >= min_items and abs(new_theta - theta) < 0.01 and new_se < se_threshold:
            break
        
        theta = new_theta
        se = new_se

    # Calculate the total SAPS score from the patient's responses
    total_saps_score = sum(patient_responses)
    
    # Store the results for this patient
    results.append({
        "patient_id": patient_id,
        "theta": theta,
        "se": se,
        "administered_items": len(administered_items),
        "total_saps_score": total_saps_score
    })
    
    print(f"  Final: theta={theta:.2f}, se={se:.2f}, items={len(administered_items)}, total_saps={total_saps_score}")

# Analyze results
cat_scores = [result['theta'] for result in results]
saps_scores = [result['total_saps_score'] for result in results]
num_items = [result['administered_items'] for result in results]

correlation, _ = pearsonr(cat_scores, saps_scores)
mean_items = np.mean(num_items)
accuracy = sum(abs(cat - saps) <= 3 for cat, saps in zip(cat_scores, saps_scores)) / len(results)

print(f"\nCorrelation between CAT and total SAPS scores: {correlation:.2f}")
print(f"Mean number of items administered: {mean_items:.2f}")
print(f"Accuracy within ±3 points: {accuracy:.2%}")


plt.hist(num_items, bins=range(min(num_items), max(num_items) + 2, 1))
plt.xlabel('Number of Items Administered')
plt.ylabel('Frequency')
plt.title('Distribution of Items Administered')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Test complete.")