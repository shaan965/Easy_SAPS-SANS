import pandas as pd

# Load the CSV file
total_answer = pd.read_csv('total_answer.csv')

# Define the column mappings
saps_columns = [
    'Pr_SAPS_AH', 'Pr_SAPS_VCom', 'Pr_SAPS_VCon', 'Pr_SAPS_SoH',
    'Pr_SAPS_Olf_H', 'Pr_SAPS_VisHL', 'Pr_SAPS_D_Prsctn', 'Pr_SAPS_D_Jelsy',
    'Pr_SAPS_D_Guilt_Sin', 'Pr_SAPS_D_Grnd', 'Pr_SAPS_D_Relgn', 'Pr_SAPS_D_Somtc',
    'Pr_SAPS_D_Ref', 'Pr_SAPS_D_Cntrl', 'Pr_SAPS_D_MindRdn', 'Pr_SAPS_Thgt_Brd',
    'Pr_SAPS_Thgt_Insrtn', 'Pr_SAPS_Thgt_Wthdrl', 'Pr_SAPS_BB_Clthg_Appr',
    'Pr_SAPS_BB_Scl_Sxl_Bhvr', 'Pr_SAPS_BB_Aggr_Agitn', 'Pr_SAPS_BB_Reptv_Strtypd',
    'Pr_SAPS_FTD_Drlmnt', 'Pr_SAPS_FTD_Tngtly', 'Pr_SAPS_FTD_Inchrnce', 'Pr_SAPS_FTD_Illgclty',
    'Pr_SAPS_FTD_Crcmtntly', 'Pr_SAPS_FTD_PrSpch', 'Pr_SAPS_FTD_DsSpch', 'Pr_SAPS_FTD_Clng'
]

sans_columns = [
    'Pr_SANS_AFB_Unchg_F_Exp', 'Pr_SANS_AFB_Sptn_Mov', 'Pr_SANS_AFB_Exps_Gestr',
    'Pr_SANS_AFB_Pr_Eye_Cntct', 'Pr_SANS_AFB_Affctv_NonResp', 'Pr_SANS_AFB_Lck_Vocl_Inflctn',
    'Pr_SANS_AFB_Inappr_Affct', 'Pr_SANS_Alg_Pvrty_Spch', 'Pr_SANS_Alg_Pvrty_Spch_Cntnt',
    'Pr_SANS_Alg_Blckng', 'Pr_SANS_Alg_Incrsd_Rspns_Latncy', 'Pr_SANS_AvltnApthy_Grmg',
    'Pr_SANS_AvltnApthy_Imprstnce', 'Pr_SANS_AvltnApthy_Physcl_Anrg', 'Pr_SANS_AA_Recrtnl_Actvty',
    'Pr_SANS_AA_Sxl_Intrst', 'Pr_SANS_AA_Intmcy', 'Pr_SANS_AA_Reltn', 'Pr_SANS_Attn_Scl_Inattn',
    'Pr_SANS_Attn_Inattn_Mntl_Tst'
]

# Calculate the CAT SAP and SAN scores by summing the relevant columns
total_answer['CAT SAP'] = total_answer[saps_columns].sum(axis=1)
total_answer['CAT SAN'] = total_answer[sans_columns].sum(axis=1)

# Calculate the total scores and accuracy
results = []
for index, row in total_answer.iterrows():
    actual_sap = row['Actual SAP']
    cat_sap = row['CAT SAP']
    actual_san = row['Actual SAN']
    cat_san = row['CAT SAN']
    
    actual_total = actual_sap + actual_san
    cat_total = cat_sap + cat_san
    accuracy = cat_total / actual_total * 100
    
    results.append({
        'SL no': row['SL no'],
        'Actual SAP score': actual_sap,
        'CAT SAP score': cat_sap,
        'Actual SAN score': actual_san,
        'CAT SAN score': cat_san,
        'CAT Total': cat_total,
        'Actual Total': actual_total,
        'Accuracy (%)': accuracy
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print results for all patients
print(results_df)

# Save the results to a CSV file
results_df.to_csv('results.csv', index=False)