import pandas as pd
from xpinyin import Pinyin
from fuzzywuzzy import fuzz
from concurrent.futures import ProcessPoolExecutor

def get_pinyin(name):
    if pd.isna(name): 
        return ''
    
    p = Pinyin()
    return p.get_pinyin(name, '')

def calculate_similarity(str1, str2):
    return fuzz.token_sort_ratio(str1, str2)

def map_tables_parallel(chunk, table2, name_threshold=80, specialty_threshold=80, name_weight=0.6, specialty_weight=0.4):
    mapping_results_chunk = []

    for index1, row1 in chunk.iterrows():
        for index2, row2 in table2.iterrows():
            
            name_similarity = calculate_similarity(row1['Pinyin'], row2['Pinyin'])
            specialty_similarity = calculate_similarity(row1['hcp_specialty_name'], row2['specialty_1'])

            
            weighted_similarity = (name_weight * name_similarity) + (specialty_weight * specialty_similarity)

            
            if name_similarity >= name_threshold and specialty_similarity >= specialty_threshold:
                
                mapping_results_chunk.append({'processed_hcp': row1['processed_hcp'], 'se_id': row2['se_id'], 'Similarity': weighted_similarity})

    return mapping_results_chunk

if __name__ == "__main__":

    table1 = pd.read_csv('veeva_hcp_demographic_profile.csv')
    table2 = pd.read_csv('influential_hcp.csv')

    table1['Pinyin'] = table1['hcp_formatted_name__v'].apply(get_pinyin)
    table2['Pinyin'] = table2['original_full_name'].apply(get_pinyin)

    name_threshold = 80
    specialty_threshold = 70
    name_weight = 0.7
    specialty_weight = 0.3

    chunk_size = 1000
    chunks = [table1[i:i + chunk_size] for i in range(0, len(table1), chunk_size)]

    # Process chunks in parallel
    with ProcessPoolExecutor() as executor:
        # Pass table2 as an argument to the function
        mapping_results = list(executor.map(map_tables_parallel, chunks, [table2]*len(chunks), [name_threshold]*len(chunks), [specialty_threshold]*len(chunks), [name_weight]*len(chunks), [specialty_weight]*len(chunks)))

    # Flatten the list of lists into a single list
    mapping_results = [item for sublist in mapping_results for item in sublist]
   
    result_df = pd.DataFrame(mapping_results)

    result_df.to_csv('output70.csv', index=False)