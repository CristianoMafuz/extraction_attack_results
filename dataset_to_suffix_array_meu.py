from transformers import AutoTokenizer
import json

def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

import json

def load_jsonl_file(filepath):
    valid_lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
                valid_lines.append(obj)
            except json.JSONDecodeError as e:
                print(f"JSON inválido na linha {i+1}: {e}")
                print(f"Preview da linha: {line[:100]}...\n")
                continue  # Pula a linha com erro
    return valid_lines


def binary_search(pat, txt, suffArr, n):
    '''
    This function runs a binary search in the original text using the suffix array searching a pattern.
    Input:
        pat: Pattern to be found in the text;
        txt: Original text (dataset)
        suffArr: Suffix Array of 'txt'
        n: len(txt)
    Output:
        True: The pattern 'pat' is contained in the original text 'txt' (dataset)
        False: The pattern 'pat' is NOT contained in the original text 'txt' (dataset)
   '''

    # Get the length of the pattern
    m = len(pat)
     
    # Initialize left and right indexes
    l = 0
    r = n-1
     
    # Do simple binary search for the pat in txt using the built suffix array
    while l <= r:
       
        # Find the middle index of the current subarray
        mid = l + (r - l)//2
         
        # Get the substring of txt starting from suffArr[mid] and of length m
        res = txt[suffArr[mid]:suffArr[mid]+m]
         
        # If the substring is equal to the pattern
        if res == pat:
           
            # Print the index and return
            return True
           
        # If the substring is less than the pattern
        if res < pat:
           
            # Move to the right half of the subarray
            l = mid + 1
        else:
           
            # Move to the left half of the subarray
            r = mid - 1
             
    # If the pattern is not found
    return False

def count_tokens_verbatim(output, S, A, tokenizer):
    output = tokenizer.tokenize(output)
    output = [x for x in output if x != '<0x0A>']

    pattern = ''
    v_tokens_list = []
    v_tokens = 0

    for j in range(len(output)):
        pattern += output[j]
        if binary_search(pattern.replace("▁", " "), S, A, len(S)):
            v_tokens += 1
            if j == (len(output) - 1) and v_tokens > 0:
                if v_tokens > 10:
                    v_tokens_list.append([v_tokens, pattern.replace("▁", " ")])
        else:
            if v_tokens > 10:
                v_tokens_list.append([v_tokens, pattern[:-len(output[j])].replace("▁", " ")])
            v_tokens = 0
            pattern = output[j]

    return v_tokens_list

# Carregar tokenizer e dados
tokenizer = AutoTokenizer.from_pretrained('pucpr-br/Clinical-BR-Mistral-7B-v0.2')
dataset = load_json_file("dataset-suffix-array.json")
S = dataset["string"]
A = dataset["suffix_array"]
responses = load_jsonl_file("attack_results_nucleus(6)_mistral.jsonl")

# Processar e salvar somente os trechos com >10 tokens
filtered_results = []
for idx, item in enumerate(responses):
    output_text = item["response"]
    matches = count_tokens_verbatim(output_text, S, A, tokenizer)
    if matches:
        filtered_results.append({
            "id": idx,
            "matches_above_10_tokens": matches
        })
        print(f"Encontrado na resposta {idx} com {len(matches)} trechos")

# Salvar os resultados
with open("verbatim_attack_results_nucleus(6)_mistral.jsonl", "w", encoding="utf-8") as f:
    json.dump(filtered_results, f, indent=2, ensure_ascii=False)

print("Salvo em verbatim_matches.json")
