import json
import re
import matplotlib.pyplot as plt

def extract_json_objects(text):
    """
    Robust extraction of JSON objects from a string, 
    even if they are embedded in markdown or other text.
    """
    if not text:
        return []
    
    # 1. Clean up "thinking" tags if present (common in DeepSeek and others)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. Try to find content within markdown blocks
    code_blocks = re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if code_blocks:
        objs = []
        for block in code_blocks:
            try:
                loaded = json.loads(block)
                if isinstance(loaded, list):
                    objs.extend(loaded)
                else:
                    objs.append(loaded)
            except json.JSONDecodeError:
                # If the block itself isn't a single object/list, try searching for objects inside it
                objs.extend(extract_json_objects(block))
        if objs:
            return objs

    # 3. Fallback: Search for anything that looks like a JSON object or array
    results = []
    i = 0
    while i < len(text):
        if text[i] in '{[':
            found_at_this_pos = False
            for j in range(len(text), i, -1):
                try:
                    loaded = json.loads(text[i:j])
                    if isinstance(loaded, list):
                        results.extend(loaded)
                    else:
                        results.append(loaded)
                    i = j # Skip to the end of this object
                    found_at_this_pos = True
                    break
                except json.JSONDecodeError:
                    continue
            if not found_at_this_pos:
                i += 1
        else:
            i += 1
    return results

def plot_profits(weeks, basic_profits, llm_profits, model_name, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(weeks, basic_profits, label='Basic Vending Machine', marker='o', linestyle='--', color='blue')
    plt.plot(weeks, llm_profits, label=f'LLM Vending Machine ({model_name})', marker='s', linestyle='-', color='green')
    
    plt.title(f'Weekly Net Profit Comparison ({model_name})')
    plt.xlabel('Week')
    plt.ylabel('Net Profit/Loss')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.savefig(filename)
    print(f"\nProfit graph saved as {filename}")
    plt.close()

def plot_multi_profits(weeks, basic_profits, model_a_profits, model_b_profits, model_a_name, model_b_name, filename='multi_profit_comparison.png'):
    plt.figure(figsize=(12, 7))
    plt.plot(weeks, basic_profits, label='Basic Vending Machine', marker='o', linestyle='--', color='blue')
    plt.plot(weeks, model_a_profits, label=f'Model A ({model_a_name})', marker='s', linestyle='-', color='green')
    plt.plot(weeks, model_b_profits, label=f'Model B ({model_b_name})', marker='^', linestyle='-', color='red')
    
    plt.title('Multi-Model Weekly Net Profit Comparison')
    plt.xlabel('Week')
    plt.ylabel('Net Profit/Loss')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.savefig(filename)
    print(f"\nMulti-model profit graph saved as {filename}")
    plt.close()
