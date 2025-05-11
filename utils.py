import random
def extract_search_procedure(text):
    lines = text.split('\n')
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('|-') or line.startswith(' |-'):
            start_idx = i
            break
    
    if start_idx is not None:
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            if not (line.startswith('|-') or line.startswith(' |-')):
                end_idx = i
                break
        
        if end_idx is not None:
            search_procedure = '\n'.join(lines[start_idx:end_idx])
            indentation_levels = []
            for i in range(start_idx, end_idx):
                spaces = len(lines[i]) - len(lines[i].lstrip())
                level = spaces
                indentation_levels.append(level)
        else:
            search_procedure = '\n'.join(lines[start_idx:])
            indentation_levels = []
            for i in range(start_idx, len(lines)):
                spaces = len(lines[i]) - len(lines[i].lstrip())
                level = spaces
                indentation_levels.append(level)
        
        return search_procedure, indentation_levels
    
    return None, None

def sag_extract(search_proc, indentation):
    try:
        tuples = []
        lines = search_proc.split('\n')
        deepest_level = max(indentation)
        print("LINES", len(lines))
        for k in range(1, len(lines)-1):
            if indentation[k-1] < indentation[k] and indentation[k] < indentation[k+1]:
                state = '\n'.join(lines[:k])
                
                action = lines[k]
                
                for j in range(k+1, len(lines)):
                    if indentation[j] == indentation[k]:
                        break
                    else:    
                        goal = lines[j]
                        tuples.append((state, action, goal))

        print(f"Extracted {len(tuples)} (state, action, goal) tuples")
        
        if tuples:
            sample_idx = random.randint(0, len(tuples)-1)
            print(f"Sample tuple {sample_idx}:")
            print(f"STATE:\n{tuples[sample_idx][0]}")
            print(f"ACTION:\n{tuples[sample_idx][1]}")
            print(f"GOAL:\n{tuples[sample_idx][2]}")

        return tuples
    except Exception as e:
        print(f"Error in sag_extract: {e}")
        return []