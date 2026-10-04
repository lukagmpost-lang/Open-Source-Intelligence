#!/usr/bin/env python3

with open('/Users/2950639/Desktop/OSI/Open-Source-Intelligence/src/osi/answer.py', 'r') as f:
    lines = f.readlines()

# Find the line after the docstring and initial checks
insert_line = None
for i, line in enumerate(lines):
    if line.strip() == 'data_prompt = _answer_data_prompt(result, asked)':
        insert_line = i + 1  # Insert after this line
        break

if insert_line is not None:
    # Insert the debug line
    lines.insert(insert_line, '    # DEBUG: Print that we\\'re in write_answer and about to make LLM call\n')
    lines.insert(insert_line + 1, '    print(f"[DEBUG] In write_answer, about to call _request with use_llm={use_llm}", file=sys.stderr)\n')
    
    # Write back to file
    with open('/Users/2950639/Desktop/OSI/Open-Source-Intelligence/src/osi/answer.py', 'w') as f:
        f.writelines(lines)
    
    print("Successfully added debug print at start of write_answer main logic")
else:
    print("Could not find the insertion point")