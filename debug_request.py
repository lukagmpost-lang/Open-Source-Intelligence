#!/usr/bin/env python3

with open('/Users/2950639/Desktop/OSI/Open-Source-Intelligence/src/osi/answer.py', 'r') as f:
    lines = f.readlines()

# Find the _request function
start_line = None
end_line = None

for i, line in enumerate(lines):
    if line.strip() == 'def _request(prompt: str) -> str | None:':
        start_line = i
        break

if start_line is not None:
    # Find the end of the function (next function definition or end of indent)
    for i in range(start_line + 1, len(lines)):
        if lines[i].startswith('    def ') or lines[i].startswith('class ') or (not lines[i].startswith('    ') and lines[i].strip() != ''):
            end_line = i
            break
    else:
        end_line = len(lines)
    
    if end_line is not None:
        # Replace the function
        new_lines = [
            '    def _request(prompt: str) -> str | None:\n',
            '        try:\n',
            '            result = call_llm(\n',
            '                prompt,\n',
            '                run_id=run_id,\n',
            '                n_nodes=_size(result, \"n_nodes\", \"nodes\"),\n',
            '                n_edges=_size(result, \"n_edges\", \"edges\"),\n',
            '                question=asked,\n',
            '                system=answer_system_prompt(domain, source),\n',
            '            )\n',
            '            # DEBUG: Print raw LLM output\n',
            '            print(f\"[DEBUG] Raw LLM output: {repr(result)}\", file=sys.stderr)\n',
            '            return result\n',
            '        except (OSError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError, ValueError):\n',
            '            return None\n',
            '\n'
        ]
        
        # Replace the old function with new one
        lines = lines[:start_line] + new_lines + lines[end_line:]
        
        # Write back to file
        with open('/Users/2950639/Desktop/OSI/Open-Source-Intelligence/src/osi/answer.py', 'w') as f:
            f.writelines(lines)
        
        print("Successfully added debug print to _request function")
    else:
        print("Could not find the end of the _request function")
else:
    print("Could not find the _request function")