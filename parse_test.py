import re

def parse_xml_function_call(text):
    if '<function_calls>' not in text:
        return []
    
    # isolate the function_calls block
    match = re.search(r'<function_calls>(.*?)</function_calls>', text, re.DOTALL)
    if not match:
        return []
        
    block = match.group(1)
    
    # find invokes
    invokes = re.findall(r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', block, re.DOTALL)
    tool_calls = []
    
    for i, (func_name, args_block) in enumerate(invokes):
        # basic arg parsing
        args = {}
        if args_block.strip():
            # try to parse args like <foo>bar</foo>
            arg_matches = re.findall(r'<([^>]+)>(.*?)</\1>', args_block, re.DOTALL)
            for k, v in arg_matches:
                args[k] = v.strip()
        
        tool_calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": args
            }
        })
    return tool_calls

print(parse_xml_function_call('<function_calls> <invoke name="address_path"> </invoke> </function_calls>'))
