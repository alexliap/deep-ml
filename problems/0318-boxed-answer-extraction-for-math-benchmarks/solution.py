def extract_boxed_answer(response: str) -> str:
    """
    Extract the answer from within \boxed{...} in a model response.
    
    Args:
        response: The model's text response containing a boxed answer
    
    Returns:
        The content inside the last \boxed{}, or empty string if not found
    """
    marker = "\\boxed"
    end = len(response)
    while True:
        start = response.rfind(marker, 0, end)
        if start == -1:
            return ""
        end = start

        i = start + len(marker)
        while i < len(response) and response[i].isspace():
            i += 1
        if i >= len(response) or response[i] != "{":
            continue

        depth = 0
        j = i
        while j < len(response):
            c = response[j]
            if c == "\\":
                j += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return response[i + 1:j].strip()
            
            j += 1
