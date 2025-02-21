
# pred = ['6','x','2','-','2','x','-','-','0']
#pred = ['6','*','2']

import re
from collections import Counter

def correct_equation(predicted_eq):
    # Ensure equation has an "=" sign, otherwise assume "=0"
    if "=" not in predicted_eq:
        predicted_eq += "=0"

    # Extract variables and determine the most frequent one
    variables = re.findall(r'[a-zA-Z]', predicted_eq)
    if not variables:
        return None  # No valid variable found
    main_var = Counter(variables).most_common(1)[0][0]  # Pick the most common variable

    # Replace incorrect variable names (e.g., y → x)
    corrected_eq = re.sub(r'[a-zA-Z]', main_var, predicted_eq)

    # Ensure multiplication between coefficients and variables (e.g., 5x → 5*x)
    corrected_eq = re.sub(r'(\d)(?=' + main_var + r')', r'\1*', corrected_eq)

    # Fix cases where exponents are missing (e.g., xx → x^2)
    corrected_eq = re.sub(r'(\b' + main_var + r')(\1+)', lambda m: f"{main_var}^{len(m.group(2)) + 1}", corrected_eq)

    # Fix standalone numbers attached to variables (e.g., x3 → x^3)
    corrected_eq = re.sub(r'(\b' + main_var + r')(\d+)', r'\1^\2', corrected_eq)

    # Convert `^` to `**` for SymPy compatibility
    corrected_eq = corrected_eq.replace("^", "**")

    # Replace `x**0` with `1`
    corrected_eq = re.sub(rf'{main_var}\*\*0', '1', corrected_eq)

    # Simplify `6*1` to `6`
    corrected_eq = re.sub(r'(\d+)\*1\b', r'\1', corrected_eq)

    # Remove invalid characters
    corrected_eq = re.sub(r'[^0-9+\-*/=.' + main_var + ']', '', corrected_eq)

    return corrected_eq

# Example Usage:
# predicted_eq = "yx-5x+6x0"
# fixed_eq = correct_equation(predicted_eq)
# print(f"Fixed Equation: {fixed_eq}")


def format_equation(pred):
    i = 0
    while (i < (len(pred)-1)):
        if pred[i] == "-" and  pred[i+1] == "-":
            pred[i] = "="
            del pred[i+1]
        if pred[i].isalpha() and pred[i+1].isdigit():
            pred.insert(i+1,"^")
        i+=1

    final_string = ''.join(pred)
    print('Processed equation : ', final_string)
    return final_string