from sympy import symbols, Eq, solve, simplify, Rational, sympify, factor, nsolve
import re
from PIL import Image, ImageDraw, ImageFont

# def render_text_as_image(text, filename="output.png"):
#     """Render text as an image and display it."""
#     font = ImageFont.load_default()
#     image = Image.new("RGB", (800, 400), "white")
#     draw = ImageDraw.Draw(image)
    
#     lines = text.split("\n")
#     y_offset = 10
#     for line in lines:
#         draw.text((10, y_offset), line, fill="black", font=font)
#         y_offset += 15
    
#     image.save(filename)
#     # image.show()

def render_text_as_image(text, filename="output.png"):
    """Render text as an image and display it with a fallback font."""
    try:
        font = ImageFont.truetype("comic.ttf", 28)  # Use Comic Sans if available
    except OSError:
        print("Warning: 'comic.ttf' not found, using default font.")
        font = ImageFont.load_default()  # Fallback font

    padding = 20
    lines = text.split("\n")
    formatted_lines = []

    for line in lines:
        if any(char.isdigit() or char in "+-*/=^" for char in line):  # Identifying math parts
            line = line.replace("**", "^")  # Replace '**' with '^' only in math parts
            line = line.replace("*", "")
        formatted_lines.append(line)

    # Determine image size
    image_width = max([font.getbbox(line)[2] for line in formatted_lines]) + 2 * padding
    image_height = len(formatted_lines) * 20 + 2 * padding

    # Create image
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)

    y_offset = padding
    for line in formatted_lines:
        if line.startswith("Step") or line.startswith("Final"):  # Left-align step descriptions
            draw.text((padding, y_offset), line, fill="black", font=font)
        else:  # Center-align math equations
            text_width = font.getbbox(line)[2]
            draw.text(((image_width - text_width) // 2, y_offset), line, fill="black", font=font)
        y_offset += 20

    image.save(filename)



def show_error_message(message):
    """Display an error message as an image."""
    error_text = f"Error: {message}\n\nPossible Causes:\n- Check equation formatting.\n- Ensure all variables & operators are valid.\n- Try simplifying the input."
    render_text_as_image(error_text)

def parse_polynomial(equation_str, var):
    """Parses a polynomial equation from a string and returns a symbolic expression."""
    equation_str = equation_str.replace(" ", "")
    equation_str = re.sub(r"(\d+)/(\d+)([a-zA-Z])", r"(\1/\2)*\3", equation_str)

    if "=" in equation_str:
        lhs, rhs = equation_str.split("=")
        rhs = sympify(rhs)  
        lhs = sympify(lhs) - rhs  
    else:
        lhs = sympify(equation_str)
    
    return lhs

def polynomial_division(dividend, divisor, var):
    """Performs synthetic division (divides dividend by (x - root))."""
    x = symbols(var)
    quotient, remainder = dividend.as_poly(x).div((x - divisor).as_poly(x))
    return quotient.as_expr()

def find_rational_root(polynomial, var):
    """Finds a rational root using the Rational Root Theorem."""
    from sympy import divisors

    coeffs = polynomial.as_coefficients_dict()
    p_factors = divisors(int(coeffs.get(1, 0))) if coeffs.get(1, 0) != 0 else []
    q_factors = divisors(int(next(iter(coeffs.values()))))

    possible_roots = set()
    for p in p_factors:
        for q in q_factors:
            possible_roots.add(Rational(p, q))
            possible_roots.add(Rational(-p, q))

    for root in possible_roots:
        if polynomial.subs(symbols(var), root) == 0:
            return root
    
    return None

def solve_polynomial(equation_str):
    """Solves a polynomial equation and returns output as a string."""
    output_text = ""

    match = re.search(r"[a-zA-Z]", equation_str)
    if match:
        var = match.group()
    else:
        return "Error: No valid variable found in the equation."

    x = symbols(var)
    output_text += f"\nStep 1: Given Polynomial Equation\n  {equation_str}\n"

    polynomial = parse_polynomial(equation_str, var)

    if polynomial.as_poly(x).degree() == 1:
        solution = solve(Eq(polynomial, 0), x)
        output_text += f"\nStep 2: Solving Linear Equation\n"
        output_text += f"\nFinal Step: Solution\n  {var} = {solution[0]}\n"
        return output_text

    factored = factor(polynomial)
    if factored != polynomial:
        output_text += f"\nStep 2: Factorized Form\n  {factored} = 0\n"
        solutions = solve(Eq(polynomial, 0), x)
    else:
        rational_root = find_rational_root(polynomial, var)
        if rational_root is None:
            output_text += "\nStep 2: No Rational Roots Found. Using numerical methods.\n"
            solutions = solve(Eq(polynomial, 0), x)
            if not solutions:
                solutions = [nsolve(polynomial, x, 0)]
        else:
            output_text += f"\nStep 2: Found a Rational Root: {var} = {rational_root}\n"
            reduced_polynomial = polynomial_division(polynomial, rational_root, var)
            if reduced_polynomial == 1:
                solutions = [rational_root]
            else:
                output_text += f"\nStep 3: Reduced Polynomial Equation\n  {simplify(reduced_polynomial)} = 0\n"
                solutions = solve(Eq(reduced_polynomial, 0), x)
                solutions.append(rational_root)

    output_text += "\nFinal Step: Solutions\n"
    for idx, sol in enumerate(solutions, 1):
        if sol.is_real:
            output_text += f"  Solution {idx}: {var} = {sol.evalf():.4f}\n"
        else:
            real_part, imag_part = sol.evalf().as_real_imag()
            output_text += f"  Solution {idx}: {var} = {real_part:.4f} {'+' if imag_part >= 0 else '-'} {abs(imag_part):.4f}i\n"

    return output_text

def evaluate_expression(expr_str):
    """Evaluates a mathematical expression and returns output as a string."""
    try:
        result = sympify(expr_str).evalf()
        return f"Result: {result:.4f}\n"
    except Exception as e:
        return f"Error evaluating expression: {e}"

def compute_equation(input_equation):
    """Computes the equation and renders the result as an image."""
    output_text =""

    if "=" in input_equation:
        output_text += solve_polynomial(input_equation)
    else:
        output_text += evaluate_expression(input_equation)

    render_text_as_image(output_text)

# # Get user input and compute the equation
# inputeq = str(input("Enter an equation or expression: "))
# compute_equation(inputeq)
