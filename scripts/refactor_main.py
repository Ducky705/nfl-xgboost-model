
import os

MAIN_PATH = "d:/Programs/Sports Betting/NFL Model/v0.0.2/main.py"

encoding = 'utf-8'
try:
    with open(MAIN_PATH, "r", encoding=encoding) as f:
        lines = f.readlines()
except UnicodeDecodeError:
    encoding = 'cp1252'
    with open(MAIN_PATH, "r", encoding=encoding) as f:
        lines = f.readlines()

new_lines = []

# 1. Process Lines (Deletions & Import Updates)
skip_mode = False
for i, line in enumerate(lines):
    # 0-indexed line numbers
    # Delete 235 (index 234) to 2233 (index 2232)
    # Actually, verify start/end markers dynamically if possible, but hardcoded is safer given verification
    if i == 234: # Start of deletions
        skip_mode = True
    
    if i == 2233: # Resume (Line 2234)
        skip_mode = False
    
    if not skip_mode:
        # Import fix
        if "import v2_features" in line:
            line = line.replace("import v2_features", "import src.features as v2_features")
        new_lines.append(line)

content = "".join(new_lines)

# 2. Content Replacements (Usage Updates)
# Update render_page definition
content = content.replace("def render_page(filename, title, model_name, bets_list, hist_df, template_name):", 
                          "def render_page(filename, title, model_name, bets_list, hist_df, template_filename):")

# Update render_page body
content = content.replace("template = env.from_string(template_name)", "template = env.get_template(template_filename)")

# Update calls
content = content.replace("ORION_TEMPLATE", '"orion.html"')
content = content.replace("PULSAR_TEMPLATE", '"pulsar.html"')
content = content.replace("QUASAR_TEMPLATE", '"quasar.html"')

# Update Selector Usage
content = content.replace("selector_template = env.from_string(SELECTOR_TEMPLATE)", 'selector_template = env.get_template("selector.html")')

# Save
with open(MAIN_PATH, "w", encoding='utf-8') as f:
    f.write(content)

print("Successfully refactored main.py")
