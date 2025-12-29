
encoding = 'utf-8'
try:
    with open("main.py", "r", encoding=encoding) as f:
        lines = f.readlines()
except UnicodeDecodeError:
    encoding = 'cp1252'
    with open("main.py", "r", encoding=encoding) as f:
        lines = f.readlines()

print(f"Total lines: {len(lines)}")
print(f"Line 235 (Index 234): {lines[234].strip()}")
print(f"Line 2233 (Index 2232): {lines[2232].strip()}")
print(f"Line 2234 (Index 2233): {lines[2233].strip()}")

# Check for other templates in the range
content = "".join(lines[234:2233])
print(f"Contains ORION_TEMPLATE: {'ORION_TEMPLATE' in content}")
print(f"Contains PULSAR_TEMPLATE: {'PULSAR_TEMPLATE' in content}")
print(f"Contains QUASAR_TEMPLATE: {'QUASAR_TEMPLATE' in content}")
print(f"Contains SELECTOR_TEMPLATE: {'SELECTOR_TEMPLATE' in content}")
