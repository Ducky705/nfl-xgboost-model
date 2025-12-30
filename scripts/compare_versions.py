import pandas as pd

# Load both files
v3_df = pd.read_csv('data/betting_history_backup_v3.csv')
v4_df = pd.read_csv('data/betting_history.csv')

print("=" * 50)
print("V3 RESULTS (Backup)")
print("=" * 50)
print(f"Total Bets: {len(v3_df)}")
print(f"Total Profit: {v3_df['profit'].sum():.2f}u")

v3_graded = v3_df[v3_df['status'] == 'GRADED']
v3_wins = len(v3_graded[v3_graded['result'] == 'WIN'])
v3_losses = len(v3_graded[v3_graded['result'] == 'LOSS'])
v3_pushes = len(v3_graded[v3_graded['result'] == 'PUSH'])
print(f"Record: {v3_wins}-{v3_losses}-{v3_pushes}")

# By Type
for t in v3_df['type'].unique():
    sub = v3_df[v3_df['type'] == t]
    print(f"  [{t}] Bets: {len(sub)}, Profit: {sub['profit'].sum():.2f}u")

print()
print("=" * 50)
print("V4 RESULTS (Current)")
print("=" * 50)
print(f"Total Bets: {len(v4_df)}")
print(f"Total Profit: {v4_df['profit'].sum():.2f}u")

v4_graded = v4_df[v4_df['status'] == 'GRADED']
v4_wins = len(v4_graded[v4_graded['result'] == 'WIN'])
v4_losses = len(v4_graded[v4_graded['result'] == 'LOSS'])
v4_pushes = len(v4_graded[v4_graded['result'] == 'PUSH'])
print(f"Record: {v4_wins}-{v4_losses}-{v4_pushes}")

# By Type
for t in v4_df['type'].unique():
    sub = v4_df[v4_df['type'] == t]
    print(f"  [{t}] Bets: {len(sub)}, Profit: {sub['profit'].sum():.2f}u")

print()
print("=" * 50)
print("COMPARISON")
print("=" * 50)
print(f"V3 Profit: {v3_df['profit'].sum():.2f}u")
print(f"V4 Profit: {v4_df['profit'].sum():.2f}u")
print(f"Difference: {v4_df['profit'].sum() - v3_df['profit'].sum():.2f}u")
