"""Quick performance analysis script"""
import pandas as pd

df = pd.read_csv('data/betting_history.csv')
graded = df[df['status']=='GRADED']

print('=== PERFORMANCE BY BET TYPE ===\n')

for t in ['spread', 'total', 'moneyline']:
    sub = graded[graded['type']==t]
    units = sub['units'].sum()
    profit = sub['profit'].sum()
    roi = (profit/units*100) if units > 0 else 0
    wins = len(sub[sub['result']=='WIN'])
    losses = len(sub[sub['result']=='LOSS'])
    pushes = len(sub[sub['result']=='PUSH'])
    
    print(f'{t.upper():12} {wins}W-{losses}L-{pushes}P | Units: {units:6.1f} | Profit: {profit:+7.2f}u | ROI: {roi:+.1f}%')

print('\n=== WEEKLY BREAKDOWN ===\n')
for t in ['spread', 'total', 'moneyline']:
    sub = graded[graded['type']==t]
    weekly = sub.groupby('week').agg({
        'profit': 'sum',
        'result': lambda x: (x=='WIN').sum()
    })
    wins_by_week = sub.groupby('week')['result'].apply(lambda x: (x=='WIN').sum())
    total_by_week = sub.groupby('week').size()
    
    print(f'\n{t.upper()} by Week:')
    for w in range(1, 18):
        if w in weekly.index:
            p = weekly.loc[w, 'profit']
            print(f'  W{w:02d}: {p:+6.2f}u')
