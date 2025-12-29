import pandas as pd
df = pd.read_csv('data/betting_history.csv')
g = df[df['status']=='GRADED']
print('V3.2 RESULTS:')
print(f"Total: {len(g)} bets")
wins = (g['result']=='WIN').sum()
losses = (g['result']=='LOSS').sum()
print(f'Record: {wins}W-{losses}L')
print(f'Win Rate: {wins/(wins+losses)*100:.1f}%')
print(f'Total Profit: {g["profit"].sum():.2f}u')
print()
for t in ['spread','total','moneyline']:
    gt = g[g['type']==t]
    w = (gt['result']=='WIN').sum()
    l = (gt['result']=='LOSS').sum()
    print(f'{t.upper()}: {w}W-{l}L, {gt["profit"].sum():.2f}u')
