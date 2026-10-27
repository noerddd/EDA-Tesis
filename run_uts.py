"""Reproduce UTS analysis from the RAW ROV CSV.
Usage: python run_uts.py --input 'combined_log_sb(water-quality).csv' --out output --spline-df 8
Deps: numpy pandas scipy statsmodels patsy matplotlib.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,statsmodels.api as sm
from patsy import dmatrix
from scipy.stats import spearmanr
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=argparse.ArgumentParser();P.add_argument('--input',required=True);P.add_argument('--out',default='output');P.add_argument('--spline-df',type=int,default=8);A=P.parse_args()
O=Path(A.out);O.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(A.input); raw_n=len(d)
x=d.ts.str.extract(r'^(\d+):(\d+(?:\.\d+)?)$').astype(float);d['sec']=60*x[0]+x[1];d['gap']=d.sec.diff();d['session']=((d.gap>1)|(d.gap<-.000001)).cumsum()
sess=d.groupby('session',sort=True).agg(n=('ts','size'),duration=('sec',lambda v:v.iloc[-1]-v.iloc[0]),missing=('ph',lambda v:int(v.isna().sum())),low_ph=('ph',lambda v:int((v<5).sum())),zero_tds=('tds_mg_per_l',lambda v:int((v==0).sum())))
sess.to_csv(O/'sessions.csv'); ids=sess.index[sess.duration>180].tolist()
metrics=['depth_m','groundspeed_m_s','att_roll_deg','att_pitch_deg','att_yaw_deg','do_mg_per_l','ph','tds_mg_per_l']
t=d.groupby(['session','sec'],sort=False,as_index=False)[metrics].median()
t['elapsed']=t.sec-t.groupby('session').sec.transform('min')
t=t[t.session.isin(ids)&t.ph.between(6.5,8.5)&(t.tds_mg_per_l>0)&t.do_mg_per_l.notna()].copy()
t['bin']=np.floor(t.elapsed).astype(int);t['sin']=np.sin(np.deg2rad(t.att_yaw_deg));t['cos']=np.cos(np.deg2rad(t.att_yaw_deg))
s=t.groupby(['session','bin'],as_index=False)[metrics[:-3]+metrics[-3:]+['sin','cos']].mean() # 1Hz per session
s['elapsed']=s['bin'].astype(float);s.to_csv(O/'analysis_1hz.csv',index=False)
stat=s[['do_mg_per_l','ph','tds_mg_per_l']].agg(['count','mean','std','min','median','max']).T
stat.to_csv(O/'before_after_summary.csv')
# no full-raw imputation: compare raw valid and per-second values; show decision effects
pd.DataFrame({'variable':['rows','missing_any_sensor','ph_under_5','zero_tds','duplicate_rows','retained_1hz'], 'raw':[raw_n,int(d[['do_mg_per_l','ph','tds_mg_per_l']].isna().any(axis=1).sum()),int((d.ph<5).sum()),int((d.tds_mg_per_l==0).sum()),int(d.drop(columns=['sec','gap','session']).duplicated().sum()),np.nan], 'processed':[len(s),0,0,0,0,len(s)]}).to_csv(O/'processing_comparison.csv',index=False)
rows=[]; fitdetails=[]
def matrices(df,flex):
 pieces=[]
 for sid in sorted(df.session.unique()):
  flag=df.session.eq(sid).to_numpy();z=df.loc[flag,'elapsed'].to_numpy();z=(z-z.min())/(z.max()-z.min()); b=np.asarray(dmatrix(f'bs(z, df={flex}, degree=3, include_intercept=False)-1',{'z':z}))
  for j in range(b.shape[1]):
   arr=np.zeros(len(df));arr[flag]=b[:,j];pieces.append(pd.Series(arr,index=df.index,name=f't_{sid}_{j}'))
 base=sm.add_constant(pd.concat([pd.get_dummies(df.session.astype(str),drop_first=True,dtype=float),*pieces,df[['depth_m','groundspeed_m_s']]],axis=1)).astype(float)
 ori=df[['att_roll_deg','att_pitch_deg','sin','cos']].copy();ori[['att_roll_deg','att_pitch_deg']]/=10
 return base,pd.concat([base,ori],axis=1),list(ori.columns)
for flex in [4,6,8,10,12]:
 base,full,oc=matrices(s,flex)
 for y in ['do_mg_per_l','ph','tds_mg_per_l']:
  b=sm.OLS(s[y],base).fit(cov_type='HAC',cov_kwds={'maxlags':10});m=sm.OLS(s[y],full).fit(cov_type='HAC',cov_kwds={'maxlags':10});R=np.zeros((4,len(full.columns)))
  for j,k in enumerate(oc):R[j,list(full.columns).index(k)]=1
  p=float(m.wald_test(R,scalar=True).pvalue)
  rows.append({'spline_df':flex,'outcome':y,'n':len(s),'r2_base':b.rsquared,'r2_full':m.rsquared,'delta_r2':m.rsquared-b.rsquared,'joint_hac_p':p,'pitch_coef_per_10deg':m.params['att_pitch_deg'],'pitch_ci_lo':m.conf_int().loc['att_pitch_deg',0],'pitch_ci_hi':m.conf_int().loc['att_pitch_deg',1],'pitch_p':m.pvalues['att_pitch_deg']})
  if flex==A.spline_df:
   for k in oc: fitdetails.append({'outcome':y,'term':k,'coef':m.params[k],'ci_low':m.conf_int().loc[k,0],'ci_high':m.conf_int().loc[k,1],'p_hac':m.pvalues[k]})
r=pd.DataFrame(rows);r.to_csv(O/'sensitivity.csv',index=False);pd.DataFrame(fitdetails).to_csv(O/'coefficients.csv',index=False)
q=[]
for v in ['do_mg_per_l','ph','tds_mg_per_l','depth_m','groundspeed_m_s','att_roll_deg','att_pitch_deg']:
 a,b=s[v].quantile([.25,.75]);bad=((s[v]<a-1.5*(b-a))|(s[v]>b+1.5*(b-a))).sum();q.append({'variable':v,'iqr_outliers':int(bad),'pct':100*bad/len(s)})
pd.DataFrame(q).to_csv(O/'iqr.csv',index=False)
fig,axs=plt.subplots(1,3,figsize=(12,3.8))
for ax,y in zip(axs,['do_mg_per_l','ph','tds_mg_per_l']):
 g=r[r.outcome==y];ax.plot(g.spline_df,g.joint_hac_p,marker='o');ax.axhline(.05,color='gray',linestyle='--');ax.set_title(y);ax.set_xlabel('Spline degrees of freedom');ax.set_ylabel('Joint HAC p-value');ax.set_ylim(-.02,1)
fig.tight_layout();fig.savefig(O/'sensitivity.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(12,3.8))
for ax,y in zip(axs,['do_mg_per_l','ph','tds_mg_per_l']):
 for sid,g in s.groupby('session'):ax.plot(g.elapsed,g[y],label=f'Sesi {sid}',alpha=.85)
 ax.set_title(y);ax.set_xlabel('Detik sejak awal sesi');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(O/'sensor_trends.png',dpi=160);plt.close(fig)
summary={'raw_rows':raw_n,'sessions':len(sess),'retained_sessions':ids,'one_hz_rows':len(s),'missing_any_sensor':int(d[['do_mg_per_l','ph','tds_mg_per_l']].isna().any(axis=1).sum()),'ph_under_5':int((d.ph<5).sum()),'zero_tds':int((d.tds_mg_per_l==0).sum()),'main':r[r.spline_df==A.spline_df].to_dict('records')}
(O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
