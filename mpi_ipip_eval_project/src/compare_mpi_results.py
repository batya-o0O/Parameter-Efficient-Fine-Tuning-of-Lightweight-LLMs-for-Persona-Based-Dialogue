import argparse,glob,json,os,re,pandas as pd
TRAITS=['O','C','E','A','N']
def read(p):
 df=pd.read_csv(p); return {r.trait:float(r['mean']) for _,r in df.iterrows()}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--results_dir',required=True); ap.add_argument('--target_profile',required=True); ap.add_argument('--output_dir',required=True); args=ap.parse_args(); os.makedirs(args.output_dir,exist_ok=True); target=json.load(open(args.target_profile)); rows=[]
 for p in glob.glob(os.path.join(args.results_dir,'*_target_profile_summary.csv')):
  model=os.path.basename(p).replace('_target_profile_summary.csv',''); s=read(p); row={'model':model}; errs=[]
  for t in TRAITS: row[f'{t}_pred']=s[t]; row[f'{t}_target']=target[t]; row[f'{t}_abs_error']=abs(s[t]-target[t]); errs.append(row[f'{t}_abs_error'])
  row['MAE']=sum(errs)/5; row['alignment']=1-row['MAE']/4; rows.append(row)
 pd.DataFrame(rows).to_csv(os.path.join(args.output_dir,'target_alignment.csv'),index=False)
 rows=[]
 for model in ['base','lora']:
  row={'model':model}; corr=[]
  for t in TRAITS:
   hp=os.path.join(args.results_dir,f'{model}_high_{t}_summary.csv'); lp=os.path.join(args.results_dir,f'{model}_low_{t}_summary.csv')
   if os.path.exists(hp) and os.path.exists(lp):
    h=read(hp)[t]; l=read(lp)[t]; row[f'{t}_delta']=h-l; row[f'{t}_correct']=int(h-l>0); corr.append(int(h-l>0))
  row['directional_accuracy']=sum(corr)/len(corr) if corr else None; rows.append(row)
 pd.DataFrame(rows).to_csv(os.path.join(args.output_dir,'trait_controllability.csv'),index=False)
if __name__=='__main__': main()
