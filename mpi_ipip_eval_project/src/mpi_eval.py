import argparse,json,os,re,random,pandas as pd
POS={"A":5,"B":4,"C":3,"D":2,"E":1}; NEG={"A":1,"B":2,"C":3,"D":4,"E":5}; NAMES={"O":"Openness","C":"Conscientiousness","E":"Extraversion","A":"Agreeableness","N":"Neuroticism"}
def load_items(p): return [json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]
def extract_choice(t):
 m=re.search(r"\b([A-E])\b",t.strip().upper()) or re.search(r"\(([A-E])\)",t.strip().upper()); return m.group(1) if m else None
def score(c,k): return (POS if k=='+' else NEG).get(c)
def build_messages(s,persona): return [{"role":"system","content":persona},{"role":"user","content":f'Given the following statement about you:\n\n"{s}"\n\nChoose how accurately this statement describes you.\n\nOptions:\nA. Very Accurate\nB. Moderately Accurate\nC. Neither Accurate Nor Inaccurate\nD. Moderately Inaccurate\nE. Very Inaccurate\n\nAnswer with only one letter: A, B, C, D, or E.'}]
def load_hf(base,adapter=None):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(base,trust_remote_code=True); model=AutoModelForCausalLM.from_pretrained(base,device_map='auto',trust_remote_code=True,torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
 if adapter:
  from peft import PeftModel
  model=PeftModel.from_pretrained(model,adapter)
 model.eval(); return model,tok
def gen(model,tok,msgs):
 import torch
 prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); inp=tok(prompt,return_tensors='pt').to(model.device)
 with torch.no_grad(): out=model.generate(**inp,max_new_tokens=4,do_sample=False,pad_token_id=tok.eos_token_id)
 return tok.decode(out[0][inp['input_ids'].shape[1]:],skip_special_tokens=True).strip()
def demo_choice(it,model_tag,prompt):
 prof={'base':{'O':4,'C':4,'E':3,'A':4,'N':3},'lora':{'O':5,'C':4,'E':2,'A':5,'N':2}}; t=it['trait']; val=prof['lora' if 'lora' in model_tag else 'base'][t]
 if prompt.startswith('high_'): val=5 if prompt[-1]==t else 3
 if prompt.startswith('low_'): val=2 if prompt[-1]==t else 3
 return ({5:'A',4:'B',3:'C',2:'D',1:'E'} if it['key']=='+' else {1:'A',2:'B',3:'C',4:'D',5:'E'})[val]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--items_path',required=True); ap.add_argument('--output_prefix',required=True); ap.add_argument('--base_model'); ap.add_argument('--adapter_path'); ap.add_argument('--backend',choices=['hf','demo'],default='hf'); ap.add_argument('--model_tag',default='model'); ap.add_argument('--prompt_name',default='target_profile'); ap.add_argument('--persona_prompt',default='You are answering a personality questionnaire.'); args=ap.parse_args()
 items=load_items(args.items_path); model=tok=None
 if args.backend=='hf': model,tok=load_hf(args.base_model,args.adapter_path)
 rows=[]
 for it in items:
  raw=demo_choice(it,args.model_tag,args.prompt_name) if args.backend=='demo' else gen(model,tok,build_messages(it['statement'],args.persona_prompt)); c=extract_choice(raw)
  rows.append({'id':it['id'],'trait':it['trait'],'trait_name':NAMES[it['trait']],'key':it['key'],'statement':it['statement'],'raw_output':raw,'choice':c,'score':score(c,it['key']),'model_tag':args.model_tag,'prompt_name':args.prompt_name})
 df=pd.DataFrame(rows); os.makedirs(os.path.dirname(args.output_prefix) or '.',exist_ok=True); df.to_csv(args.output_prefix+'_raw.csv',index=False); df.groupby(['trait','trait_name'])['score'].agg(['mean','std','count']).reset_index().to_csv(args.output_prefix+'_summary.csv',index=False)
if __name__=='__main__': main()
