import pandas as pd
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors

from src.utils import all_rools_valid, smiles_valid, mol_weight

from typing import List, Dict
import re


PROMPT = """
Create {n} UNIQUE new antioxidant supplement molecules as SMILES strings that meet ALL criteria:
1. Class: Aromatic amines OR phenols 
2. Structure:
   - ONLY atoms: C, H, O, N, P, S
   - Molecular weight ≤ 1000 g/mol
   - Neutral molecules (no charges/radicals)
   - logP > 1 (hexane-soluble)
   
Additional guidance:
- Include multiple electron-donating groups (e.g., -NH2, -OH) 
- Avoid steric hindrance near active sites
- Consider conjugated systems for radical stabilization
- Prefer compact aromatic cores with branched substituents
- Make more benzene rings

Example smiles should be similar to:

{examples}

Output should ONLY SMILES sring/s, without explanation
"""


model_name_or_id = "OpenDFM/ChemDFM-v1.5-8B"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_id)

generation_config = GenerationConfig(
    do_sample=True,
    top_k=20,
    top_p=0.9,
    temperature=0.5,
    max_new_tokens=512,
    repetition_penalty=1.05,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

def get_answer(prompt: str, n_molecules: int, example_smiles: List[str], model: LlamaForCausalLM) -> Dict[str, str]:

    formated_prompt = prompt.replace('{n}', str(n_molecules)).replace('{examples}', '\n\n'.join(example_smiles))
    
    input_text = f"[Round 0]\nHuman: {formated_prompt}\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    inputs.pop('token_type_ids', None)


    outputs = model.generate(**inputs, generation_config=generation_config)
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(input_text):]
    generated_text = generated_text.replace(' ', '')
    pattern = re.compile(r'^\d+\.\s*(.*?)\s*$', re.MULTILINE)
    smiles_list = pattern.findall(generated_text)
    
    if len(smiles_list) == 0:
        smiles_list = generated_text.split('\n')
    
    return smiles_list


