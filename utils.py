from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import Descriptors


def smiles_valid(smiles_str: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles_str)
    except Exception:
        return False
    
    if mol is None:
        return False

    return True

aromatic_amine_smarts = Chem.MolFromSmarts('[N,n;H1,H0;$(N-[a])]-[a]') # этот шаблон ищет ароматические амины, в которых азот связан с ароматическим кольцом
phenol_smarts = Chem.MolFromSmarts('[OH]-[c,C]1:[c,C]:[c,C]:[c,C]:[c,C]:[c,C]:1') # этот шаблон ищет фенолы, в которых гидроксильная группа связана с ароматическим кольцом


def is_aromatic_amine_or_phenol(smiles_str: str) -> bool:
    mol = Chem.MolFromSmiles(smiles_str)
    return mol.HasSubstructMatch(phenol_smarts) or \
           mol.HasSubstructMatch(aromatic_amine_smarts)


def has_radical(mol: Mol) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return True
    return False


def neutral_and_no_radical(smiles_str: str) -> bool:
    mol = Chem.MolFromSmiles(smiles_str)
    return Chem.GetFormalCharge(mol) == 0 and not has_radical(mol)

def mol_weight(smiles_str: str) -> float:
    mol = Chem.MolFromSmiles(smiles_str)
    return Descriptors.MolWt(mol)

allowed_atoms = {'C', 'H', 'O', 'N', 'P', 'S'}

def only_allowed_atoms(smiles_str: str) -> bool:
    mol = Chem.MolFromSmiles(smiles_str)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed_atoms:
            return False
    return True

def log_p(smiles_str: str) -> float:
    mol = Chem.MolFromSmiles(smiles_str)
    logp = Descriptors.MolLogP(mol)
    return logp


def all_rools_valid(smiles_str) -> bool:
    return smiles_valid(smiles_str) and \
        neutral_and_no_radical(smiles_str) and \
        mol_weight(smiles_str) <= 1000 and \
        only_allowed_atoms(smiles_str) and \
        log_p(smiles_str) > 1 and \
        '@' not in smiles_str and \
        '.' not in smiles_str
    


def sa_score(smiles_str) -> float:
    mol = Chem.MolFromSmiles(smiles_str)
    return sascorer.calculateScore(mol)