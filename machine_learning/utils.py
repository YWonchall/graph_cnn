import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def smiles_to_fingerprint(smiles, radius=2, n_bits=128):
    """将 SMILES 转换为 Morgan Fingerprint 特征向量"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # 无法解析 SMILES
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fingerprint)

def smiles_to_descriptors(smiles):
    """提取分子描述符"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptors = {
        'MolWt': Descriptors.MolWt(mol),         # 分子量
        'LogP': Descriptors.MolLogP(mol),       # LogP
        'NumHDonors': Descriptors.NumHDonors(mol), # 氢键供体数量
        'NumHAcceptors': Descriptors.NumHAcceptors(mol) # 氢键受体数量
    }
    return descriptors

def extract_feature(smiles_list, method="fingerprint", **kwargs):
    """将 SMILES 转换为特征矩阵"""
    if method == "fingerprint":
        features = [smiles_to_fingerprint(smiles, **kwargs) for smiles in smiles_list]
    elif method == "descriptors":
        features = [smiles_to_descriptors(smiles) for smiles in smiles_list]
    else:
        raise ValueError("Unsupported method.")
    
    return features