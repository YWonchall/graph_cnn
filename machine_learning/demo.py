from machine_learning.utils import extract_feature

smiles = ["CN1CCN(c2ccc(Nc3ncc4nc(Nc5ccc(F)cc5)n([C@H]5CCCN(C(=O)CO)C5)c4n3)cc2)CC1"]
print(extract_feature(smiles_list=smiles))