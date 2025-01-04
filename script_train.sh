python train.py \
    -T'./data/BIT/train.csv' \
    -V'./data/BIT/test.csv' \
    -X1 'Solute SMILES' \
    -Y 'LogS' \
    -O './results/' \
    -b 256 \
    -e 100
