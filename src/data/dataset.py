from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# 최신 방식으로 Morgan Fingerprint 계산
def get_morgan_fp(mol, fp_bits=2048, radius=2):
    generator = GetMorganGenerator(radius=radius, fpSize=fp_bits)
    fp = generator.GetFingerprint(mol)
    return list(fp)

# 전체 특성 추출 함수 내부에 반영
def extract_features(smiles, fp_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * (6 + fp_bits + 10)

        basic_features = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
        ]

        fp_array = get_morgan_fp(mol, fp_bits)

        additional_features = [
            Descriptors.FractionCSP3(mol),
            Lipinski.HeavyAtomCount(mol),
            Lipinski.NumAliphaticRings(mol),
            Lipinski.NumAromaticRings(mol),
            Lipinski.NumSaturatedRings(mol),
            Descriptors.RingCount(mol),
            Descriptors.NHOHCount(mol),
            Descriptors.NOCount(mol),
            Descriptors.PEOE_VSA1(mol),
            Descriptors.EState_VSA1(mol),
        ]

        return basic_features + fp_array + additional_features

    except Exception:
        return [0] * (6 + fp_bits + 10)
