'''
Author: QHGG
Date: 2021-03-01 22:45:52
LastEditTime: 2021-03-22 14:14:46
LastEditors: QHGG
Description: DrugVQA
FilePath: /drugVQA/app.py
'''
import torch
from sklearn import metrics
from utils import *
from map import *
from model import *
import warnings
warnings.filterwarnings("ignore")
from loguru import logger

from flask import Flask, jsonify, request
import json
app = Flask(__name__)

modelArgs = {}
modelArgs['batch_size'] = 1
modelArgs['lstm_hid_dim'] = 64
modelArgs['d_a'] = 32
modelArgs['r'] = 10
modelArgs['n_chars_smi'] = 247
modelArgs['n_chars_seq'] = 21
modelArgs['dropout'] = 0.2
modelArgs['in_channels'] = 8
modelArgs['cnn_channels'] = 32
modelArgs['cnn_layers'] = 4
modelArgs['emb_dim'] = 30
modelArgs['dense_hid'] = 64
modelArgs['task_type'] = 0
modelArgs['n_classes'] = 1


smiles_letter = ['[102Ru]', '[80Se]', '[N-]', '[33SH2]', '[Ag+]', 'L', '[Fr]', '[C@@H-]', '[SiH2]', '[Sr+2]', '[P@@H]', '[CH2]', '[10B]', '[SH]', '[Gd]', '[Ru+2]', '[81R-]', '[Ar]', '[85Sr+2]', '[29Si]', '[Cu+2]', '[CH3-]', '[56Ni]', '[Na+]', '[Au+]', '[Co+3]', '[Se-]', '[NH-]', '[113In]', ')', '[Ir]', '[NH+]', '[109Cd]', '[R-]', '[43K]', '[15NH2]', '[Zn+2]', '[P@H]', '[Ce]', '[As+5]', '[Pt+4]', 'n', '6', '[Ca+2]', '[121Sn]', '4', '[nH+]', 'R', '[F]', '[Pd]', '3', '[R]', '\\', '#', '[H-]', '[16OH2]', '[O-2]', '[nH]', '[Ni+2]', '[CH]', 'I', '[Ho]', '[C@H-]', '[209Pb]', 'p', 'F', '[18F]', '[183W]', '[34SH2]', '[Co+2]', '[CH2-]', '[Sb]', '[Be+2]', '[n+]', '[13C]', '[Fe+2]', '[B]', '[178W]', '[67Ga+3]', '[3H]', '[63Cu]', '[79R-]', '[n-]', 's', '[C+]', '[205Pb]', '[H+]', '[O-]', '(', '[H]', '[OH2+]', '[Mn]', '[Se]', '[78Se]', '[s+]', '[NH2-]', '[Fe]', '[C@@]', '[OH-]', '[S@]', '5', '[173Ta]', '[11CH4]', '[L-]', '[128IH]', '[Mo]', '[As+]', '[B-]', '[129IH]', '[Ta+5]', '[68Ge]', 'C', '[77RH]', '/', '[Hg+]', '[NH4+]', '[P@@+]', '[CH-]', '[84RH]', '[43Ca]', '[S-2]', '[11CH3]', '[Fe+3]', '[Ca]', '[125Sb]', '[Sb-]', '[P@]', '[210Tl]', '[PH+]', '[K]', '[SH+]', 'o', '[Si+4]', '[89Sr+2]', '[I]', '[15NH3]', 'O', '[14CH2]', '[33P]', 'c', '=', '[C-]', '[127IH]', '[141Ce]', '[75Se]', '8', '[11B]', '[P@@]', 'B', '.', 'N', '[P@+]', '[Hg]', '[252Cf]', '[Pb+2]', '[O+]', '[56Fe]', '[13CH]', '[45Ca+2]', '[Zr]', '[45K]', '[C@H]', '[In+3]', '[S@@]', '[Si]', '[99Tc]', '[Hg+2]', '[o+]', '[Ba+2]', '[N@+]', '1', '9', '[AlH4-]', '[44Ca]', '[N+]', '[Cu]', '[9Li]', '[Tl+]', '[Co]', '[Re]', '[OH3+]', '[Mn+2]', '[Ru+]', '[L]', '[Li+]', '[V]', '<pad>', '7', '[115Cd]', '[Mg+2]', '[P+]', '[Pd+2]', '[I-]', '[Cr+3]', '[Zn]', '[C@]', '[110Ag]', '[214Pb]', '2', '[60Fe]', 'S', '[NH2+]', '[P@@H+]', '[W]', '[Al+3]', '[F-]', '[OH]', '[125IH]', '[N@@+]', '[C]', '[13NH3]', '[Bi]', '[218Po]', '[S-]', '[As]', '[SiH3]', '[23Na]', '[Ge]', '[197Au]', '[32P]', '[12CH4]', '[C@@H]', '[17NH3]', '[NH3+]', '[132IH]', '[99Mo]', '[207Pb]', '[7Li]', '[2H]', '[Al]', '[OH+]', '[Zr+4]', '[S+]', '[125I]', '-', '[Ag]', '[Sb+3]', '[3He]', '[90Sr+2]', '[K+]', '[Y+3]', '[Ti]', '[188W]', '[Rb+]', '[Pt+2]', '[14C]', 'P']



# active = 0
# for batch_idx,(lines, contactmap,properties) in enumerate(train_loader):
#     lines = ('CNS(=O)(=O)c1ccc2c(c1)CCN2S(=O)(=O)c3cc(ccc3C(=O)OC)C(=O)OC',)
#     print(smiles_letters)
#     input, seq_lengths, y = make_variables(lines, properties,smiles_letters)
#     print(contactmap.shape, seq_lengths, lines)

#     attention_model.hidden_state = attention_model.init_hidden()
#     contactmap = create_variable(contactmap)
#     y_pred,att = attention_model(input,contactmap)
#     # print(batch_idx, y_pred.item(), lines)
#     if y_pred.item() > 0.1:
#         active += 1
#     if batch_idx % 100 == 0:
#         # print(active)
#         active = 0
attention_model = DrugVQA(modelArgs,block = ResidualBlock).cuda()
attention_model.load_state_dict(torch.load('./2021-03-03 02:30:38DUDE30Res-fold3-30.pkl', map_location='cpu'))
attention_model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # data = json.loads(request.data)
        lines = (request.form['smi'], )
        logger.warning('smiles:{}'.format(request.form['smi']))
        input, seq_lengths, y = make_variables(lines, torch.tensor([0]),smiles_letter)
        # print(contactmap.shape, seq_lengths, lines)
        attention_model.hidden_state = attention_model.init_hidden()
        contactmap = create_variable(feature2D)
        y_pred, _ = attention_model(input,contactmap)
        # print(lines, seq, y_pred.item())

        
        return jsonify({'lines': lines, 'seq': seq, 'pred': y_pred.item()})


    
if __name__ == "__main__":
    app.run()


# ('c1c(nc([nH]c1=O)SCC(=O)N2CCC[C@H](C2)C(=O)N)N',) tensor([0]) ['[102Ru]', '[80Se]', '[N-]', '[33SH2]', '[Ag+]', 'L', '[Fr]', '[C@@H-]', '[SiH2]', '[Sr+2]', '[P@@H]', '[CH2]', '[10B]', '[SH]', '[Gd]', '[Ru+2]', '[81R-]', '[Ar]', '[85Sr+2]', '[29Si]', '[Cu+2]', '[CH3-]', '[56Ni]', '[Na+]', '[Au+]', '[Co+3]', '[Se-]', '[NH-]', '[113In]', ')', '[Ir]', '[NH+]', '[109Cd]', '[R-]', '[43K]', '[15NH2]', '[Zn+2]', '[P@H]', '[Ce]', '[As+5]', '[Pt+4]', 'n', '6', '[Ca+2]', '[121Sn]', '4', '[nH+]', 'R', '[F]', '[Pd]', '3', '[R]', '\\', '#', '[H-]', '[16OH2]', '[O-2]', '[nH]', '[Ni+2]', '[CH]', 'I', '[Ho]', '[C@H-]', '[209Pb]', 'p', 'F', '[18F]', '[183W]', '[34SH2]', '[Co+2]', '[CH2-]', '[Sb]', '[Be+2]', '[n+]', '[13C]', '[Fe+2]', '[B]', '[178W]', '[67Ga+3]', '[3H]', '[63Cu]', '[79R-]', '[n-]', 's', '[C+]', '[205Pb]', '[H+]', '[O-]', '(', '[H]', '[OH2+]', '[Mn]', '[Se]', '[78Se]', '[s+]', '[NH2-]', '[Fe]', '[C@@]', '[OH-]', '[S@]', '5', '[173Ta]', '[11CH4]', '[L-]', '[128IH]', '[Mo]', '[As+]', '[B-]', '[129IH]', '[Ta+5]', '[68Ge]', 'C', '[77RH]', '/', '[Hg+]', '[NH4+]', '[P@@+]', '[CH-]', '[84RH]', '[43Ca]', '[S-2]', '[11CH3]', '[Fe+3]', '[Ca]', '[125Sb]', '[Sb-]', '[P@]', '[210Tl]', '[PH+]', '[K]', '[SH+]', 'o', '[Si+4]', '[89Sr+2]', '[I]', '[15NH3]', 'O', '[14CH2]', '[33P]', 'c', '=', '[C-]', '[127IH]', '[141Ce]', '[75Se]', '8', '[11B]', '[P@@]', 'B', '.', 'N', '[P@+]', '[Hg]', '[252Cf]', '[Pb+2]', '[O+]', '[56Fe]', '[13CH]', '[45Ca+2]', '[Zr]', '[45K]', '[C@H]', '[In+3]', '[S@@]', '[Si]', '[99Tc]', '[Hg+2]', '[o+]', '[Ba+2]', '[N@+]', '1', '9', '[AlH4-]', '[44Ca]', '[N+]', '[Cu]', '[9Li]', '[Tl+]', '[Co]', '[Re]', '[OH3+]', '[Mn+2]', '[Ru+]', '[L]', '[Li+]', '[V]', '<pad>', '7', '[115Cd]', '[Mg+2]', '[P+]', '[Pd+2]', '[I-]', '[Cr+3]', '[Zn]', '[C@]', '[110Ag]', '[214Pb]', '2', '[60Fe]', 'S', '[NH2+]', '[P@@H+]', '[W]', '[Al+3]', '[F-]', '[OH]', '[125IH]', '[N@@+]', '[C]', '[13NH3]', '[Bi]', '[218Po]', '[S-]', '[As]', '[SiH3]', '[23Na]', '[Ge]', '[197Au]', '[32P]', '[12CH4]', '[C@@H]', '[17NH3]', '[NH3+]', '[132IH]', '[99Mo]', '[207Pb]', '[7Li]', '[2H]', '[Al]', '[OH+]', '[Zr+4]', '[S+]', '[125I]', '-', '[Ag]', '[Sb+3]', '[3He]', '[90Sr+2]', '[K+]', '[Y+3]', '[Ti]', '[188W]', '[Rb+]', '[Pt+2]', '[14C]', 'P']