"""Round 2, column 2: find each white book by its spine word (A FUTURE, B ROBOT, C LIBRARY, D AGENT) and compare the
estimate with the layout through the robot's gz ground-truth pose."""
import json, subprocess, re, sys, math, rclpy, numpy as np
from scipy.spatial.transform import Rotation
from erc_bot.live import Live, Abort
from erc_bot.mission import Bot, WORDS
lay = json.load(open('/opt/results/dev2/layout.json'))
rclpy.init(); r = Live('t_words', sim_time=True); r.ready()
bot = Bot(r, '/opt/model/det_v3_736x1280.onnx', 2, 'A')
def W_O():
    out = subprocess.run(['gz', 'model', '-m', 'tiago_pro', '-p'], capture_output=True, text=True).stdout
    nums = [list(map(float, m.split())) for m in re.findall(r'\[([-\d\. e]+)\]', out)]
    W = np.eye(4); W[:3, :3] = Rotation.from_euler('xyz', nums[-1]).as_matrix(); W[:3, 3] = nums[-2]
    return W @ np.linalg.inv(r.odom_T_base())
bot.start(); plate, n, t = bot.find_column()
col = lay['markers'].index(2) + 1
print('digit 2 is physical column', col, 'white books there:', [b['name'] for b in lay['books'] if b['column'] == col])
for letter in 'ABCD':
    bot.target = WORDS[letter]
    try:
        book, row, plate, n, t = bot.find_book(plate, n, t)
    except Abort as e:
        print(letter, WORDS[letter], 'NOT FOUND', e); continue
    truth = [b for b in lay['books'] if b['name'].endswith('_white_' + letter)][0]
    est = (W_O() @ np.r_[book, 1])[:3]
    print(letter, WORDS[letter], 'row', row, 'truth', truth['name'], 'est-truth', np.round(est - np.array([truth['x'], truth['y'], truth['z']]), 3), flush=True)
