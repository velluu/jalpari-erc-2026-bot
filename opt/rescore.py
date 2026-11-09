#!/usr/bin/env python3
"""Rescore trial results written by servers that still ran the 19:00 held rule (book above rest + 3 cm, which rows 1
and 2 never meet in the carry posture). v2: book within 15 cm of the gripper and more than 15 cm from its rest pose.
Runs over every queue until killed: python3 rescore.py Q1 Q2 ..."""
import glob, json, math, os, sys, time
while True:
    for q in sys.argv[1:]:
        for f in glob.glob(f'{q}/results/*.json'):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if d.get('held_rule') == 'v2' or 'J' not in d or d.get('found') is False:
                continue
            last = [e['bk'] for e in d.get('events', []) if e.get('bk')]
            out = math.hypot(*last[-1][:2]) if last else 0.0
            held = bool(d.get('book_to_tool', 9) < 0.15 and out > 0.15)
            grasp = 'grasp' in (d.get('stages') or [])
            m = 1.5 if '/s1' in q or 'col4' in q else 1.0
            J = 2 * m * grasp + 2 * m * (grasp and held) - 0.5 * len(d.get('collisions') or []) - 0.5 * len(d.get('palm_events') or [])
            if not grasp:
                J += 0.3 * bool(d.get('bilateral')) + 0.3 * min(1.0, (d.get('max_lift') or 0) / 0.04)
            d.update(held=held, held_rule='v2', J_v1=d['J'], J=round(J, 3))
            json.dump(d, open(f + '.tmp', 'w'), indent=1); os.replace(f + '.tmp', f)
            print('rescored', f, d['J_v1'], '->', d['J'], flush=True)
    time.sleep(60)
