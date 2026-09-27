from pathlib import Path
import json
import cv2
import fitz

OUT = Path('reports/materials')
OUT.mkdir(parents=True, exist_ok=True)
pdf = fitz.open(r'C:/Users/Saaik/Downloads/6ab5628d5a817_amazon_ml_challenge_problem_statement (1).pdf')
(OUT / 'problem_statement.txt').write_text('\n'.join(p.get_text() for p in pdf), encoding='utf-8')
for i, p in enumerate(pdf):
    p.get_pixmap(matrix=fitz.Matrix(1, 1)).save(str(OUT / f'pdf_{i+1}.png'))
cap = cv2.VideoCapture(r'C:/Users/Saaik/Downloads/6ab509c5b7036_ml_challenge_2026_video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
print(json.dumps({'video_seconds': duration, 'fps': fps, 'pdf_pages': len(pdf)}), flush=True)
frames = []
for sec in range(0, int(duration), 20):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(str(OUT / f'video_{sec:04}.jpg'), frame)
        frame = cv2.resize(frame, (640, 360))
        cv2.putText(frame, f'{sec//60}:{sec%60:02}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        frames.append(frame)
cap.release()
import numpy as np
for start in range(0, len(frames), 12):
    group=frames[start:start+12]
    while len(group)%2: group.append(np.zeros_like(group[0]))
    cv2.imwrite(str(OUT/f'contact_{start//12}.jpg'), np.vstack([np.hstack(group[i:i+2]) for i in range(0,len(group),2)]))
