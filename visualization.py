import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


with open('data/activitynet_annotations/anet_anno_action.json', 'r') as f:
    ground_truths = json.load(f)

with open('output/result_proposal.json', 'r') as f:
    results = json.load(f)['results']

for vid_id in tqdm(results.keys()):
    fig = plt.figure(figsize=(6.4, 1.8))
    ax = plt.gca()

    duration, gt_segments = ground_truths['v_' + vid_id]['duration_second'], [gt['segment'] for gt in ground_truths['v_' + vid_id]['annotations']]
    gt_segments = np.array(gt_segments)

    ax.hlines(np.arange(len(gt_segments)) + 1, gt_segments[:, 0], gt_segments[:, 1], color='b', linewidth=3)

    segments = []
    for proposal in results[vid_id]:
        score, segment = proposal['score'], proposal['segment']
        if score >= 0.5:
            segments.append(segment)

    h = float(len(gt_segments) + len(segments) + 1) / 20
    w = duration / 100
    

    for i, segment in enumerate(gt_segments):
       ax.vlines(segment[0], i + 1 - h, i + 1 + h, color='b', linewidth=3)
       plt.text(segment[0] - w, i + 1, int(np.around(segment[0])), ha='right', va='center')
       ax.vlines(segment[1], i + 1 - h, i + 1 + h, color='b', linewidth=3)
       plt.text(segment[1] + w, i + 1, int(np.around(segment[1])), ha='left', va='center')

    idx = len(gt_segments) + 1
    for segment in segments:
        ax.hlines(idx, segment[0], segment[1], color='r', linewidth=3)
        ax.vlines([segment[0], segment[1]], idx - h, idx + h, color='r', linewidth=3)
        plt.text(segment[0] - w, idx, int(np.around(segment[0])), ha='right', va='center')
        plt.text(segment[1] + w, idx, int(np.around(segment[1])), ha='left', va='center')
        idx += 1

    ax.set_xlim([-1, duration + 1])
    ax.set_ylim([0, idx])
    ax.get_yaxis().set_visible(False)
    plt.grid()
    plt.savefig('visualizations/' + vid_id)
    plt.close()

