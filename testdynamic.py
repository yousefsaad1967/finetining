import os
import json
import math
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

from GRU_BERT_model import GRUBERT
from threshold import Threshold
from clustering import HierarchicalClustering

class ConfigArgs:
    def __init__(self):
        self.window_size = 480
        self.drop_out = 0.1
        self.output_size = 1
        self.window_stride = 480 
        self.batch_size = 32
        
        self.cutoff = 400.0 
        self.min_on = 60
        self.min_off = 12
        
        self.threshold_method = 'mp' 
        self.n_clusters = 2
        self.threshold = 40.0

class NILMDataset(Dataset):
    def __init__(self, x, y, status, window_size=480, stride=30):
        self.x = x
        self.y = y
        self.status = status
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        num_samples = max(0, (len(self.x) - self.window_size))
        return int(np.ceil(num_samples / self.stride) + 1)

    def __getitem__(self, index):
        start_index = index * self.stride
        end_index = min(len(self.x), start_index + self.window_size)
        
        seq = self.padding_seqs(self.x[start_index : end_index])
        target_energy = self.padding_seqs(self.y[start_index : end_index])
        target_status = self.padding_seqs(self.status[start_index : end_index])
        
        return (torch.tensor(seq, dtype=torch.float64), 
                torch.tensor(target_energy, dtype=torch.float64).unsqueeze(-1), 
                torch.tensor(target_status, dtype=torch.float64).unsqueeze(-1))

    def padding_seqs(self, in_array):
        if len(in_array) == self.window_size:
            return in_array
        out_array = np.zeros(self.window_size)
        out_array[:len(in_array)] = in_array
        return out_array

def load_test_data(csv_filepath, args, skip_rows=1000000, num_rows=20000, pretrained_stats=None):
    train_data = pd.read_csv(csv_filepath, nrows=100000)
    
    train_x = train_data['Aggregate'].values.astype(np.float64)
    train_y = train_data['Appliance1'].values.astype(np.float64)
    
    if pretrained_stats is not None:
        x_mean, x_std = pretrained_stats
    else:
        x_mean = np.mean(train_x)
        x_std = np.std(train_x)
    
    if args.threshold_method in ['vs', 'mp']:
        thresh_manager = Threshold(appliances=['Appliance1'], method=args.threshold_method, num_status=args.n_clusters)
        thresh_manager.update_appliance_threshold(train_y, 'Appliance1')
    elif args.threshold_method == 'custom':
        hc_model = HierarchicalClustering(distance="average", n_cluster=args.n_clusters)
        hc_model.perform_clustering(train_y)
        hc_model.compute_thresholds_and_centroids(centroid="median")
        thresh_manager = Threshold(appliances=['Appliance1'], method="custom")
        thresh_manager.set_thresholds_and_centroids(
            np.expand_dims(hc_model.thresh, axis=0),
            np.expand_dims(hc_model.centroids, axis=0)
        )
    
    raw_threshold = thresh_manager.thresholds[0][1]
    args.threshold = float(np.clip(raw_threshold, 30.0, 60.0))
    
    print(f"---> [Testing] Raw Threshold: {raw_threshold:.2f} Watts")
    print(f"---> [Testing] Clipped Threshold via {args.threshold_method}: {args.threshold:.2f} Watts")
    
    skip_list = range(1, skip_rows + 1)
    data = pd.read_csv(csv_filepath, skiprows=skip_list, nrows=num_rows)
    
    x = data['Aggregate'].values.astype(np.float64)
    y = data['Appliance1'].values.astype(np.float64)
    
    x = (x - x_mean) / (x_std + 1e-6)
    
    y_reshaped = y.reshape(-1, 1)
    initial_status = thresh_manager.power_to_status(y_reshaped).flatten()
    status = np.zeros_like(y)
    
    status_diff = np.diff(initial_status.astype(int))
    events_idx = status_diff.nonzero()[0] + 1

    if len(events_idx) == 0:
        return x, y, status

    if initial_status[0]:
        events_idx = np.insert(events_idx, 0, 0)
    if initial_status[-1]:
        events_idx = np.append(events_idx, len(initial_status))

    if events_idx.size % 2 != 0:
        events_idx = np.append(events_idx, len(initial_status))

    events_idx = events_idx.reshape((-1, 2))
    on_events, off_events = events_idx[:, 0], events_idx[:, 1]

    if len(on_events) > 0:
        off_duration = np.insert(on_events[1:] - off_events[:-1], 0, 1000)
        on_events = on_events[off_duration > args.min_off]
        off_events = off_events[np.roll(off_duration, -1) > args.min_off]

        on_duration = off_events - on_events
        mask = on_duration >= args.min_on
        on_events = on_events[mask]
        off_events = off_events[mask]

    for on, off in zip(on_events, off_events):
        status[on:off] = 1.0
        
    return x, y, status

def test_model():
    torch.set_default_dtype(torch.float64)
    args = ConfigArgs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    UK_DALE_MEAN = 418.623901
    UK_DALE_STD = 504.039630
    pretrained_stats = (UK_DALE_MEAN, UK_DALE_STD)
    
    x, y, status = load_test_data('refit/CLEAN_House2.csv', args, skip_rows=1000000, num_rows=20000, pretrained_stats=pretrained_stats)
    
    test_dataset = NILMDataset(x, y, status, args.window_size, args.window_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = GRUBERT(args)
    
    model_path = 'models/House_2_model.pth'
    # model_path = 'fridge.pth'
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()
    
    all_gt_energy = []
    all_pred_energy = []
    all_gt_status = []
    all_pred_status = []
    
    with torch.no_grad():
        for seqs, labels_energy, status_batch in test_loader:
            seqs = seqs.to(device)
            labels_energy = labels_energy.to(device)
            status_batch = status_batch.to(device)
            
            logits = model(seqs)
            
            logits_energy = logits * args.cutoff
            logits_energy[logits_energy < 5] = 0
            logits_energy = torch.min(logits_energy, torch.tensor(args.cutoff, device=device).double())
            
            logits_status = (logits_energy >= args.threshold) * 1.0
            logits_energy = logits_energy * logits_status
            
            all_gt_energy.append(labels_energy.cpu().numpy())
            all_pred_energy.append(logits_energy.cpu().numpy())
            all_gt_status.append(status_batch.cpu().numpy())
            all_pred_status.append(logits_status.cpu().numpy())
            
    gt_e = np.concatenate(all_gt_energy).reshape(-1)
    pred_e = np.concatenate(all_pred_energy).reshape(-1)
    gt_s = np.concatenate(all_gt_status).reshape(-1)
    pred_s = np.concatenate(all_pred_status).reshape(-1)
    
    abs_err = np.mean(np.abs(gt_e - pred_e))
    
    temp = np.full(gt_e.shape, 1e-9)
    rel_err = np.mean(np.nan_to_num(np.abs(gt_e - pred_e) / np.max((gt_e, pred_e, temp), axis=0)))
    
    tn, fp, fn, tp = confusion_matrix(gt_s, pred_s, labels=[0, 1]).ravel()
    acc = (tn + tp) / max((tn + fp + fn + tp), 1)
    precision = tp / max((tp + fp), 1)
    recall = tp / max((tp + fn), 1)
    f1 = 2 * (precision * recall) / max((precision + recall), 1e-9)
    
    print(f"Mean Absolute Error (MAE): {abs_err:.4f}")
    print(f"Mean Relative Error (MRE): {rel_err:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    os.makedirs('logs', exist_ok=True)
    
    result_dict = {
        'metrics': {
            'MAE': float(abs_err),
            'MRE': float(rel_err),
            'Accuracy': float(acc),
            'F1_Score': float(f1),
            'Precision': float(precision),
            'Recall': float(recall)
        },
        'predictions': {
            'ground_truth_energy': gt_e.tolist(),
            'predicted_energy': pred_e.tolist(),
            'ground_truth_status': gt_s.tolist(),
            'predicted_status': pred_s.tolist()
        }
    }
    
    # Create dynamic JSON filename based on the model name
    json_filename = f'logs/{model_name}_1mtest_result.json'
    
    with open(json_filename, 'w') as f:
        json.dump(result_dict, f, indent=4)
        
    print(f"Testing completed. Results saved in {json_filename}")
if __name__ == "__main__":
    test_model()