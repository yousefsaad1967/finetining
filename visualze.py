import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def plot_comparison(log1_path, log2_path, model1_name="Model 1", model2_name="Model 2", num_samples=1000):
    os.makedirs('image', exist_ok=True)
    
    data1 = load_results(log1_path)
    data2 = load_results(log2_path)
    
    metrics1 = data1['metrics']
    metrics2 = data2['metrics']
    
    large_metrics = ['MAE']
    small_metrics = ['MRE', 'Accuracy', 'F1_Score', 'Precision', 'Recall']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    width = 0.35
    
    x1 = np.arange(len(large_metrics))
    v1_large = [metrics1[m] for m in large_metrics]
    v2_large = [metrics2[m] for m in large_metrics]
    
    ax1.bar(x1 - width/2, v1_large, width, label=model1_name, color='#87CEEB')
    ax1.bar(x1 + width/2, v2_large, width, label=model2_name, color='#FA8072')
    
    ax1.set_ylabel('Error Value')
    ax1.set_title('Mean Absolute Error (Lower is Better)')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(large_metrics)
    ax1.legend()
    
    for i, v in enumerate(v1_large):
        ax1.text(i - width/2, v + (max(v1_large + v2_large) * 0.02), f"{v:.2f}", ha='center', fontweight='bold')
    for i, v in enumerate(v2_large):
        ax1.text(i + width/2, v + (max(v1_large + v2_large) * 0.02), f"{v:.2f}", ha='center', fontweight='bold')

    x2 = np.arange(len(small_metrics))
    v1_small = [metrics1[m] for m in small_metrics]
    v2_small = [metrics2[m] for m in small_metrics]
    
    ax2.bar(x2 - width/2, v1_small, width, label=model1_name, color='#87CEEB')
    ax2.bar(x2 + width/2, v2_small, width, label=model2_name, color='#FA8072')
    
    ax2.set_ylabel('Score (0 to 1)')
    ax2.set_title('Performance Scores (Higher is Better)')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(small_metrics)
    ax2.set_ylim(0, max(max(v1_small), max(v2_small)) + 0.15) 
    ax2.legend()
    
    for i, v in enumerate(v1_small):
        ax2.text(i - width/2, v + 0.02, f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')
    for i, v in enumerate(v2_small):
        ax2.text(i + width/2, v + 0.02, f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('image/metrics_comparison2(1m).png', dpi=300)
    plt.show()

    gt_energy = data1['predictions']['ground_truth_energy'][:num_samples]
    pred_energy1 = data1['predictions']['predicted_energy'][:num_samples]
    pred_energy2 = data2['predictions']['predicted_energy'][:num_samples]
    
    plt.figure(figsize=(16, 6))
    plt.plot(gt_energy, label='Ground Truth', color='black', linewidth=2, alpha=0.7)
    plt.plot(pred_energy1, label=f'Predicted ({model1_name})', linestyle='--', color='blue', alpha=0.8)
    plt.plot(pred_energy2, label=f'Predicted ({model2_name})', linestyle=':', color='red', alpha=0.8)
    
    plt.title(f'Energy Prediction Comparison (First {num_samples} Samples)')
    plt.xlabel('Time Step')
    plt.ylabel('Energy (Watts)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('image/timeseries_comparison2(1m).png', dpi=300)
    plt.show()

if __name__ == "__main__":
    log_file_1 = 'logs/fridge_1mtest_result.json'
    log_file_2 = 'logs/House_2_model_1mtest_result.json'  
    
    plot_comparison(
        log1_path=log_file_1, 
        log2_path=log_file_2, 
        model1_name="Fridge Model (House 2)", 
        model2_name="Fine-tuned Model (House 2)",
        num_samples=800  
    )