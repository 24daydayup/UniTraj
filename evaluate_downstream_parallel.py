"""
Multi-GPU Parallel Evaluation Script for UniTraj downstream tasks
Modified for parallel execution on 3 GPUs (0, 1, 2)
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import time

# Import project modules
from utils.unitraj_diffusion import UniTrajDiffusion
from utils.dataset import TrajectoryDataset, Normalize
from utils.logger import Logger
from torch.utils.data import DataLoader


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in meters
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in meters
    r = 6371000
    return c * r


class TrajectoryEvaluator:
    """Evaluator for trajectory prediction and completion tasks"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.logger = Logger(log_path="evaluation_results.log")
        
    def denormalize_trajectory(self, traj_normalized, original, transform):
        """
        Denormalize trajectory back to original coordinate space
        Args:
            traj_normalized: [L, 2] normalized trajectory (lon, lat)
            original: [2] original offset (lon, lat)
            transform: Normalize transform object
        Returns:
            traj_denorm: [L, 2] denormalized trajectory in real coordinates
        """
        # 正确的反归一化流程：
        # 1. 先反归一化：traj_normalized * std + mean
        # 2. 再添加原始偏移量恢复真实坐标
        
        # 转换为tensor进行反归一化
        traj_tensor = torch.tensor(traj_normalized, dtype=torch.float32)
        
        # 反归一化：traj_normalized * std + mean
        traj_denorm_tensor = traj_tensor * transform.std + transform.mean
        
        # 转换为numpy并添加原始偏移量
        traj_denorm = traj_denorm_tensor.numpy() + original.cpu().numpy()
        
        return traj_denorm

    def calculate_mae(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
        """
        Mean Absolute Error in meters using Haversine distance
        """
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return 0.0
        
        pred_valid = pred[valid_indices]
        gt_valid = gt[valid_indices]
        
        # Calculate haversine distance for each point
        distances = []
        for i in range(len(pred_valid)):
            lon1, lat1 = gt_valid[i, 0], gt_valid[i, 1]  # lon, lat
            lon2, lat2 = pred_valid[i, 0], pred_valid[i, 1]
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            distances.append(dist)
        
        return np.mean(distances)

    def calculate_rmse(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
        """
        Root Mean Squared Error in meters using Haversine distance
        """
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return 0.0
        
        pred_valid = pred[valid_indices]
        gt_valid = gt[valid_indices]
        
        # Calculate haversine distance for each point
        distances = []
        for i in range(len(pred_valid)):
            lon1, lat1 = gt_valid[i, 0], gt_valid[i, 1]
            lon2, lat2 = pred_valid[i, 0], pred_valid[i, 1]
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            distances.append(dist)
        
        return np.sqrt(np.mean(np.array(distances) ** 2))

    def evaluate_batch(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor, 
                      eval_mask: torch.Tensor, originals: torch.Tensor,
                      transform) -> Dict:
        """
        Evaluate a batch of trajectories with denormalization
        """
        batch_size = pred_traj.shape[0]
        mae_list, rmse_list = [], []
        
        pred_np = pred_traj.cpu().numpy()  # [B, 2, L]
        gt_np = gt_traj.cpu().numpy()      # [B, 2, L]
        mask_np = eval_mask.cpu().numpy()  # [B, L]
        
        for i in range(batch_size):
            pred_i = pred_np[i].T  # [L, 2] (lon, lat)
            gt_i = gt_np[i].T      # [L, 2]
            mask_i = mask_np[i]    # [L]
            original_i = originals[i]  # [2]
            
            if mask_i.sum() == 0:
                continue
            
            # Denormalize trajectories
            pred_denorm = self.denormalize_trajectory(pred_i, original_i, transform)
            gt_denorm = self.denormalize_trajectory(gt_i, original_i, transform)
            
            # Calculate metrics in meters
            mae = self.calculate_mae(pred_denorm, gt_denorm, mask_i)
            rmse = self.calculate_rmse(pred_denorm, gt_denorm, mask_i)
            
            mae_list.append(mae)
            rmse_list.append(rmse)
        
        return {
            'MAE': np.mean(mae_list) if mae_list else 0.0,
            'RMSE': np.mean(rmse_list) if rmse_list else 0.0,
        }


def ddim_sample(model, x_obs, mask, intervals, n_steps=50):
    """
    DDIM sampling for trajectory completion/prediction
    """
    device = x_obs.device
    B, _, L = x_obs.shape

    # === Get encoder features ===
    if intervals is not None:
        intervals_emb = intervals.unsqueeze(-1)  # [B, L, 1]
        interval_embeddings = model.interval_embedding(intervals_emb)  # [B, L, C]
    else:
        interval_embeddings = model.interval_embedding(torch.zeros(B, L, 1, device=device))

    # [FIX-C] Build per-sample mask index lists (no padding to avoid treating 0 as masked)
    mask_indices = [torch.where(mask[b, 0] == 1)[0].cpu().numpy() for b in range(B)]
    features, _ = model.encoder(x_obs, interval_embeddings, mask_indices)
    enc_feat = features[0]  # [B, C]

    # === DDIM sampling with inpainting constraint on observed points ===
    x_t = torch.randn_like(x_obs)  # Start from noise
    delta_t = intervals.unsqueeze(1) if intervals is not None else torch.zeros(B, 1, L, device=device)
    z = torch.randn_like(x_obs)     # [FIX-C] fixed noise to re-noise observed points each step

    times = torch.linspace(model.T - 1, 0, steps=n_steps + 1, device=device, dtype=torch.long)
    
    for i in range(n_steps):
        t_now = times[i]
        t_next = times[i + 1]
        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        # [FIX-C] re-noise observed positions to q(x_t | x0 = x_obs) and clamp them
        alphabar_now = model.alphabar[t_now].view(1, 1, 1)
        x_obs_t = torch.sqrt(alphabar_now) * x_obs + torch.sqrt(1.0 - alphabar_now) * z
        x_t = x_t * mask + x_obs_t * (1 - mask)

        # predict epsilon and DDIM update
        eps_pred = model.denoiser(x_t, t_batch.float(), x_obs, mask, delta_t, enc_feat=enc_feat)

        alphabar_next = model.alphabar[t_next].view(1, 1, 1)
        x0_pred = (x_t - torch.sqrt(1.0 - alphabar_now) * eps_pred) / torch.sqrt(alphabar_now)
        x_t = torch.sqrt(alphabar_next) * x0_pred + torch.sqrt(1.0 - alphabar_next) * eps_pred
    
    # Combine observed and predicted
    x_pred = x_obs * (1 - mask) + x_t * mask
    return x_pred


def evaluate_model_on_gpu(gpu_id: int, model_files: List[str], result_queue):
    """
    Evaluate models on a specific GPU
    Args:
        gpu_id: GPU device ID (0, 1, or 2)
        model_files: List of model weight files to evaluate
        result_queue: Queue for collecting results
    """
    device = torch.device(f'cuda:{gpu_id}')
    
    # Prepare dataset (shared across all models)
    test_file_path = "data/worldtrace_test.pkl"
    normalize_transform = Normalize()
    test_dataset = TrajectoryDataset(
        data_path=test_file_path,
        max_len=200,
        transform=normalize_transform,
        mask_ratio=0.5
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )
    
    # Initialize evaluator
    evaluator = TrajectoryEvaluator(device=device)
    
    for model_file in model_files:
        try:
            print(f"GPU {gpu_id}: Loading model {model_file}")
            
            # Load model
            model = UniTrajDiffusion(
                trajectory_length=200,
                patch_size=1,
                embedding_dim=128,
                encoder_layers=8,
                encoder_heads=4,
                mask_ratio=0.5,
                T=1000,
            )
            
            model_path = f"/vol/zc/UniTraj/UniTraj/worldtrace_bs=1024/10-22-18-41-41/models/{model_file}"
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            
            # Evaluate trajectory prediction
            print(f"GPU {gpu_id}: Evaluating prediction for {model_file}")
            prediction_results = evaluate_trajectory_prediction(
                model=model,
                dataloader=test_dataloader,
                evaluator=evaluator,
                device=device,
                transform=normalize_transform,
                n_steps=50,
                num_predict=8,
                save_dir=f'results/prediction_gpu{gpu_id}_{model_file.replace(".pt", "")}'
            )
            
            # Evaluate trajectory completion
            print(f"GPU {gpu_id}: Evaluating completion for {model_file}")
            completion_results = evaluate_trajectory_completion(
                model=model,
                dataloader=test_dataloader,
                evaluator=evaluator,
                device=device,
                transform=normalize_transform,
                n_steps=50,
                save_dir=f'results/completion_gpu{gpu_id}_{model_file.replace(".pt", "")}'
            )
            
            # Collect results
            result = {
                'gpu_id': gpu_id,
                'model_file': model_file,
                'prediction_mae': prediction_results['MAE'],
                'prediction_rmse': prediction_results['RMSE'],
                'completion_mae': completion_results['MAE'],
                'completion_rmse': completion_results['RMSE'],
                'timestamp': time.time()
            }
            
            result_queue.put(result)
            print(f"GPU {gpu_id}: Completed evaluation for {model_file}")
            
        except Exception as e:
            print(f"GPU {gpu_id}: Error evaluating {model_file}: {e}")
            result_queue.put({
                'gpu_id': gpu_id,
                'model_file': model_file,
                'error': str(e)
            })


def evaluate_trajectory_prediction(model, dataloader, evaluator, device, transform,
                                   n_steps=50, num_predict=8, save_dir='results/prediction'):
    """
    Evaluate trajectory prediction task (predict last N points)
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {'MAE': [], 'RMSE': []}
    
    evaluator.logger.info("Evaluating Trajectory Prediction...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            traj = batch['trajectory'].to(device)  # [B, 2, L]
            intervals = batch['intervals'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            originals = batch['original'].to(device)  # [B, 2]
            
            B, _, L = traj.shape
            
            # Create prediction mask: mask last num_predict points
            pred_mask = torch.zeros(B, 1, L, device=device)
            for b in range(B):
                valid_len = attention_mask[b].sum().item()
                if valid_len > num_predict:
                    pred_mask[b, 0, int(valid_len - num_predict):int(valid_len)] = 1
            
            # Observed trajectory (set masked points to 0)
            x_obs = traj * (1 - pred_mask)
            
            # DDIM sampling
            x_pred = ddim_sample(model, x_obs, pred_mask, intervals, n_steps=n_steps)
            
            # [FIX-A] Evaluate only on predicted points within valid (non-PAD) region
            # boolean mask keeps evaluator logic unchanged
            eval_mask = ((pred_mask[:, 0, :] > 0) & (attention_mask > 0))
            results = evaluator.evaluate_batch(x_pred, traj, eval_mask, originals, transform)
            
            all_results['MAE'].append(results['MAE'])
            all_results['RMSE'].append(results['RMSE'])
    
    # Calculate average metrics
    final_results = {
        'MAE': float(np.mean(all_results['MAE'])),
        'RMSE': float(np.mean(all_results['RMSE'])),
    }
    
    evaluator.logger.info(f"Prediction Results - MAE: {final_results['MAE']:.2f}m, "
                         f"RMSE: {final_results['RMSE']:.2f}m")
    
    # Save results
    with open(save_dir / 'prediction_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    return final_results


def evaluate_trajectory_completion(model, dataloader, evaluator, device, transform,
                                   n_steps=50, save_dir='results/completion'):
    """
    Evaluate trajectory completion task (complete missing points using dataset masks)
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {'MAE': [], 'RMSE': []}
    
    evaluator.logger.info("Evaluating Trajectory Completion...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            traj = batch['trajectory'].to(device)  # [B, 2, L]
            intervals = batch['intervals'].to(device)
            indices = batch['indices']  # Mask indices from dataset
            attention_mask = batch['attention_mask'].to(device)
            originals = batch['original'].to(device)  # [B, 2]
            
            B, _, L = traj.shape
            
            # Create mask from indices (only within valid length)
            mask = torch.zeros(B, 1, L, device=device)
            for b in range(B):
                if isinstance(indices, torch.Tensor):
                    mask_idx = indices[b].to(device)
                else:
                    mask_idx = torch.tensor(indices[b], device=device)
                valid_len = int(attention_mask[b].sum().item())
                if valid_len > 0:
                    valid_idx = mask_idx[(mask_idx >= 0) & (mask_idx < valid_len)]
                    if valid_idx.numel() > 0:
                        mask[b, 0, valid_idx] = 1
            
            # Observed trajectory
            x_obs = traj * (1 - mask)
            
            # DDIM sampling
            x_pred = ddim_sample(model, x_obs, mask, intervals, n_steps=n_steps)
            
            # [FIX-A] Evaluate only on masked points that are valid (exclude PAD)
            eval_mask = ((mask[:, 0, :] > 0) & (attention_mask > 0))
            results = evaluator.evaluate_batch(x_pred, traj, eval_mask, originals, transform)
            
            all_results['MAE'].append(results['MAE'])
            all_results['RMSE'].append(results['RMSE'])
    
    # Calculate average metrics
    final_results = {
        'MAE': float(np.mean(all_results['MAE'])),
        'RMSE': float(np.mean(all_results['RMSE'])),
    }
    
    evaluator.logger.info(f"Completion Results - MAE: {final_results['MAE']:.2f}m, "
                         f"RMSE: {final_results['RMSE']:.2f}m")
    
    # Save results
    with open(save_dir / 'completion_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    return final_results


def main():
    """Main function for multi-GPU parallel evaluation"""
    # Get all model files
    model_dir = "/vol/zc/UniTraj/UniTraj/worldtrace_bs=1024/10-22-18-41-41/models"
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pt')])
    
    print(f"Found {len(model_files)} model files:")
    for i, model_file in enumerate(model_files):
        print(f"  {i+1}. {model_file}")
    
    # Distribute models across 3 GPUs
    gpu_assignments = {0: [], 1: [], 2: []}
    for i, model_file in enumerate(model_files):
        gpu_id = i % 3
        gpu_assignments[gpu_id].append(model_file)
    
    print("\nGPU assignments:")
    for gpu_id, files in gpu_assignments.items():
        print(f"GPU {gpu_id}: {len(files)} models")
    
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    
    # Start processes
    processes = []
    for gpu_id, model_files in gpu_assignments.items():
        if model_files:  # Only start process if there are models to evaluate
            p = mp.Process(target=evaluate_model_on_gpu, 
                          args=(gpu_id, model_files, result_queue))
            processes.append(p)
            p.start()
    
    # Collect results
    all_results = []
    for _ in range(len(model_files)):
        result = result_queue.get()
        all_results.append(result)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Sort results by model file name
    all_results.sort(key=lambda x: x['model_file'])
    
    # Save comprehensive results
    results_dir = Path("parallel_evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(results_dir / 'detailed_results.json', 'w') as f:
        json.dump(all_results, f, indent=4, default=str)
    
    # Save summary results
    summary = []
    for result in all_results:
        if 'error' not in result:
            summary.append({
                'model_file': result['model_file'],
                'gpu_id': result['gpu_id'],
                'prediction_mae': result['prediction_mae'],
                'prediction_rmse': result['prediction_rmse'],
                'completion_mae': result['completion_mae'],
                'completion_rmse': result['completion_rmse']
            })
    
    with open(results_dir / 'summary_results.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print final summary
    print("\n" + "="*80)
    print("PARALLEL EVALUATION SUMMARY")
    print("="*80)
    
    for result in summary:
        print(f"\nModel: {result['model_file']} (GPU {result['gpu_id']})")
        print(f"  Prediction - MAE: {result['prediction_mae']:.2f}m, RMSE: {result['prediction_rmse']:.2f}m")
        print(f"  Completion - MAE: {result['completion_mae']:.2f}m, RMSE: {result['completion_rmse']:.2f}m")
    
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()