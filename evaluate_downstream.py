"""
Evaluation script for UniTraj downstream tasks: Trajectory Prediction and Completion
Following NeurIPS 2025 paper: UniTraj
Core metrics: MAE, RMSE (in meters)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict
import sys

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
        # Reverse normalization: multiply by std and add mean
        if hasattr(transform, 'mean') and hasattr(transform, 'std'):
            # Convert tensors to numpy for proper broadcasting
            mean_np = transform.mean.cpu().numpy()
            std_np = transform.std.cpu().numpy()
            traj_denorm = traj_normalized * std_np + mean_np
        else:
            traj_denorm = traj_normalized
        
        # Add back the original offset
        traj_denorm = traj_denorm + original.cpu().numpy()
        
        return traj_denorm
    
    def calculate_mae(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
        """
        Mean Absolute Error in meters using Haversine distance
        Args:
            pred: [L, 2] predicted trajectory (lon, lat) in degrees
            gt: [L, 2] ground truth trajectory (lon, lat) in degrees
            mask: [L] boolean mask for valid points
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
        Args:
            pred_traj: [B, 2, L] predicted trajectories (normalized)
            gt_traj: [B, 2, L] ground truth trajectories (normalized)
            eval_mask: [B, L] boolean mask for evaluation points
            originals: [B, 2] original offsets
            transform: Normalize transform
        Returns:
            Dictionary with MAE, RMSE metrics in meters
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
    Args:
        model: UniTrajDiffusion model
        x_obs: [B, 2, L] observed trajectory points
        mask: [B, 1, L] mask (1 for missing, 0 for observed)
        intervals: [B, L] time intervals
        n_steps: number of DDIM sampling steps
    Returns:
        x_pred: [B, 2, L] completed trajectory
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
    """Main evaluation function"""
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # Load trained weights
    model_path = "/vol/zc/UniTraj/UniTraj/worldtrace_bs=1024/10-21-09-44-26/models/best_model_epoch_58.pt"
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Prepare dataset
    test_file_path = "data/worldtrace_test.pkl"
    print(f"Loading dataset from: {test_file_path}")
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
    
    # Evaluate trajectory prediction
    print("\n" + "="*50)
    print("Evaluating Trajectory Prediction Task")
    print("="*50)
    prediction_results = evaluate_trajectory_prediction(
        model=model,
        dataloader=test_dataloader,
        evaluator=evaluator,
        device=device,
        transform=normalize_transform,
        n_steps=50,  # DDIM sampling steps
        num_predict=8,  # Predict last 8 points
        save_dir='results/prediction'
    )
    
    # Evaluate trajectory completion
    print("\n" + "="*50)
    print("Evaluating Trajectory Completion Task")
    print("="*50)
    completion_results = evaluate_trajectory_completion(
        model=model,
        dataloader=test_dataloader,
        evaluator=evaluator,
        device=device,
        transform=normalize_transform,
        n_steps=50,  # DDIM sampling steps
        save_dir='results/completion'
    )
    
    # Print summary
    print("\n" + "="*50)
    print("Evaluation Summary (in meters)")
    print("="*50)
    print("\nTrajectory Prediction:")
    print(f"  MAE:  {prediction_results['MAE']:.2f}")
    print(f"  RMSE: {prediction_results['RMSE']:.2f}")
    
    print("\nTrajectory Completion:")
    print(f"  MAE:  {completion_results['MAE']:.2f}")
    print(f"  RMSE: {completion_results['RMSE']:.2f}")


if __name__ == "__main__":
    main()
