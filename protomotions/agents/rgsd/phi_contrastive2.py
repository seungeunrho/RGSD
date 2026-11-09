import os
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE

import torch
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    def __init__(self, motions, n_frames=1):
        self.n_frames = n_frames
        self.data = []

        for motion_id, (motion, _) in enumerate(motions):
            motion_len = len(motion)
            for i in range(motion_len):
                self.data.append((motion_id, i))

        self.motions = motions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        motion_id, curr_idx = self.data[idx]
        motion, _ = self.motions[motion_id]
        feature_dim = len(motion[0])

        # Indices from (curr_idx - n_frames + 1) to curr_idx
        start_idx = curr_idx - self.n_frames + 1
        frames = []

        for i in range(start_idx, curr_idx + 1):
            if i < 0:
                frames.append(torch.tensor(motion[0], dtype=torch.float32))
            else:
                frames.append(torch.tensor(motion[i], dtype=torch.float32))

        frames_tensor = torch.stack(frames).flip(0).reshape(-1)

        return frames_tensor, torch.tensor(motion_id, dtype=torch.long)



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class ContrastiveLearning:
    def __init__(self, cfg, motion_tensor_lst, dt, n_history, latent_dim, device):
        self.motion_tensor_lst = motion_tensor_lst
        self.motions = self._read_data_file(motion_tensor_lst, dt, max_length=cfg.max_motion_length)
        self.cfg = cfg
        
        self.dataset = MotionDataset(self.motions, n_frames=n_history)
        self.dataloader = DataLoader(self.dataset, batch_size=cfg.batch_size, shuffle=True)
        self.dataloader_iter = iter(self.dataloader)

        self.motion_index = 0

        self.device = device
        self.dataloader_counter = 0
        self.latent_dim = latent_dim

        self.motion_std_tensor = self._compute_std_for_each_motion(motion_tensor_lst)
        self.n_history = n_history
        self.batch_size = cfg.batch_size
        self.attach_encoder_phase_var = cfg.attach_encoder_phase_var




    def load_model(self, model_state_dict):
        self.model.model.load_state_dict(model_state_dict, strict=False)

    def _compute_std_for_each_motion(self, motion_tensor_lst):
        motion_std_lst = []
        for motion_tensor in motion_tensor_lst:
            std_tensor = motion_tensor.std(dim=0)
            motion_std_lst.append(std_tensor)
        return torch.stack(motion_std_lst)

    def _read_data_file(self, motion_tensor_lst, dt, max_length=150):
        tot_traj = []
        for i, traj in enumerate(motion_tensor_lst):
            traj = motion_tensor_lst[i].cpu().numpy()
            data_length = min(len(traj), max_length)
            capped_traj = traj[:data_length]
            tot_traj.append((capped_traj, f"traj{i}"))
        return tot_traj

    def draw_figures_of_data(self, x_idx, y_idx):
        plt.figure(figsize=(10, 6))
        for motion in self.motions:
            plt.scatter(motion[0][:, x_idx], motion[0][:, y_idx],
                        label=f'{motion[1]}')

        plt.grid(True)
        plt.legend()
        plt.show()

    def add_columnwise_scaled_noise(self, mini_batch: torch.Tensor, col_std: torch.Tensor, noise_scale: float = 0.1) -> torch.Tensor:
        """
        Add column-wise Gaussian noise to a tensor, scaled by each column's standard deviation.

        Args:
            tensor (torch.Tensor): Input tensor of shape [N, D].
            noise_scale (float): Multiplier for the noise (relative to each column's std).

        Returns:
            torch.Tensor: Tensor with added column-wise scaled noise.
        """
        if self.attach_encoder_phase_var:
            start_idx = 1
        else:
            start_idx = 0

        n_data = mini_batch.size()[0]

        # Generate noise with shape [N, D], scaled per colum
        if self.n_history > 1:
            mini_batch = mini_batch.view(n_data, self.n_history, -1)
            noise = torch.randn_like(mini_batch[:,:,start_idx:]) \
                    * col_std.unsqueeze(1)[:,:,start_idx:] \
                    * noise_scale
            mini_batch[:,:,start_idx:] += noise
            mini_batch = mini_batch.view(n_data, -1)
        else:
            noise = torch.randn_like(mini_batch[:,start_idx:]) * col_std[:,start_idx:] * noise_scale
            mini_batch[:,1:] += noise

        return mini_batch

    def compute_loss_for_batch(self, model, batch):
        points, labels = batch
        points = points.to(self.device)
        labels = labels.to(self.device)
        # Forward pass
        orig_embeddings = model({"historical_self_obs": points})
        embeddings = nn.functional.normalize(orig_embeddings, p=2, dim=-1)

        # --- InfoNCE Loss ---
        temperature = 1.0
        B = embeddings.size(0)
        # Compute similarity matrix
        logits = torch.matmul(embeddings, embeddings.T) / temperature
        # For each anchor, define positives and negatives
        # Here, positives are any other sample with the same label.
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # (B, B)
        # Remove self-positives
        mask.fill_diagonal_(False)

        # For cross-entropy, pick one positive index per anchor at random
        positive_indices = []
        for i in range(B):
            
            pos_idxs = mask[i].nonzero(as_tuple=True)[0]
            if len(pos_idxs) > 0:
                positive_indices.append(pos_idxs[torch.randint(len(pos_idxs), (1,))])
            else:
                positive_indices.append(torch.tensor([i], device=self.device))

        positive_indices = torch.stack(positive_indices).to(self.device)

        # Cross-entropy expects logits and target indices
        loss_fn = nn.CrossEntropyLoss()
        loss_nce = loss_fn(logits, positive_indices.squeeze(1))

        return loss_nce


    def compute_loss_once(self, model):
        data_continues = True
        batch = next(self.dataloader_iter)
        loss = self.compute_loss_for_batch(model, batch)
        self.dataloader_counter += 1

        if self.dataloader_counter >= len(self.dataloader):
            self.dataloader_iter = iter(self.dataloader)
            self.dataloader_counter = 0
            data_continues = False

        return loss, data_continues

    def train(self, model, optimizer, epochs):
        self.model = model
        for epoch in range(epochs):
            total_loss = 0.0

            # if epoch%self.cfg.draw_interval==0:
            if epoch in [0,1,5,10,100,500,1000,2000]:
                self.draw_latent_figure(epoch)
            for batch in self.dataloader:
                optimizer.zero_grad()
                loss = self.compute_loss_for_batch(self.model, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    def save_model(self):
        torch.save(self.model.model.state_dict(), self.model_path)

    def compute_ref_data_latents(self, model, n_frames=10):
        latent_lst = []

        with torch.no_grad():
            for i, (motion, motion_accronym) in enumerate(self.motions):
                if len(motion) < n_frames:
                    raise ValueError("Motion too short for the given number of frames.")
                
                motion_frame = []
                for j in range(len(motion) - n_frames + 1):
                    motion_frame.append(motion[j:j + n_frames][::-1].reshape(1, -1))

             
                motion_points = torch.tensor(motion_frame, dtype=torch.float32).to(self.device).squeeze(1)
                latent = model({"historical_self_obs":motion_points})
                latent_vec = torch.mean(latent,dim=0)
                latent_lst.append(latent_vec)
                aa=1

        data_latents = torch.stack(latent_lst)
        data_latents = torch.nn.functional.normalize(data_latents, dim=-1)

        return data_latents
    

    def draw_latent_figure(self, model, epoch, file_name, n_frames=10,
                            n_components=3, perplexity=30.0, random_state=0):
        """
        Draw a t-SNE plot of latent embeddings for each trajectory.
        Uses ALL latent dimensions (e.g., 16-D) and projects them to 2D (or 3D).
        """

        fig = plt.figure(figsize=(10, 6))

        # 2D or 3D axes depending on n_components (t-SNE output dim)
        if n_components == 2:
            ax = fig.add_subplot()
        elif n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            raise ValueError("t-SNE n_components must be 2 or 3.")

        all_latents = []
        motion_ranges = []  # list of (start_idx, end_idx, label)
        idx_start = 0

        with torch.no_grad():
            for i, (motion, motion_accronym) in enumerate(self.motions):
                if len(motion) < n_frames:
                    raise ValueError("Motion too short for the given number of frames.")
                
                motion_frame = []
                for j in range(len(motion) - n_frames + 1):
                    # (n_frames, ...) -> flatten
                    motion_frame.append(motion[j:j + n_frames][::-1].reshape(1, -1))

                motion_points = torch.tensor(motion_frame, dtype=torch.float32).to(self.device).squeeze(1)

                # latent_points shape: [num_windows, latent_dim]  (e.g., latent_dim=16)
                latent_points = model({"historical_self_obs": motion_points}).cpu().numpy()

                all_latents.append(latent_points)

                idx_end = idx_start + latent_points.shape[0]
                motion_ranges.append((idx_start, idx_end, motion_accronym))
                idx_start = idx_end

        # Concatenate all latents and run t-SNE on the full latent dimension
        all_latents = np.concatenate(all_latents, axis=0)  # [N_total, latent_dim]

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate='auto',
            init='random',
            random_state=random_state,
        )
        embedded = tsne.fit_transform(all_latents)  # [N_total, n_components]

        # Plot, split back by motion
        for (start, end, label) in motion_ranges:
            pts = embedded[start:end]

            if n_components == 2:
                ax.scatter(pts[:, 0], pts[:, 1], label=label, s=10)
                # (Optional) annotate every 10th point within each motion
                # for k in range(0, pts.shape[0], 10):
                #     ax.text(pts[k, 0], pts[k, 1], str(k), fontsize=6, color='black')
            else:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label=label, s=10)

        ax.set_title(f"t-SNE of latent embeddings (epoch {epoch})")
        if n_components == 2:
            ax.set_xlabel("t-SNE dim 1")
            ax.set_ylabel("t-SNE dim 2")
        else:
            ax.set_xlabel("t-SNE dim 1")
            ax.set_ylabel("t-SNE dim 2")
            ax.set_zlabel("t-SNE dim 3")

        ax.legend()
        plt.tight_layout()
        plt.savefig(file_name, dpi=300)
        plt.close(fig)

        return ax



    # def draw_latent_figure(self, model, epoch, file_name, n_frames=10):
    #     fig = plt.figure(figsize=(10, 6))

    #     if self.latent_dim== 2:
    #         ax = fig.add_subplot()
    #     else:
    #         ax = fig.add_subplot(111, projection='3d')


    #     lst = []
    #     with torch.no_grad():
    #         for i, (motion, motion_accronym) in enumerate(self.motions):
    #             if len(motion) < n_frames:
    #                 raise ValueError("Motion too short for the given number of frames.")
                
    #             motion_frame = []
    #             for j in range(len(motion) - n_frames + 1):
    #                 motion_frame.append(motion[j:j + n_frames][::-1].reshape(1, -1))

             
    #             motion_points = torch.tensor(motion_frame, dtype=torch.float32).to(self.device).squeeze(1)
    #             latent_points = model({"historical_self_obs":motion_points}).cpu().numpy()
    #             lst.append(latent_points)

    #             if self.latent_dim== 2:
    #                 ax.scatter(latent_points[:, 0], latent_points[:, 1],
    #                     label=f'{motion_accronym}')
    #                 for idx in range(0, len(latent_points), 10):
    #                     ax.text(latent_points[idx, 0], latent_points[idx, 1],
    #                             str(idx), fontsize=7, color='black')
    #             else :
    #                 ax.scatter(latent_points[:, 0], latent_points[:, 1], latent_points[:, 2],
    #                     label=f'{motion_accronym}')
    #                 # for idx in range(0, len(latent_points), 10):
    #                 #     ax.text(latent_points[idx, 0], latent_points[idx, 1], latent_points[idx, 2],
    #                 #             str(idx), fontsize=7, color='black')


    #     diffusion_scores = diffusion_score(lst)
    #     diffusion,diffusion_std,  cos_mat = inter_motion_diffusion(lst)

    #     print("---- diffusion score", np.mean(diffusion_scores), np.std(diffusion_scores))


    #     print("Inter-motion diffusion:", diffusion, diffusion_std)
    #     # for i, score in enumerate(diffusion_scores):
    #     #     print(f"Motion {i} diffusion score: {score:.4f}")
    #     # plt.legend()
    #     plt.title(f'Latent embedding at epoch {epoch}')
    #     plt.grid(True)
    #     plt.savefig(file_name)
    #     plt.show()

    #     return ax
    
def diffusion_score(arr_list):
    """
    arr_list: list of numpy arrays, each of shape (n, 16)
    returns: list of diffusion scores (one per array)
    """
    scores = []
    for arr in arr_list:
        # compute mean direction
        mean_dir = arr.mean(axis=0)
        # normalize
        mean_dir = mean_dir / np.linalg.norm(mean_dir)

        # normalize each vector
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr_norm = arr / norms

        # cosine similarity with mean direction
        cos_sims = (arr_norm @ mean_dir)

        # average similarity
        scores.append(cos_sims.mean())

    return scores

def inter_motion_diffusion(arr_list):
    """
    Compute an inter-motion diffusion score.

    Returns:
      diffusion: scalar in [0, 2] roughly, where
        - smaller  -> motions more aligned (less diffused)
        - larger   -> motions more spread (more diffused)
      cos_matrix: (M, M) pairwise cosine similarity between motion mean directions
    """
    mean_dirs = compute_motion_means(arr_list)  # (M, 16)
    cos_matrix = mean_dirs @ mean_dirs.T        # (M, M), since all are unit-length

    norms = np.linalg.norm(cos_matrix, axis=1, keepdims=True)
    cos_matrix = cos_matrix / norms

    # ignore diagonal (self-similarity = 1)
    M = cos_matrix.shape[0]
    mask = ~np.eye(M, dtype=bool)
    mean_cos = cos_matrix[mask].mean()

    diffusion = cos_matrix.mean()   # higher => more diffused between motions
    diffusion_std = cos_matrix[mask].std()   # higher => more diffused between motions

    return diffusion, diffusion_std, cos_matrix

def compute_motion_means(arr_list):
    """
    arr_list: list of arrays, each (n_frames, 16)
    returns: array of shape (n_motions, 16) of normalized mean directions
    """
    means = []
    for arr in arr_list:
        mean_dir = arr.mean(axis=0)
        norm = np.linalg.norm(mean_dir)
        if norm == 0:
            # degenerate case: all zeros (or perfectly symmetric)
            means.append(np.zeros_like(mean_dir))
        else:
            means.append(mean_dir / norm)
    return np.vstack(means)

if __name__ == '__main__':
    set_seed(43)
    epochs = 1100
    batch_size = 256
    learning_rate = 0.001

    # input_dim = 68 - 12 -2 - 21
    input_dim = 68 - 12 - 2 +1 # (+1 for phase)
    output_dim = 2
    x_idx, y_idx = 1, 2
    # traj_idx = [0,2,3,5,6,8,10,11,15,16,17]
    traj_idx = [0,2,3,5,6,8,11,17]
    # traj_idx = [0,2,3,5,6,8,11,17,18,19,20,21,22,23,24,25,26,27,28,29]

    num_traj = len(traj_idx)
    data_path = "/home/srho31/Desktop/data/humanoid_traj/naive_rl_8dir_30epi_ver0208.npz"
    model_path = f"/home/srho31/Desktop/model/cosine_humanoid_{input_dim}_outputdim{output_dim}_epoch{epochs}_numtraj{num_traj}_withphase.pt"



    algo = ContrastiveLearning(data_path, model_path, input_dim, output_dim, batch_size, num_traj, traj_idx = traj_idx)
    algo.draw_figures_of_data( x_idx, y_idx )

    if os.path.exists(model_path):
        pass
    else:
        optimizer = optim.Adam(algo.model.parameters(), lr=learning_rate)
        algo.train(optimizer, epochs)
        algo.save_model()
    algo.draw_latent_figure(epochs)







