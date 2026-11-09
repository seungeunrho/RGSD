import os
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


import torch
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    def __init__(self, motions, n_frames=1):
        self.n_frames = n_frames
        self.data = []

        for motion_id, (motion, accronym) in enumerate(motions):
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
                frames.append(torch.zeros(feature_dim))
            else:
                frames.append(torch.tensor(motion[i], dtype=torch.float32))

        frames_tensor = torch.stack(frames).flip(0).reshape(-1)

        return (
            frames_tensor,                        # shape: (n_frames * feature_dim,)
            torch.tensor(motion_id, dtype=torch.long),
            torch.tensor(curr_idx, dtype=torch.long)
        )


# Function to calculate the traveled distance of a motion in latent space
def compute_traveled_distance(latent_points):
    distance = 0.0
    for i in range(1, len(latent_points)):
        # Calculate Euclidean distance between consecutive points
        dist = np.linalg.norm(latent_points[i] - latent_points[i - 1])
        distance += dist
    return distance


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
        dataset = MotionDataset(self.motions, n_frames=n_history)
        self.dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        self.dataloader_iter = iter(self.dataloader)
        self.device = device
        self.dataloader_counter = 0
        self.latent_dim = latent_dim

        self.motion_std_tensor = self._compute_std_for_each_motion(motion_tensor_lst)

    def load_model(self, model_state_dict):
        self.model.model.load_state_dict(model_state_dict)

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
            # phase = np.arange(data_length).reshape(data_length, 1) * dt
            # traj = np.concatenate((phase, capped_traj), axis=1)
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

    def add_columnwise_scaled_noise(self, tensor: torch.Tensor, col_std: torch.Tensor, noise_scale: float = 0.1) -> torch.Tensor:
        """
        Add column-wise Gaussian noise to a tensor, scaled by each column's standard deviation.

        Args:
            tensor (torch.Tensor): Input tensor of shape [N, D].
            noise_scale (float): Multiplier for the noise (relative to each column's std).

        Returns:
            torch.Tensor: Tensor with added column-wise scaled noise.
        """
        # Generate noise with shape [N, D], scaled per column
        noise = torch.randn_like(tensor[:,1:]) * col_std[:,1:] * noise_scale
        tensor[:,1:] += noise

        return tensor

    def compute_loss_for_batch(self, model, batch):
        # Unpack the batch
        points, labels, frame_indices = batch
        points = points.to(self.device)
        labels = labels.to(self.device)
        frame_indices = frame_indices.to(self.device)

        points = self.add_columnwise_scaled_noise(points, 
                                                  self.motion_std_tensor[labels],
                                                  noise_scale=self.cfg.noise_scale)


        # Forward pass
        # orig_embeddings = model(points)
        orig_embeddings = model({
            "historical_self_obs":points,
        })

        # If you still want the original triplet loss using cosine distance,
        # you *might* skip normalization here if you want Euclidean distances
        # to be meaningful. However, if you keep normalization,
        # the Euclidean norm is bounded.
        embeddings = nn.functional.normalize(orig_embeddings, p=2, dim=-1)

        # ---- Original Triplet loss (with the "positive" and "negative" sampling) ----
        anchor_indices = torch.arange(len(labels)).to(self.device)
        positive_indices = []
        negative_indices = []

        for i, label in enumerate(labels):
            pos_idx = (labels == label).nonzero(as_tuple=True)[0]
            neg_idx = (labels != label).nonzero(as_tuple=True)[0]
            if len(pos_idx) > 0 and len(neg_idx) > 0:
                # Randomly select one positive (excluding itself if you want)
                positive_indices.append(pos_idx[torch.randint(len(pos_idx), (1,))])
                negative_indices.append(neg_idx[torch.randint(len(neg_idx), (1,))])

        triplet_loss_val = torch.tensor(0.0, device=self.device)
        if positive_indices and negative_indices:
            positive_indices = torch.stack(positive_indices)
            negative_indices = torch.stack(negative_indices)

            anchors = embeddings[anchor_indices]
            positives = embeddings[positive_indices].squeeze(1)
            negatives = embeddings[negative_indices].squeeze(1)

            triplet_loss_val = self.triplet_loss(anchors, positives, negatives, margin=self.cfg.loss_margin)

        # ---- New Index-Distance Loss for positive pairs (same trajectory) ----
        # index_dist_loss_val = index_distance_loss(embeddings, labels, frame_indices)
        index_dist_loss_val = index_distance_loss_all_pairs(orig_embeddings, labels, frame_indices)

        # Combine the two losses. You can also add a weighting factor alpha if needed.
        total_loss = 1000*triplet_loss_val + 0.05*index_dist_loss_val

        return total_loss

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

    def train(self, optimizer, epochs):
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            data_continues = True
            
            if epoch%self.cfg.draw_interval==0:
                self.draw_latent_figure(epoch)
            
            
            while data_continues:
                optimizer.zero_grad()
                loss, data_continues = self.compute_loss_once(self.model)
                loss.backward()
                total_loss += loss.item()

                # Update weights
                optimizer.step()

            # Print epoch loss
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    def save_model(self):
        torch.save(self.model.model.state_dict(), self.model_path)

    # Triplet loss function
    def triplet_loss(self, anchor, positive, negative, margin=0.2):
        pos_dist = 1 - torch.cosine_similarity(anchor, positive, dim=-1)
        neg_dist = 1 - torch.cosine_similarity(anchor, negative, dim=-1)
        loss = torch.relu(pos_dist - neg_dist + margin).mean()
        # loss = torch.square(pos_dist).mean() - torch.square(neg_dist).mean()

        return loss

    def draw_latent_figure(self, model, epoch, file_name, n_frames=10):
        fig = plt.figure(figsize=(10, 6))

        if self.latent_dim== 2:
            ax = fig.add_subplot()
        else:
            ax = fig.add_subplot(111, projection='3d')

        total_distances = []
        skill_vec_lst = []

        with torch.no_grad():
            for i, (motion, motion_accronym) in enumerate(self.motions):
                motion_points = torch.tensor(motion, dtype=torch.float32).to(self.device)
                latent_points = []

                # Skip motions that are too short
                if len(motion_points) < n_frames:
                    continue

                for j in range(len(motion_points) - n_frames + 1):
                    frames = motion_points[j:j + n_frames].reshape(-1).unsqueeze(0)  # Shape: [1, n_frames, D]
                    latent = model({"historical_self_obs":frames}).cpu().numpy()
                    latent_points.append(latent)

                latent_points = np.vstack(latent_points)

                traveled_distance = compute_traveled_distance(latent_points)
                total_distances.append(traveled_distance)

                print(f"Traveled distance for Motion {i + 1}: {traveled_distance:.4f}")


                if self.latent_dim== 2:
                    ax.scatter(latent_points[:, 0], latent_points[:, 1],
                        label=f'{motion_accronym} - {traveled_distance:.1f}')
                else :
                    ax.scatter(latent_points[:, 0], latent_points[:, 1], latent_points[:, 2],
                        label=f'{motion_accronym} - {traveled_distance:.1f}')
            
                skill_vec = self.compute_latent_vec_for_motion(latent_points)
                skill_vec_lst.append(skill_vec)


        plt.legend()
        plt.title(f'Visualization at {epoch}')
        plt.grid(True)
        plt.savefig(file_name)
        plt.show()

        return ax, skill_vec_lst
    
    def compute_latent_vec_for_motion(self, latent_points):
        n_points = len(latent_points)
        idx1 = int(n_points/3)
        idx2 = int(n_points*2/3)
        idx3 = int(n_points) - 1

        skill_vec = (latent_points[idx1]+ latent_points[idx2] + latent_points[idx3])/3
        return skill_vec



def index_distance_loss_all_pairs(embeddings, labels, frame_indices):
    """
    Compute the index-distance loss for all pairs of data points
    from the same trajectory within the batch.

    embeddings: (B, latent_dim) Latent vector of each sample in the batch
    labels: (B,) Trajectory IDs for each sample
    frame_indices: (B,) Index of the frame within the trajectory
    """
    device = embeddings.device
    unique_labels = labels.unique()
    loss_val = torch.tensor(0.0, device=device)
    count = 0

    # Loop over unique trajectory labels
    for lbl in unique_labels:
        idx = torch.where(labels == lbl)[0]
        if len(idx) <= 1:
            continue  # Skip if there's only one frame with this label

        # Extract embeddings and frame indices for this trajectory
        e_label = embeddings[idx]  # (N, latent_dim)
        i_label = frame_indices[idx]  # (N,)

        # Compute pairwise Euclidean distances in latent space: (N, N)
        dist_matrix = torch.cdist(e_label, e_label, p=2)

        # Compute pairwise absolute differences in frame indices: (N, N)
        i_label_2d = i_label.unsqueeze(1)  # Shape: (N, 1)
        index_diff_matrix = (i_label_2d - i_label_2d.T).abs().float()

        # Compute MSE between distance and index difference
        mse_label = (dist_matrix - index_diff_matrix).pow(2).mean()

        # Accumulate loss
        loss_val += mse_label
        count += 1

    # Normalize by the number of trajectories in the batch
    if count > 0:
        loss_val /= count

    return loss_val

def index_distance_loss(embeddings, labels, frame_indices):
    """
    embeddings: (B, latent_dim)  latent vector of each sample in the batch
    labels:     (B,)            trajectory IDs for each sample
    frame_indices: (B,)         index of the frame within the trajectory
    """
    device = embeddings.device
    unique_labels = labels.unique()

    loss_val = torch.tensor(0.0, device=device)
    count = 0

    # For each trajectory label in this batch,
    # compute pairwise distance of embeddings and compare to index difference.
    for lbl in unique_labels:
        idx = torch.where(labels == lbl)[0]
        if len(idx) <= 1:
            continue  # If there's only one frame with this label, skip

        # embeddings of this label, shape: (N, latent_dim)
        e_label = embeddings[idx]
        # indices of the frames in this trajectory, shape: (N,)
        i_label = frame_indices[idx]

        # Pairwise Euclidean distances in the latent space: (N, N)
        dist_matrix = torch.cdist(e_label, e_label, p=2)

        # Pairwise absolute differences in frame indices: (N, N)
        i_label_2d = i_label.unsqueeze(1)  # shape: (N, 1)
        index_diff_matrix = (i_label_2d - i_label_2d.T).abs().float()

        # We want dist_matrix ~ index_diff_matrix
        # Use MSE: (dist_matrix - index_diff_matrix)^2
        mse_label = (dist_matrix - index_diff_matrix).pow(2).mean()

        loss_val += mse_label
        count += 1

    if count > 0:
        loss_val = loss_val / count  # Average over the label groups

    return loss_val

if __name__ == '__main__':
    set_seed(43)


    epochs = 1100
    batch_size = 256
    learning_rate = 0.001

    # data_path = "/home/srho31/Desktop/data/cube_east_traj/lgsd_east.npz"
    # model_path = f"/home/srho31/Desktop/model/cosine_lgsd_east_{input_dim}_outputdim{output_dim}_epoch{epochs}.pt"
    # x_idx_, y_idx = 4,5
    # num_traj = 10

    # input_dim = 39
    # output_dim = 2
    # data_path = "/home/srho31/Desktop/data/ant_traj/lsd_walk_8dir.npz"
    # model_path = f"/home/srho31/Desktop/model/cosine_ant_{input_dim}_outputdim{output_dim}_epoch{epochs}.pt"
    # x_idx, y_idx = 0,1
    # num_traj = 10

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







