import sys
sys.path.append('../')
sys.path.append("../submodules")
sys.path.append('../submodules/RoMa')

from matplotlib import pyplot as plt
from PIL import Image
import torch
import numpy as np

#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist

import torch.nn.functional as F
from romatch import roma_outdoor, roma_indoor
from .model_utils import load_roma_model_with_custom_cache
from utils.sh_utils import RGB2SH
from romatch.utils import get_tuple_transform_ops

import time
from collections import defaultdict
from tqdm import tqdm


def pairwise_distances(matrix):
    """
    Computes the pairwise Euclidean distances between all vectors in the input matrix.

    Args:
        matrix (torch.Tensor): Input matrix of shape [N, D], where N is the number of vectors and D is the dimensionality.

    Returns:
        torch.Tensor: Pairwise distance matrix of shape [N, N].
    """
    # Compute squared pairwise distances
    squared_diff = torch.cdist(matrix, matrix, p=2)
    return squared_diff


def k_closest_vectors(matrix, k):
    """
    Finds the k-closest vectors for each vector in the input matrix based on Euclidean distance.

    Args:
        matrix (torch.Tensor): Input matrix of shape [N, D], where N is the number of vectors and D is the dimensionality.
        k (int): Number of closest vectors to return for each vector.

    Returns:
        torch.Tensor: Indices of the k-closest vectors for each vector, excluding the vector itself.
    """
    # Compute pairwise distances
    distances = pairwise_distances(matrix)

    # For each vector, sort distances and get the indices of the k-closest vectors (excluding itself)
    # Set diagonal distances to infinity to exclude the vector itself from the nearest neighbors
    distances.fill_diagonal_(float('inf'))

    # Get the indices of the k smallest distances (k-closest vectors)
    _, indices = torch.topk(distances, k, largest=False, dim=1)

    return indices


def select_cameras_kmeans(cameras, K):
    """
    Selects K cameras from a set using K-means clustering.

    Args:
        cameras: NumPy array of shape (N, 16), representing N cameras with their 4x4 homogeneous matrices flattened.
        K: Number of clusters (cameras to select).

    Returns:
        selected_indices: List of indices of the cameras closest to the cluster centers.
    """
    # Ensure input is a NumPy array
    if not isinstance(cameras, np.ndarray):
        cameras = np.asarray(cameras)

    if cameras.shape[1] != 16:
        raise ValueError("Each camera must have 16 values corresponding to a flattened 4x4 matrix.")

    # Perform K-means clustering
    cluster_centers, _ = kmeans(cameras, K)

    # Assign each camera to a cluster and find distances to cluster centers
    cluster_assignments, _ = vq(cameras, cluster_centers)

    # Find the camera nearest to each cluster center
    selected_indices = []
    for k in range(K):
        cluster_members = cameras[cluster_assignments == k]
        distances = cdist([cluster_centers[k]], cluster_members)[0]
        nearest_camera_idx = np.where(cluster_assignments == k)[0][np.argmin(distances)]
        selected_indices.append(nearest_camera_idx)

    return selected_indices


def compute_warp_and_confidence(viewpoint_cam1, viewpoint_cam2, roma_model, device="cuda", verbose=False, output_dict={}):
    """
    Computes the warp and confidence between two viewpoint cameras using the roma_model.

    Args:
        viewpoint_cam1: Source viewpoint camera.
        viewpoint_cam2: Target viewpoint camera.
        roma_model: Pre-trained Roma model for correspondence matching.
        device: Device to run the computation on.
        verbose: If True, displays the images.

    Returns:
        certainty: Confidence tensor.
        warp: Warp tensor.
        imB: Processed image B as numpy array.
    """
    # Prepare images
    imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
    imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))

    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        cax1 = ax[0].imshow(imA)
        ax[0].set_title("Image 1")
        cax2 = ax[1].imshow(imB)
        ax[1].set_title("Image 2")
        fig.colorbar(cax1, ax=ax[0])
        fig.colorbar(cax2, ax=ax[1])
    
        for axis in ax:
            axis.axis('off')
        # Save the figure into the dictionary
        output_dict[f'image_pair'] = fig
   
    # Transform images
    ws, hs = roma_model.w_resized, roma_model.h_resized
    test_transform = get_tuple_transform_ops(resize=(hs, ws), normalize=True)
    im_A, im_B = test_transform((imA, imB))
    batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}

    # Forward pass through Roma model
    corresps = roma_model.forward(batch) if not roma_model.symmetric else roma_model.forward_symmetric(batch)
    finest_scale = 1
    hs, ws = roma_model.upsample_res if roma_model.upsample_preds else (hs, ws)

    # Process certainty and warp
    certainty = corresps[finest_scale]["certainty"]
    im_A_to_im_B = corresps[finest_scale]["flow"]
    if roma_model.attenuate_cert:
        low_res_certainty = F.interpolate(
            corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
        )
        certainty -= 0.5 * low_res_certainty * (low_res_certainty < 0)

    # Upsample predictions if needed
    if roma_model.upsample_preds:
        im_A_to_im_B = F.interpolate(
            im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
        )
        certainty = F.interpolate(
            certainty, size=(hs, ws), align_corners=False, mode="bilinear"
        )

    # Convert predictions to final format
    im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
    im_A_coords = torch.stack(torch.meshgrid(
        torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
        torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
        indexing='ij'
    ), dim=0).permute(1, 2, 0).unsqueeze(0).expand(im_A_to_im_B.size(0), -1, -1, -1)

    warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
    certainty = certainty.sigmoid()

    return certainty[0, 0], warp[0], np.array(imB)


def resize_batch(tensors_3d, tensors_4d, target_shape):
    """
    Resizes a batch of tensors with shapes [B, H, W] and [B, H, W, 4] to the target spatial dimensions.

    Args:
        tensors_3d: Tensor of shape [B, H, W].
        tensors_4d: Tensor of shape [B, H, W, 4].
        target_shape: Tuple (target_H, target_W) specifying the target spatial dimensions.

    Returns:
        resized_tensors_3d: Tensor of shape [B, target_H, target_W].
        resized_tensors_4d: Tensor of shape [B, target_H, target_W, 4].
    """
    target_H, target_W = target_shape

    # Resize [B, H, W] tensor
    resized_tensors_3d = F.interpolate(
        tensors_3d.unsqueeze(1), size=(target_H, target_W), mode="bilinear", align_corners=False
    ).squeeze(1)

    # Resize [B, H, W, 4] tensor
    B, _, _, C = tensors_4d.shape
    resized_tensors_4d = F.interpolate(
        tensors_4d.permute(0, 3, 1, 2), size=(target_H, target_W), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)

    return resized_tensors_3d, resized_tensors_4d


def aggregate_confidences_and_warps(viewpoint_stack, closest_indices, roma_model, source_idx, verbose=False, output_dict={}):
    """
    Aggregates confidences and warps by iterating over the nearest neighbors of the source viewpoint.

    Args:
        viewpoint_stack: Stack of viewpoint cameras.
        closest_indices: Indices of the nearest neighbors for each viewpoint.
        roma_model: Pre-trained Roma model.
        source_idx: Index of the source viewpoint.
        verbose: If True, displays intermediate results.

    Returns:
        certainties_max: Aggregated maximum confidences.
        warps_max: Aggregated warps corresponding to maximum confidences.
        certainties_max_idcs: Pixel-wise index of the image  from which we taken the best matching.
        imB_compound: List of the neighboring images.
    """
    certainties_all, warps_all, imB_compound = [], [], []

    for nn in tqdm(closest_indices[source_idx]):

        viewpoint_cam1 = viewpoint_stack[source_idx]
        viewpoint_cam2 = viewpoint_stack[nn]

        certainty, warp, imB = compute_warp_and_confidence(viewpoint_cam1, viewpoint_cam2, roma_model, verbose=verbose, output_dict=output_dict)
        certainties_all.append(certainty)
        warps_all.append(warp)
        imB_compound.append(imB)

    certainties_all = torch.stack(certainties_all, dim=0)
    target_shape = imB_compound[0].shape[:2]
    if verbose: 
        print("certainties_all.shape:", certainties_all.shape)
        print("torch.stack(warps_all, dim=0).shape:", torch.stack(warps_all, dim=0).shape)
        print("target_shape:", target_shape)        

    certainties_all_resized, warps_all_resized = resize_batch(certainties_all,
                                                              torch.stack(warps_all, dim=0),
                                                              target_shape
                                                              )

    if verbose:
        print("warps_all_resized.shape:", warps_all_resized.shape)
        for n, cert in enumerate(certainties_all):
            fig, ax = plt.subplots()
            cax = ax.imshow(cert.cpu().numpy(), cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise Confidence")
            output_dict[f'certainty_{n}'] = fig

        for n, warp in enumerate(warps_all):
            fig, ax = plt.subplots()
            cax = ax.imshow(warp.cpu().numpy()[:, :, :3], cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise warp")
            output_dict[f'warp_resized_{n}'] = fig

        for n, cert in enumerate(certainties_all_resized):
            fig, ax = plt.subplots()
            cax = ax.imshow(cert.cpu().numpy(), cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise Confidence resized")
            output_dict[f'certainty_resized_{n}'] = fig

        for n, warp in enumerate(warps_all_resized):
            fig, ax = plt.subplots()
            cax = ax.imshow(warp.cpu().numpy()[:, :, :3], cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise warp resized")
            output_dict[f'warp_resized_{n}'] = fig

    certainties_max, certainties_max_idcs = torch.max(certainties_all_resized, dim=0)
    H, W = certainties_max.shape

    warps_max = warps_all_resized[certainties_max_idcs, torch.arange(H).unsqueeze(1), torch.arange(W)]

    imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    imA = np.clip(imA * 255, 0, 255).astype(np.uint8)

    return certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all_resized, warps_all_resized



def extract_keypoints_and_colors(imA, imB_compound, certainties_max, certainties_max_idcs, matches, roma_model,
                                 verbose=False, output_dict={}):
    """
    Extracts keypoints and corresponding colors from the source image (imA) and multiple target images (imB_compound).

    Args:
        imA: Source image as a NumPy array (H_A, W_A, C).
        imB_compound: List of target images as NumPy arrays [(H_B, W_B, C), ...].
        certainties_max: Tensor of pixel-wise maximum confidences.
        certainties_max_idcs: Tensor of pixel-wise indices for the best matches.
        matches: Matches in normalized coordinates.
        roma_model: Roma model instance for keypoint operations.
        verbose: if to show intermediate outputs and visualize results

    Returns:
        kptsA_np: Keypoints in imA in normalized coordinates.
        kptsB_np: Keypoints in imB in normalized coordinates.
        kptsA_color: Colors of keypoints in imA.
        kptsB_color: Colors of keypoints in imB based on certainties_max_idcs.
    """
    H_A, W_A, _ = imA.shape
    H, W = certainties_max.shape

    # Convert matches to pixel coordinates
    kptsA, kptsB = roma_model.to_pixel_coordinates(
        matches, W_A, H_A, H, W  # W, H
    )

    kptsA_np = kptsA.detach().cpu().numpy()
    kptsB_np = kptsB.detach().cpu().numpy()
    kptsA_np = kptsA_np[:, [1, 0]]

    if verbose:
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(imA)
        ax.set_title("Reference image, imA")
        output_dict[f'reference_image'] = fig

        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(imB_compound[0])
        ax.set_title("Image to compare to image, imB_compound")
        output_dict[f'imB_compound'] = fig
    
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imA))
        cax = ax.scatter(kptsA_np[:, 0], H_A - kptsA_np[:, 1], s=.03)
        ax.set_title("Keypoints in imA")
        ax.set_xlim(0, W_A)
        ax.set_ylim(0, H_A)
        output_dict[f'kptsA'] = fig

        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imB_compound[0]))
        cax = ax.scatter(kptsB_np[:, 0], H_A - kptsB_np[:, 1], s=.03)
        ax.set_title("Keypoints in imB")
        ax.set_xlim(0, W_A)
        ax.set_ylim(0, H_A)
        output_dict[f'kptsB'] = fig

    # Keypoints are in format (row, column) so the first value is alwain in range [0;height] and second is in range[0;width]

    kptsA_np = kptsA.detach().cpu().numpy()
    kptsB_np = kptsB.detach().cpu().numpy()

    # Extract colors for keypoints in imA (vectorized)
    # New experimental version
    kptsA_x = np.round(kptsA_np[:, 0] / 1.).astype(int)
    kptsA_y = np.round(kptsA_np[:, 1] / 1.).astype(int)
    kptsA_color = imA[np.clip(kptsA_x, 0, H - 1), np.clip(kptsA_y, 0, W - 1)]
   
    # Create a composite image from imB_compound
    imB_compound_np = np.stack(imB_compound, axis=0)
    H_B, W_B, _ = imB_compound[0].shape

    # Extract colors for keypoints in imB using certainties_max_idcs
    imB_np = imB_compound_np[
            certainties_max_idcs.detach().cpu().numpy(),
            np.arange(H).reshape(-1, 1),
            np.arange(W)
        ]
    
    if verbose:
        print("imB_np.shape:", imB_np.shape)
        print("imB_np:", imB_np)
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imB_np))
        cax = ax.scatter(kptsB_np[:, 0], H_A - kptsB_np[:, 1], s=.03)
        ax.set_title("np.flipud(imB_np[0]")
        ax.set_xlim(0, W_A)
        ax.set_ylim(0, H_A)
        output_dict[f'np.flipud(imB_np[0]'] = fig


    kptsB_x = np.round(kptsB_np[:, 0]).astype(int)
    kptsB_y = np.round(kptsB_np[:, 1]).astype(int)

    certainties_max_idcs_np = certainties_max_idcs.detach().cpu().numpy()
    kptsB_proj_matrices_idx = certainties_max_idcs_np[np.clip(kptsA_x, 0, H - 1), np.clip(kptsA_y, 0, W - 1)]
    kptsB_color = imB_compound_np[kptsB_proj_matrices_idx, np.clip(kptsB_y, 0, H - 1), np.clip(kptsB_x, 0, W - 1)]

    # Normalize keypoints in both images
    kptsA_np[:, 0] = kptsA_np[:, 0] / H * 2.0 - 1.0
    kptsA_np[:, 1] = kptsA_np[:, 1] / W * 2.0 - 1.0
    kptsB_np[:, 0] = kptsB_np[:, 0] / W_B * 2.0 - 1.0
    kptsB_np[:, 1] = kptsB_np[:, 1] / H_B * 2.0 - 1.0

    return kptsA_np[:, [1, 0]], kptsB_np, kptsB_proj_matrices_idx, kptsA_color, kptsB_color

def prepare_tensor(input_array, device):
    """
    Converts an input array to a torch tensor, clones it, and detaches it for safe computation.
    Args:
        input_array (array-like): The input array to convert.
        device (str or torch.device): The device to move the tensor to.
    Returns:
        torch.Tensor: A detached tensor clone of the input array on the specified device.
    """
    if not isinstance(input_array, torch.Tensor):
        return torch.tensor(input_array, dtype=torch.float32).to(device).clone().detach()
    return input_array.clone().detach().to(device).to(torch.float32)

def triangulate_points(P1, P2, k1_x, k1_y, k2_x, k2_y, device="cuda"):
    """
    Solves for a batch of 3D points given batches of projection matrices and corresponding image points.

    Parameters:
    - P1, P2: Tensors of projection matrices of size (batch_size, 4, 4) or (4, 4)
    - k1_x, k1_y: Tensors of shape (batch_size,)
    - k2_x, k2_y: Tensors of shape (batch_size,)

    Returns:
    - X: A tensor containing the 3D homogeneous coordinates, shape (batch_size, 4)
    """
    EPS = 1e-4
    # Ensure inputs are tensors

    P1 = prepare_tensor(P1, device)
    P2 = prepare_tensor(P2, device)
    k1_x = prepare_tensor(k1_x, device)
    k1_y = prepare_tensor(k1_y, device)
    k2_x = prepare_tensor(k2_x, device)
    k2_y =  prepare_tensor(k2_y, device)
    batch_size = k1_x.shape[0]

    # Expand P1 and P2 if they are not batched
    if P1.ndim == 2:
        P1 = P1.unsqueeze(0).expand(batch_size, -1, -1)
    if P2.ndim == 2:
        P2 = P2.unsqueeze(0).expand(batch_size, -1, -1)

    # Extract columns from P1 and P2
    P1_0 = P1[:, :, 0]  # Shape: (batch_size, 4)
    P1_1 = P1[:, :, 1]
    P1_2 = P1[:, :, 2]

    P2_0 = P2[:, :, 0]
    P2_1 = P2[:, :, 1]
    P2_2 = P2[:, :, 2]

    # Reshape kx and ky to (batch_size, 1)
    k1_x = k1_x.view(-1, 1)
    k1_y = k1_y.view(-1, 1)
    k2_x = k2_x.view(-1, 1)
    k2_y = k2_y.view(-1, 1)

    # Construct the equations for each batch
    # For camera 1
    A1 = P1_0 - k1_x * P1_2  # Shape: (batch_size, 4)
    A2 = P1_1 - k1_y * P1_2
    # For camera 2
    A3 = P2_0 - k2_x * P2_2
    A4 = P2_1 - k2_y * P2_2

    # Stack the equations
    A = torch.stack([A1, A2, A3, A4], dim=1)  # Shape: (batch_size, 4, 4)

    # Right-hand side (constants)
    b = -A[:, :, 3]  # Shape: (batch_size, 4)
    A_reduced = A[:, :, :3]  # Coefficients of x, y, z

    # Solve using torch.linalg.lstsq (supports batching)
    X_xyz = torch.linalg.lstsq(A_reduced, b.unsqueeze(2)).solution.squeeze(2)  # Shape: (batch_size, 3)

    # Append 1 to get homogeneous coordinates
    ones = torch.ones((batch_size, 1), dtype=torch.float32, device=X_xyz.device)
    X = torch.cat([X_xyz, ones], dim=1)  # Shape: (batch_size, 4)

    # Now compute the errors of projections.
    seeked_splats_proj1 = (X.unsqueeze(1) @ P1).squeeze(1)
    seeked_splats_proj1 = seeked_splats_proj1 / (EPS + seeked_splats_proj1[:, [3]])
    seeked_splats_proj2 = (X.unsqueeze(1) @ P2).squeeze(1)
    seeked_splats_proj2 = seeked_splats_proj2 / (EPS + seeked_splats_proj2[:, [3]])
    proj1_target = torch.concat([k1_x, k1_y], dim=1)
    proj2_target = torch.concat([k2_x, k2_y], dim=1)
    errors_proj1 = torch.abs(seeked_splats_proj1[:, :2] - proj1_target).sum(1).detach().cpu().numpy()
    errors_proj2 = torch.abs(seeked_splats_proj2[:, :2] - proj2_target).sum(1).detach().cpu().numpy()

    return X, errors_proj1, errors_proj2



def select_best_keypoints(
        NNs_triangulated_points, NNs_errors_proj1, NNs_errors_proj2, device="cuda"):
    """
    From all the points fitted to  keypoints and corresponding colors from the source image (imA) and multiple target images (imB_compound).

    Args:
        NNs_triangulated_points:  torch tensor with keypoints coordinates (num_nns, num_points, dim). dim can be arbitrary,
            usually 3 or 4(for homogeneous representation).
        NNs_errors_proj1:  numpy array with projection error of the estimated keypoint on the reference frame (num_nns, num_points).
        NNs_errors_proj2:  numpy array with projection error of the estimated keypoint on the neighbor frame (num_nns, num_points).
    Returns:
        selected_keypoints: keypoints with the best score.
    """

    NNs_errors_proj = np.maximum(NNs_errors_proj1, NNs_errors_proj2)

    # Convert indices to PyTorch tensor
    indices = torch.from_numpy(np.argmin(NNs_errors_proj, axis=0)).long().to(device)

    # Create index tensor for the second dimension
    n_indices = torch.arange(NNs_triangulated_points.shape[1]).long().to(device)

    # Use advanced indexing to select elements
    NNs_triangulated_points_selected = NNs_triangulated_points[indices, n_indices, :]  # Shape: [N, k]

    return NNs_triangulated_points_selected, np.min(NNs_errors_proj, axis=0)



def init_gaussians_with_corr(gaussians, scene, cfg, device, verbose = False, roma_model=None):
    """
    For a given input gaussians and a scene we instantiate a RoMa model(change to indoors if necessary) and process scene
    training frames to extract correspondences. Those are used to initialize gaussians
    Args:
        gaussians: object gaussians of the class GaussianModel that we need to enrich with gaussians.
        scene: object of the Scene class.
        cfg: configuration. Use init_wC
    Returns:
        gaussians: inplace transforms object gaussians of the class GaussianModel.

    """
    if roma_model is None:
        if cfg.roma_model == "indoors":
            roma_model = load_roma_model_with_custom_cache(model_type="indoor", device=device)
        else:
            roma_model = load_roma_model_with_custom_cache(model_type="outdoor", device=device)
        roma_model.upsample_preds = False
        roma_model.symmetric = False
    M = cfg.matches_per_ref
    upper_thresh = roma_model.sample_thresh
    scaling_factor = cfg.scaling_factor
    expansion_factor = 1
    keypoint_fit_error_tolerance = cfg.proj_err_tolerance
    visualizations = {}
    viewpoint_stack = scene.getTrainCameras().copy()
    NUM_REFERENCE_FRAMES = min(cfg.num_refs, len(viewpoint_stack))
    NUM_NNS_PER_REFERENCE = min(cfg.nns_per_ref , len(viewpoint_stack))
    # Select cameras using K-means
    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)

    selected_indices = select_cameras_kmeans(cameras=viewpoint_cam_all.detach().cpu().numpy(), K=NUM_REFERENCE_FRAMES)
    selected_indices = sorted(selected_indices)
   

    # Find the k-closest vectors for each vector
    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)
    closest_indices = k_closest_vectors(viewpoint_cam_all, NUM_NNS_PER_REFERENCE)
    if verbose: print("Indices of k-closest vectors for each vector:\n", closest_indices)

    closest_indices_selected = closest_indices[:, :].detach().cpu().numpy()

    all_new_xyz = []
    all_new_features_dc = []
    all_new_features_rest = []
    all_new_opacities = []
    all_new_scaling = []
    all_new_rotation = []

    # Run roma_model.match once to kinda initialize the model
    with torch.no_grad():
        viewpoint_cam1 = viewpoint_stack[0]
        viewpoint_cam2 = viewpoint_stack[1]
        imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
        imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))
        warp, certainty_warp = roma_model.match(imA, imB, device=device)
        print("Once run full roma_model.match warp.shape:", warp.shape)
        print("Once run full roma_model.match certainty_warp.shape:", certainty_warp.shape)
        del warp, certainty_warp
        torch.cuda.empty_cache()

    for source_idx in tqdm(sorted(selected_indices)):
        # 1. Compute keypoints and warping for all the neigboring views
        with torch.no_grad():
            # Call the aggregation function to get imA and imB_compound
            certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all, warps_all = aggregate_confidences_and_warps(
                viewpoint_stack=viewpoint_stack,
                closest_indices=closest_indices_selected,
                roma_model=roma_model,
                source_idx=source_idx,
                verbose=verbose, output_dict=visualizations
            )


        # Triangulate keypoints
        with torch.no_grad():
            matches = warps_max
            certainty = certainties_max
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
            matches, certainty = (
                matches.reshape(-1, 4),
                certainty.reshape(-1),
            )

            # Select based on certainty elements with high confidence. These are basically all of
            # kptsA_np.
            good_samples = torch.multinomial(certainty,
                                             num_samples=min(expansion_factor * M, len(certainty)),
                                             replacement=False)

        certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all, warps_all
        reference_image_dict = {
            "ref_image": imA,
            "NNs_images": imB_compound,
            "certainties_all": certainties_all,
            "warps_all": warps_all,
            "triangulated_points": [],
            "triangulated_points_errors_proj1": [],
            "triangulated_points_errors_proj2": []

        }
        with torch.no_grad():
            for NN_idx in tqdm(range(len(warps_all))):
                matches_NN = warps_all[NN_idx].reshape(-1, 4)[good_samples]

                # Extract keypoints and colors
                kptsA_np, kptsB_np, kptsB_proj_matrices_idcs, kptsA_color, kptsB_color = extract_keypoints_and_colors(
                    imA, imB_compound, certainties_max, certainties_max_idcs, matches_NN, roma_model
                )

                proj_matrices_A = viewpoint_stack[source_idx].full_proj_transform
                proj_matrices_B = viewpoint_stack[closest_indices_selected[source_idx, NN_idx]].full_proj_transform
                triangulated_points, triangulated_points_errors_proj1, triangulated_points_errors_proj2 = triangulate_points(
                    P1=torch.stack([proj_matrices_A] * M, axis=0),
                    P2=torch.stack([proj_matrices_B] * M, axis=0),
                    k1_x=kptsA_np[:M, 0], k1_y=kptsA_np[:M, 1],
                    k2_x=kptsB_np[:M, 0], k2_y=kptsB_np[:M, 1])

                reference_image_dict["triangulated_points"].append(triangulated_points)
                reference_image_dict["triangulated_points_errors_proj1"].append(triangulated_points_errors_proj1)
                reference_image_dict["triangulated_points_errors_proj2"].append(triangulated_points_errors_proj2)

        with torch.no_grad():
            NNs_triangulated_points_selected, NNs_triangulated_points_selected_proj_errors = select_best_keypoints(
                NNs_triangulated_points=torch.stack(reference_image_dict["triangulated_points"], dim=0),
                NNs_errors_proj1=np.stack(reference_image_dict["triangulated_points_errors_proj1"], axis=0),
                NNs_errors_proj2=np.stack(reference_image_dict["triangulated_points_errors_proj2"], axis=0))

        # 4. Save as gaussians
        viewpoint_cam1 = viewpoint_stack[source_idx]
        N = len(NNs_triangulated_points_selected)
        with torch.no_grad():
            new_xyz = NNs_triangulated_points_selected[:, :-1]
            all_new_xyz.append(new_xyz)  # seeked_splats
            all_new_features_dc.append(RGB2SH(torch.tensor(kptsA_color.astype(np.float32) / 255.)).unsqueeze(1))
            all_new_features_rest.append(torch.stack([gaussians._features_rest[-1].clone().detach() * 0.] * N, dim=0))
            # new version that sets points with large error invisible
            # TODO: remove those points instead. However it doesn't affect the performance.
            mask_bad_points = torch.tensor(
                NNs_triangulated_points_selected_proj_errors > keypoint_fit_error_tolerance,
                dtype=torch.float32).unsqueeze(1).to(device)
            all_new_opacities.append(torch.stack([gaussians._opacity[-1].clone().detach()] * N, dim=0) * 0. - mask_bad_points * (1e1))

            dist_points_to_cam1 = torch.linalg.norm(viewpoint_cam1.camera_center.clone().detach() - new_xyz,
                                                    dim=1, ord=2)
            #all_new_scaling.append(torch.log(((dist_points_to_cam1) / 1. * scaling_factor).unsqueeze(1).repeat(1, 3)))
            all_new_scaling.append(gaussians.scaling_inverse_activation((dist_points_to_cam1 * scaling_factor).unsqueeze(1).repeat(1, 3)))
            all_new_rotation.append(torch.stack([gaussians._rotation[-1].clone().detach()] * N, dim=0))

    all_new_xyz = torch.cat(all_new_xyz, dim=0) 
    all_new_features_dc = torch.cat(all_new_features_dc, dim=0)
    new_tmp_radii = torch.zeros(all_new_xyz.shape[0])
    prune_mask = torch.ones(all_new_xyz.shape[0], dtype=torch.bool)
    
    gaussians.densification_postfix(all_new_xyz[prune_mask].to(device),
                                    all_new_features_dc[prune_mask].to(device),
                                    torch.cat(all_new_features_rest, dim=0)[prune_mask].to(device),
                                    torch.cat(all_new_opacities, dim=0)[prune_mask].to(device),
                                    torch.cat(all_new_scaling, dim=0)[prune_mask].to(device),
                                    torch.cat(all_new_rotation, dim=0)[prune_mask].to(device),
                                    new_tmp_radii[prune_mask].to(device))
    
    return viewpoint_stack, closest_indices_selected, visualizations



def extract_keypoints_and_colors_single(imA, imB, matches, roma_model, verbose=False, output_dict={}):
    """
    Extracts keypoints and corresponding colors from a source image (imA) and a single target image (imB).

    Args:
        imA: Source image as a NumPy array (H_A, W_A, C).
        imB: Target image as a NumPy array (H_B, W_B, C).
        matches: Matches in normalized coordinates (torch.Tensor).
        roma_model: Roma model instance for keypoint operations.
        verbose: If True, outputs intermediate visualizations.
    Returns:
        kptsA_np: Keypoints in imA (normalized).
        kptsB_np: Keypoints in imB (normalized).
        kptsA_color: Colors of keypoints in imA.
        kptsB_color: Colors of keypoints in imB.
    """
    H_A, W_A, _ = imA.shape
    H_B, W_B, _ = imB.shape

    # Convert matches to pixel coordinates
    # Matches format: (B, 4) = (x1_norm, y1_norm, x2_norm, y2_norm)
    kptsA = matches[:, :2]  # [N, 2]
    kptsB = matches[:, 2:]  # [N, 2]

    # Scale normalized coordinates [-1,1] to pixel coordinates
    kptsA_pix = torch.zeros_like(kptsA)
    kptsB_pix = torch.zeros_like(kptsB)

    # Important! [Normalized to pixel space]
    kptsA_pix[:, 0] = (kptsA[:, 0] + 1) * (W_A - 1) / 2
    kptsA_pix[:, 1] = (kptsA[:, 1] + 1) * (H_A - 1) / 2

    kptsB_pix[:, 0] = (kptsB[:, 0] + 1) * (W_B - 1) / 2
    kptsB_pix[:, 1] = (kptsB[:, 1] + 1) * (H_B - 1) / 2

    kptsA_np = kptsA_pix.detach().cpu().numpy()
    kptsB_np = kptsB_pix.detach().cpu().numpy()

    # Extract colors
    kptsA_x = np.round(kptsA_np[:, 0]).astype(int)
    kptsA_y = np.round(kptsA_np[:, 1]).astype(int)
    kptsB_x = np.round(kptsB_np[:, 0]).astype(int)
    kptsB_y = np.round(kptsB_np[:, 1]).astype(int)

    kptsA_color = imA[np.clip(kptsA_y, 0, H_A-1), np.clip(kptsA_x, 0, W_A-1)]
    kptsB_color = imB[np.clip(kptsB_y, 0, H_B-1), np.clip(kptsB_x, 0, W_B-1)]

    # Normalize keypoints into [-1, 1] for downstream triangulation
    kptsA_np_norm = np.zeros_like(kptsA_np)
    kptsB_np_norm = np.zeros_like(kptsB_np)

    kptsA_np_norm[:, 0] = kptsA_np[:, 0] / (W_A - 1) * 2.0 - 1.0
    kptsA_np_norm[:, 1] = kptsA_np[:, 1] / (H_A - 1) * 2.0 - 1.0

    kptsB_np_norm[:, 0] = kptsB_np[:, 0] / (W_B - 1) * 2.0 - 1.0
    kptsB_np_norm[:, 1] = kptsB_np[:, 1] / (H_B - 1) * 2.0 - 1.0

    return kptsA_np_norm, kptsB_np_norm, kptsA_color, kptsB_color



def init_gaussians_with_corr_fast(gaussians, scene, cfg, device, verbose=False, roma_model=None):
    timings = defaultdict(list)

    if roma_model is None:
        if cfg.roma_model == "indoors":
            roma_model = load_roma_model_with_custom_cache(model_type="indoor", device=device)
        else:
            roma_model = load_roma_model_with_custom_cache(model_type="outdoor", device=device)
        roma_model.upsample_preds = False
        roma_model.symmetric = False

    M = cfg.matches_per_ref
    upper_thresh = roma_model.sample_thresh
    scaling_factor = cfg.scaling_factor
    expansion_factor = 1
    keypoint_fit_error_tolerance = cfg.proj_err_tolerance
    visualizations = {}
    viewpoint_stack = scene.getTrainCameras().copy()
    NUM_REFERENCE_FRAMES = min(cfg.num_refs, len(viewpoint_stack))
    NUM_NNS_PER_REFERENCE = 1  # Only ONE neighbor now!

    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)

    selected_indices = select_cameras_kmeans(cameras=viewpoint_cam_all.detach().cpu().numpy(), K=NUM_REFERENCE_FRAMES)
    selected_indices = sorted(selected_indices)

    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)
    closest_indices = k_closest_vectors(viewpoint_cam_all, NUM_NNS_PER_REFERENCE)
    closest_indices_selected = closest_indices[:, :].detach().cpu().numpy()

    all_new_xyz = []
    all_new_features_dc = []
    all_new_features_rest = []
    all_new_opacities = []
    all_new_scaling = []
    all_new_rotation = []

    # Dummy first pass to initialize model
    with torch.no_grad():
        viewpoint_cam1 = viewpoint_stack[0]
        viewpoint_cam2 = viewpoint_stack[1]
        imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
        imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))
        warp, certainty_warp = roma_model.match(imA, imB, device=device)
        del warp, certainty_warp
        torch.cuda.empty_cache()

    # Main Loop over source_idx
    for source_idx in tqdm(sorted(selected_indices), desc="Profiling source frames"):

        # =================== Step 1: Compute Warp and Certainty ===================
        start = time.time()
        viewpoint_cam1 = viewpoint_stack[source_idx]
        NNs=closest_indices_selected.shape[1]
        viewpoint_cam2 = viewpoint_stack[closest_indices_selected[source_idx, np.random.randint(NNs)]]
        imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
        imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))
        warp, certainty_warp = roma_model.match(imA, imB, device=device)

        certainties_max = certainty_warp  # New manual sampling
        timings['aggregation_warp_certainty'].append(time.time() - start)

        # =================== Step 2: Good Samples Selection ===================
        start = time.time()
        certainty = certainties_max.reshape(-1).clone()
        certainty[certainty > upper_thresh] = 1
        good_samples = torch.multinomial(certainty, num_samples=min(expansion_factor * M, len(certainty)), replacement=False)
        timings['good_samples_selection'].append(time.time() - start)

        # =================== Step 3: Triangulate Keypoints ===================
        reference_image_dict = {
            "triangulated_points": [],
            "triangulated_points_errors_proj1": [],
            "triangulated_points_errors_proj2": []
        }

        start = time.time()
        matches_NN = warp.reshape(-1, 4)[good_samples]

        # Convert matches to pixel coordinates
        kptsA_np, kptsB_np, kptsA_color, kptsB_color = extract_keypoints_and_colors_single(
            np.array(imA).astype(np.uint8), 
            np.array(imB).astype(np.uint8), 
            matches_NN, 
            roma_model
        )

        proj_matrices_A = viewpoint_stack[source_idx].full_proj_transform
        proj_matrices_B = viewpoint_stack[closest_indices_selected[source_idx, 0]].full_proj_transform

        triangulated_points, triangulated_points_errors_proj1, triangulated_points_errors_proj2 = triangulate_points(
            P1=torch.stack([proj_matrices_A] * M, axis=0),
            P2=torch.stack([proj_matrices_B] * M, axis=0),
            k1_x=kptsA_np[:M, 0], k1_y=kptsA_np[:M, 1],
            k2_x=kptsB_np[:M, 0], k2_y=kptsB_np[:M, 1])

        reference_image_dict["triangulated_points"].append(triangulated_points)
        reference_image_dict["triangulated_points_errors_proj1"].append(triangulated_points_errors_proj1)
        reference_image_dict["triangulated_points_errors_proj2"].append(triangulated_points_errors_proj2)
        timings['triangulation_per_NN'].append(time.time() - start)

        # =================== Step 4: Select Best Triangulated Points ===================
        start = time.time()
        NNs_triangulated_points_selected, NNs_triangulated_points_selected_proj_errors = select_best_keypoints(
            NNs_triangulated_points=torch.stack(reference_image_dict["triangulated_points"], dim=0),
            NNs_errors_proj1=np.stack(reference_image_dict["triangulated_points_errors_proj1"], axis=0),
            NNs_errors_proj2=np.stack(reference_image_dict["triangulated_points_errors_proj2"], axis=0))
        timings['select_best_keypoints'].append(time.time() - start)

        # =================== Step 5: Create New Gaussians ===================
        start = time.time()
        viewpoint_cam1 = viewpoint_stack[source_idx]
        N = len(NNs_triangulated_points_selected)
        new_xyz = NNs_triangulated_points_selected[:, :-1]
        all_new_xyz.append(new_xyz)
        all_new_features_dc.append(RGB2SH(torch.tensor(kptsA_color.astype(np.float32) / 255.)).unsqueeze(1))
        all_new_features_rest.append(torch.stack([gaussians._features_rest[-1].clone().detach() * 0.] * N, dim=0))

        mask_bad_points = torch.tensor(
            NNs_triangulated_points_selected_proj_errors > keypoint_fit_error_tolerance,
            dtype=torch.float32).unsqueeze(1).to(device)

        all_new_opacities.append(torch.stack([gaussians._opacity[-1].clone().detach()] * N, dim=0) * 0. - mask_bad_points * (1e1))

        dist_points_to_cam1 = torch.linalg.norm(viewpoint_cam1.camera_center.clone().detach() - new_xyz, dim=1, ord=2)
        all_new_scaling.append(gaussians.scaling_inverse_activation((dist_points_to_cam1 * scaling_factor).unsqueeze(1).repeat(1, 3)))
        all_new_rotation.append(torch.stack([gaussians._rotation[-1].clone().detach()] * N, dim=0))
        timings['save_gaussians'].append(time.time() - start)

    # =================== Final Densification Postfix ===================
    start = time.time()
    all_new_xyz = torch.cat(all_new_xyz, dim=0) 
    all_new_features_dc = torch.cat(all_new_features_dc, dim=0)
    new_tmp_radii = torch.zeros(all_new_xyz.shape[0])
    prune_mask = torch.ones(all_new_xyz.shape[0], dtype=torch.bool)

    gaussians.densification_postfix(
        all_new_xyz[prune_mask].to(device),
        all_new_features_dc[prune_mask].to(device),
        torch.cat(all_new_features_rest, dim=0)[prune_mask].to(device),
        torch.cat(all_new_opacities, dim=0)[prune_mask].to(device),
        torch.cat(all_new_scaling, dim=0)[prune_mask].to(device),
        torch.cat(all_new_rotation, dim=0)[prune_mask].to(device),
        new_tmp_radii[prune_mask].to(device)
    )
    timings['final_densification_postfix'].append(time.time() - start)

    # =================== Print Profiling Results ===================
    print("\n=== Profiling Summary (average per frame) ===")
    for key, times in timings.items():
        print(f"{key:35s}: {sum(times) / len(times):.4f} sec (total {sum(times):.2f} sec)")

    return viewpoint_stack, closest_indices_selected, visualizations