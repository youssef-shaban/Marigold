from .base_normals_dataset import BaseNormalsDataset
from .base_depth_dataset import DatasetMode
import os
import random
from PIL import Image
import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OrthographicCameras,
    RasterizationSettings,
    MeshRasterizer,
)

from pytorch3d.transforms import RotateAxisAngle
import torch.nn.functional as F
from torch.utils.data import get_worker_info
from functools import lru_cache
from collections import OrderedDict


class AIM2PCNormalsDataset(BaseNormalsDataset):
    def __init__(self, 
    mode: DatasetMode,
    dataset_dir: str,
    disp_name: str,
    filename_ls_path: str = "",
    using_filenames: bool = False,
    augmentation_args: dict = None,
    resize_to_hw=None,
    **kwargs,
    ) -> None:
        super().__init__(
            mode=mode,
            filename_ls_path=filename_ls_path,
            dataset_dir=dataset_dir,
            disp_name=disp_name,
            augmentation_args=augmentation_args,
            resize_to_hw=resize_to_hw,
            using_filenames=using_filenames,
            **kwargs,
        )
        self.filenames = [f for f in os.listdir(os.path.join(self.dataset_dir, "image")) if f.endswith(".jpg")]
        
        # Store rasterizer settings for lazy initialization per worker
        self.raster_settings = RasterizationSettings(
            image_size=(self.resize_to_hw[1], self.resize_to_hw[0]),
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.R2, self.T2 = look_at_view_transform(dist=2.0, elev=40, azim=180)
        
        # Lazy initialization - will be created per worker
        self.rasterizer_oblique = None
        
        # LRU mesh cache with maximum size to prevent memory leaks
        # With 16 workers, limit each worker to ~50 meshes (adjust based on mesh size)
        self.mesh_cache = OrderedDict()
        self.max_cache_size = 50  # Maximum number of meshes to cache per worker


    def _ensure_rasterizer(self):
        """Lazy initialization of rasterizer per worker on CPU to avoid GPU memory conflicts."""
        if self.rasterizer_oblique is None:
            # Use CPU for rendering to avoid GPU memory conflicts with training
            # With 16 workers on CPU, this is still much faster than original setup
            device = torch.device('cpu')
            cam_oblique = OrthographicCameras(
                R=self.R2.to(device), 
                T=self.T2.to(device), 
                focal_length=0.8,
                device=device
            )
            self.rasterizer_oblique = MeshRasterizer(
                cameras=cam_oblique, 
                raster_settings=self.raster_settings
            )

    def _get_data_item(self, index):
        rgb_path = os.path.join(self.dataset_dir, "image", f"{self.filenames[index]}")
        mesh_path = os.path.join(self.dataset_dir, "mesh", f"{self.filenames[index].replace('.jpg', '.obj')}")
        mask_path = os.path.join(self.dataset_dir, "roof_intuitive_mask", f"{self.filenames[index]}")
        
        angle_deg = random.uniform(0, 180)
        rasters = {}

        rasters.update(self._load_rgb_data(rgb_path, mask_path, angle_deg if DatasetMode.RGB_ONLY != self.mode else 0.0))
        if DatasetMode.RGB_ONLY != self.mode:
            rasters.update(self._render_normals(mesh_path, angle_deg))
        
        
        return rasters, {"index": index, "rgb_relative_path": os.path.join("image", self.filenames[index])}


    def _load_rgb_data(self, rgb_path, mask_path, angle_deg):
        rgb = Image.open(rgb_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # Save original aspect ratio before resizing
        orig_width, orig_height = rgb.size
        orig_aspect_ratio = float(orig_width) / float(orig_height) if orig_height != 0 else 0.0

        rgb = np.array(rgb)
        mask = np.array(mask)
        rgb[mask < 120] = 0
        masked_rgb = Image.fromarray(rgb, mode="RGB")
        rgb_resized = masked_rgb.resize(self.resize_to_hw)
        if DatasetMode.RGB_ONLY != self.mode:
            rgb_fliped = rgb_resized.transpose(Image.FLIP_LEFT_RIGHT)
            rgb_rotated = rgb_fliped.rotate(180)
            final_rgb = rgb_rotated.rotate(angle_deg)
        final_rgb = np.array(final_rgb)
        final_rgb = np.transpose(final_rgb, (2, 0, 1)).astype(int)

        final_rgb_norm = final_rgb / 255.0 * 2.0 - 1.0
        return {"rgb_int": torch.from_numpy(final_rgb).int()
        , "rgb_norm": torch.from_numpy(final_rgb_norm).float()
        , "orig_aspect_ratio": orig_aspect_ratio
        }

    def _render_normals(self, mesh_path, angle_deg):
        # Ensure rasterizer is initialized for this worker
        self._ensure_rasterizer()
        
        # Get device from rasterizer
        device = self.rasterizer_oblique.cameras.device
        
        # Load mesh from cache or disk with LRU eviction
        if mesh_path in self.mesh_cache:
            # Move to end to mark as recently used
            self.mesh_cache.move_to_end(mesh_path)
            mesh = self.mesh_cache[mesh_path]
        else:
            # Load new mesh
            mesh = load_objs_as_meshes([str(mesh_path)], load_textures=False, device=device)
            mesh = self._prepare_mesh(mesh)
            
            # Add to cache
            self.mesh_cache[mesh_path] = mesh
            
            # Evict oldest if cache is full
            if len(self.mesh_cache) > self.max_cache_size:
                oldest_key = next(iter(self.mesh_cache))
                del self.mesh_cache[oldest_key]
        
        transform = RotateAxisAngle(angle_deg, axis="Y", degrees=True, device=device)

        # verts_padded returns shape (N, V, 3). We keep batch dimension.
        verts = mesh.verts_padded()
        verts_rot = transform.transform_points(verts)

        mesh_rot = Meshes(
            verts=[verts_rot[0]],  # remove batch dim for constructor
            faces=mesh.faces_list(),
            textures=mesh.textures,
        )

        # Render normals
        fragments_oblique = self.rasterizer_oblique(mesh_rot)
        normals_oblique = self._normal_map_normalized(mesh_rot, fragments_oblique)
        normals_oblique = np.transpose(normals_oblique, (2, 0, 1))
        
        # Clean up to free memory
        del fragments_oblique, mesh_rot, verts_rot, transform
        
        return {"normals": torch.from_numpy(normals_oblique).float()}



    def _prepare_mesh(self, mesh):
        verts = mesh.verts_packed()
        center = verts.mean(0)
        verts_centered = verts - center
        scale = verts_centered.abs().max()
        verts_norm = verts_centered / scale

        # Apply pre-rotation around X axis to fix initial orientation if requested
        rot = RotateAxisAngle(-90, axis="X", degrees=True, device=verts.device)
        verts_norm = rot.transform_points(verts_norm[None, ...])[0]

        mesh_rot = Meshes(verts=[verts_norm], faces=mesh.faces_list())
        return mesh_rot
    def _normal_map_normalized(self, mesh: Meshes, fragments) -> np.ndarray:
        """Compute face-normal map and return as float16 array normalized to [-1, 1]."""
        pix_to_face = fragments.pix_to_face[0, ..., 0]  # (H,W)
        H, W = pix_to_face.shape
        faces = mesh.faces_packed()  # (F,3)
        verts = mesh.verts_packed()

        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)  # (F,3)

        # Map pixel faces to normals
        normals_img = face_normals[pix_to_face.clamp(min=0)]  # substitute -1 later

        normals_img = normals_img.cpu().numpy()
        # Invalid pixels (background) where pix_to_face == -1
        mask = pix_to_face.cpu().numpy() == -1
        normals_img[mask] = 0.0

        # Return as float16 in range [-1, 1] (half the size of float32)
        return normals_img.astype(np.float16)
    

    def _training_preprocess(self, rasters):
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)

        return rasters
    
    def __del__(self):
        """Clean up resources to prevent memory leaks."""
        # Clear mesh cache
        if hasattr(self, 'mesh_cache'):
            self.mesh_cache.clear()
        
        # Clean up rasterizer
        if hasattr(self, 'rasterizer_oblique') and self.rasterizer_oblique is not None:
            self.rasterizer_oblique = None