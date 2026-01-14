from .base_normals_dataset import BaseNormalsDataset, DatasetMode
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
    def __init__(
        self,
        mode: DatasetMode,
        dataset_dir: str,
        disp_name: str,
        filename_ls_path: str = "",
        using_filenames: bool = False,
        augmentation_args: dict = None,
        resize_to_hw=None,
        **kwargs,
    ) -> None:
        self.apply_random_orientation = kwargs.pop("apply_random_orientation", True)
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
        # Cache meshes on CPU to avoid GPU memory issues, only move to GPU for rendering
        # With 16 workers, limit each worker to ~20 meshes to be conservative
        self.mesh_cache = OrderedDict()
        self.max_cache_size = 20  # Maximum number of meshes to cache per worker (on CPU)


    def _ensure_rasterizer(self):
        """Lazy initialization of rasterizer per worker with GPU acceleration."""
        if self.rasterizer_oblique is None:
            # Use GPU for fast rendering, but we'll be careful with memory management
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        
        angle_deg = random.uniform(0, 180) if self.apply_random_orientation else 0.0
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
        if self.apply_random_orientation and DatasetMode.RGB_ONLY != self.mode:
            rgb_fliped = rgb_resized.transpose(Image.FLIP_LEFT_RIGHT)
            rgb_rotated = rgb_fliped.rotate(180)
            final_rgb = rgb_rotated.rotate(angle_deg)
        else:
            final_rgb = rgb_resized
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
        
        # Get GPU device from rasterizer (for rendering only)
        gpu_device = self.rasterizer_oblique.cameras.device
        
        # Load mesh from cache (CPU) or disk with LRU eviction
        # KEY OPTIMIZATION: Cache on CPU to save GPU memory
        if mesh_path in self.mesh_cache:
            # Move to end to mark as recently used
            self.mesh_cache.move_to_end(mesh_path)
            mesh_cpu = self.mesh_cache[mesh_path]
        else:
            # Load new mesh to CPU
            mesh_cpu = load_objs_as_meshes([str(mesh_path)], load_textures=False, device='cpu')
            mesh_cpu = self._prepare_mesh(mesh_cpu)
            
            # Add to cache (on CPU)
            self.mesh_cache[mesh_path] = mesh_cpu
            
            # Evict oldest if cache is full
            if len(self.mesh_cache) > self.max_cache_size:
                oldest_key = next(iter(self.mesh_cache))
                del self.mesh_cache[oldest_key]
        
        # Move mesh to GPU only for rendering
        mesh_gpu = mesh_cpu.to(gpu_device)
        
        # Transform on GPU
        verts = mesh_gpu.verts_padded()
        transform = None
        if self.apply_random_orientation:
            transform = RotateAxisAngle(angle_deg, axis="Y", degrees=True, device=gpu_device)
            verts_rot = transform.transform_points(verts)
        else:
            verts_rot = verts

        mesh_rot = Meshes(
            verts=[verts_rot[0]],  # remove batch dim for constructor
            faces=mesh_gpu.faces_list(),
            textures=mesh_gpu.textures,
        )

        # Render normals on GPU
        fragments_oblique = self.rasterizer_oblique(mesh_rot)
        normals_oblique = self._normal_map_normalized(mesh_rot, fragments_oblique)
        normals_oblique = np.transpose(normals_oblique, (2, 0, 1))
        
        # Aggressive cleanup to free GPU memory immediately
        del fragments_oblique, mesh_rot, verts_rot, mesh_gpu, verts
        if transform is not None:
            del transform
        
        # Clear GPU cache to prevent accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()