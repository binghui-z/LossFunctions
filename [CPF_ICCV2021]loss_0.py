# from https://github.com/hassony2/obman_train/blob/master/mano_train/networks/branches/contactutils.py
import torch
import trimesh
import numpy as np
from liegroups import SO3

import numpy as np
# from others.triangle_hash import TriangleHash as _TriangleHash

def check_mesh_contains(mesh, points, hash_resolution=512):
    intersector = MeshIntersector(mesh, hash_resolution)
    contains = intersector.query(points)
    return contains


class MeshIntersector:
    def __init__(self, mesh, resolution=512):
        triangles = mesh.vertices[mesh.faces].astype(np.float64)
        n_tri = triangles.shape[0]

        self.resolution = resolution
        self.bbox_min = triangles.reshape(3 * n_tri, 3).min(axis=0)
        self.bbox_max = triangles.reshape(3 * n_tri, 3).max(axis=0)
        # Tranlate and scale it to [0.5, self.resolution - 0.5]^3
        self.scale = (resolution - 1) / (self.bbox_max - self.bbox_min)
        self.translate = 0.5 - self.scale * self.bbox_min

        self._triangles = triangles = self.rescale(triangles)
        # assert(np.allclose(triangles.reshape(-1, 3).min(0), 0.5))
        # assert(np.allclose(triangles.reshape(-1, 3).max(0), resolution - 0.5))

        triangles2d = triangles[:, :, :2]
        self._tri_intersector2d = TriangleIntersector2d(
            triangles2d, resolution)

    def query(self, points):
        # Rescale points
        points = self.rescale(points)

        # placeholder result with no hits we'll fill in later
        contains = np.zeros(len(points), dtype=np.bool)

        # cull points outside of the axis aligned bounding box
        # this avoids running ray tests unless points are close
        inside_aabb = np.all(
            (0 <= points) & (points <= self.resolution), axis=1)
        if not inside_aabb.any():
            return contains

        # Only consider points inside bounding box
        mask = inside_aabb
        points = points[mask]

        # Compute intersection depth and check order
        points_indices, tri_indices = self._tri_intersector2d.query(points[:, :2])

        triangles_intersect = self._triangles[tri_indices]
        points_intersect = points[points_indices]

        depth_intersect, abs_n_2 = self.compute_intersection_depth(
            points_intersect, triangles_intersect)

        # Count number of intersections in both directions
        smaller_depth = depth_intersect >= points_intersect[:, 2] * abs_n_2
        bigger_depth = depth_intersect < points_intersect[:, 2] * abs_n_2
        points_indices_0 = points_indices[smaller_depth]
        points_indices_1 = points_indices[bigger_depth]

        nintersect0 = np.bincount(points_indices_0, minlength=points.shape[0])
        nintersect1 = np.bincount(points_indices_1, minlength=points.shape[0])
        
        # Check if point contained in mesh
        contains1 = (np.mod(nintersect0, 2) == 1)
        contains2 = (np.mod(nintersect1, 2) == 1)
        if (contains1 != contains2).any():
            print('Warning: contains1 != contains2 for some points.')
        contains[mask] = (contains1 & contains2)
        return contains

    def compute_intersection_depth(self, points, triangles):
        t1 = triangles[:, 0, :]
        t2 = triangles[:, 1, :]
        t3 = triangles[:, 2, :]

        v1 = t3 - t1
        v2 = t2 - t1
        # v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
        # v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)

        normals = np.cross(v1, v2)
        alpha = np.sum(normals[:, :2] * (t1[:, :2] - points[:, :2]), axis=1)

        n_2 = normals[:, 2]
        t1_2 = t1[:, 2]
        s_n_2 = np.sign(n_2)
        abs_n_2 = np.abs(n_2)

        mask = (abs_n_2 != 0)
    
        depth_intersect = np.full(points.shape[0], np.nan)
        depth_intersect[mask] = \
            t1_2[mask] * abs_n_2[mask] + alpha[mask] * s_n_2[mask]

        # Test the depth:
        # TODO: remove and put into tests
        # points_new = np.concatenate([points[:, :2], depth_intersect[:, None]], axis=1)
        # alpha = (normals * t1).sum(-1)
        # mask = (depth_intersect == depth_intersect)
        # assert(np.allclose((points_new[mask] * normals[mask]).sum(-1),
        #                    alpha[mask]))
        return depth_intersect, abs_n_2

    def rescale(self, array):
        array = self.scale * array + self.translate
        return array


class TriangleIntersector2d:
    def __init__(self, triangles, resolution=128):
        self.triangles = triangles
        self.tri_hash = _TriangleHash(triangles, resolution)

    def query(self, points):
        point_indices, tri_indices = self.tri_hash.query(points)
        point_indices = np.array(point_indices, dtype=np.int64)
        tri_indices = np.array(tri_indices, dtype=np.int64)
        points = points[point_indices]
        triangles = self.triangles[tri_indices]
        mask = self.check_triangles(points, triangles)
        point_indices = point_indices[mask]
        tri_indices = tri_indices[mask]
        return point_indices, tri_indices

    def check_triangles(self, points, triangles):
        contains = np.zeros(points.shape[0], dtype=np.bool)
        A = triangles[:, :2] - triangles[:, 2:]
        A = A.transpose([0, 2, 1])
        y = points - triangles[:, 2]

        detA = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
        
        mask = (np.abs(detA) != 0.)
        A = A[mask]
        y = y[mask]
        detA = detA[mask]

        s_detA = np.sign(detA)
        abs_detA = np.abs(detA)

        u = (A[:, 1, 1] * y[:, 0] - A[:, 0, 1] * y[:, 1]) * s_detA
        v = (-A[:, 1, 0] * y[:, 0] + A[:, 0, 0] * y[:, 1]) * s_detA

        sum_uv = u + v
        contains[mask] = (
            (0 < u) & (u < abs_detA) & (0 < v) & (v < abs_detA)
            & (0 < sum_uv) & (sum_uv < abs_detA)
        )
        return contains

def batch_index_select(inp, dim, index):
    views = [inp.shape[0]] + [1 if i != dim else -1 for i in range(1, len(inp.shape))]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)


def thresh_ious(gt_dists, pred_dists, thresh):
    """
    Computes the contact intersection over union for a given threshold
    """
    gt_contacts = gt_dists <= thresh
    pred_contacts = pred_dists <= thresh
    inter = (gt_contacts * pred_contacts).sum(1).float()
    union = union = (gt_contacts | pred_contacts).sum(1).float()
    iou = torch.zeros_like(union)
    iou[union != 0] = inter[union != 0] / union[union != 0]
    return iou


def masked_mean_loss(dists, mask):
    mask = mask.float()
    valid_vals = mask.sum()
    device = dists.device
    if valid_vals > 0:
        loss = (mask * dists).sum() / valid_vals
    else:
        loss = torch.Tensor([0]).to(device)
    return loss


def batch_pairwise_dist(x, y):
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = torch.sum(x ** 2, dim=2).unsqueeze(2).expand_as(zz)
    ry = torch.sum(y ** 2, dim=2).unsqueeze(1).expand_as(zz)
    P = rx + ry - 2 * zz
    return P


def thres_loss(vals, thres=25):
    """
    Args:
        vals: positive values !
    """
    thres_mask = (vals < thres).float()
    loss = masked_mean_loss(vals, thres_mask)
    return loss


# http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
# Full batch mode
def batch_mesh_contains_points(
    ray_origins, obj_triangles, direction=None,
):
    """Times efficient but memory greedy !
    Computes ALL ray/triangle intersections and then counts them to determine
    if point inside mesh

    Args:
    ray_origins: (batch_size x point_nb x 3)
    obj_triangles: (batch_size, triangle_nb, vertex_nb=3, vertex_coords=3)
    tol_thresh: To determine if ray and triangle are //
    Returns:
    exterior: (batch_size, point_nb) 1 if the point is outside mesh, 0 else
    """
    device = ray_origins.device
    if direction is None:
        direction = torch.Tensor([0.4395064455, 0.617598629942, 0.652231566745]).to(device)
    tol_thresh = 0.0000001
    # ray_origins.requires_grad = False
    # obj_triangles.requires_grad = False
    batch_size = obj_triangles.shape[0]
    triangle_nb = obj_triangles.shape[1]
    point_nb = ray_origins.shape[1]

    # Batch dim and triangle dim will flattened together
    batch_points_size = batch_size * triangle_nb
    # Direction is random but shared
    v0, v1, v2 = obj_triangles[:, :, 0], obj_triangles[:, :, 1], obj_triangles[:, :, 2]
    # Get edges
    v0v1 = v1 - v0
    v0v2 = v2 - v0

    # Expand needed vectors
    batch_direction = direction.view(1, 1, 3).expand(batch_size, triangle_nb, 3)

    # Compute ray/triangle intersections
    pvec = torch.cross(batch_direction, v0v2, dim=2)
    dets = torch.bmm(v0v1.view(batch_points_size, 1, 3), pvec.view(batch_points_size, 3, 1)).view(
        batch_size, triangle_nb
    )

    # Check if ray and triangle are parallel
    parallel = abs(dets) < tol_thresh
    invdet = 1 / (dets + 0.1 * tol_thresh)

    # Repeat mesh info as many times as there are rays
    triangle_nb = v0.shape[1]
    v0 = v0.repeat(1, point_nb, 1)
    v0v1 = v0v1.repeat(1, point_nb, 1)
    v0v2 = v0v2.repeat(1, point_nb, 1)
    hand_verts_repeated = (
        ray_origins.view(batch_size, point_nb, 1, 3)
        .repeat(1, 1, triangle_nb, 1)
        .view(ray_origins.shape[0], triangle_nb * point_nb, 3)
    )
    pvec = pvec.repeat(1, point_nb, 1)
    invdet = invdet.repeat(1, point_nb)
    tvec = hand_verts_repeated - v0
    u_val = (
        torch.bmm(tvec.view(batch_size * tvec.shape[1], 1, 3), pvec.view(batch_size * tvec.shape[1], 3, 1),).view(
            batch_size, tvec.shape[1]
        )
        * invdet
    )
    # Check ray intersects inside triangle
    u_correct = (u_val > 0) * (u_val < 1)
    qvec = torch.cross(tvec, v0v1, dim=2)

    batch_direction = batch_direction.repeat(1, point_nb, 1)
    v_val = (
        torch.bmm(
            batch_direction.view(batch_size * qvec.shape[1], 1, 3), qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    v_correct = (v_val > 0) * (u_val + v_val < 1)
    t = (
        torch.bmm(v0v2.view(batch_size * qvec.shape[1], 1, 3), qvec.view(batch_size * qvec.shape[1], 3, 1),).view(
            batch_size, qvec.shape[1]
        )
        * invdet
    )
    # Check triangle is in front of ray_origin along ray direction
    t_pos = t >= tol_thresh
    parallel = parallel.repeat(1, point_nb)
    # # Check that all intersection conditions are met
    not_parallel = parallel.logical_not()
    final_inter = v_correct * u_correct * not_parallel * t_pos
    # Reshape batch point/vertices intersection matrix
    # final_intersections[batch_idx, point_idx, triangle_idx] == 1 means ray
    # intersects triangle
    final_intersections = final_inter.view(batch_size, point_nb, triangle_nb)
    # Check if intersection number accross mesh is odd to determine if point is
    # outside of mesh
    exterior = final_intersections.sum(2) % 2 == 0
    return exterior


def penetration_loss_hand_in_obj(hand_verts, obj_verts, obj_faces, mode="max"):
    device = hand_verts.device

    # unsqueeze fist dimension so that we can use hasson's utils directly
    hand_verts = hand_verts.unsqueeze(0)
    obj_verts = obj_verts.unsqueeze(0)

    # Get obj triangle positions
    obj_triangles = obj_verts[:, obj_faces]
    exterior = batch_mesh_contains_points(
        hand_verts.detach(), obj_triangles.detach()
    )  # exterior computation transfers no gradients
    penetr_mask = ~exterior

    # only compute exterior related stuff
    valid_vals = penetr_mask.sum()
    if valid_vals > 0:
        selected_hand_verts = hand_verts[penetr_mask, :]
        selected_hand_verts = selected_hand_verts.unsqueeze(0)
        dists = batch_pairwise_dist(selected_hand_verts, obj_verts)
        mins_sel_hand_to_obj, mins_sel_hand_to_obj_idx = torch.min(dists, 2)

        # results_close = batch_index_select(obj_verts, 1, mins_sel_hand_to_obj_idx)
        # collision_vals = ((results_close - selected_hand_verts) ** 2).sum(2)
        collision_vals = mins_sel_hand_to_obj

        if mode == "max":
            penetr_loss = torch.max(collision_vals)  # max
        elif mode == "mean":
            penetr_loss = torch.mean(collision_vals)
        elif mode == "sum":
            penetr_loss = torch.sum(collision_vals)
        else:
            raise KeyError("unexpected penetration loss mode")
    else:
        penetr_loss = torch.Tensor([0.0]).to(device)
    return penetr_loss


def pairwise_dist(x, y):
    zz = torch.mm(x, y.transpose(1, 0))
    rx = torch.sum(x ** 2, dim=1).unsqueeze(1).expand_as(zz)
    ry = torch.sum(y ** 2, dim=1).unsqueeze(0).expand_as(zz)
    P = rx + ry - 2 * zz
    return P


# ! deprecated & warning
# ! this function only voxelize surface, thus the voxelized shape will not be solid
# ! this is different with what we wanted, and the original implementation
# ! in hassonys repo has similar issue and the paper is to a certain extent misleading
def intersection_volume(hand_verts, hand_faces, obj_verts, obj_faces, pitch=0.01):
    hand_trimesh = trimesh.Trimesh(vertices=np.asarray(hand_verts), faces=np.asarray(hand_faces))
    obj_trimesh = trimesh.Trimesh(vertices=np.asarray(obj_verts), faces=np.asarray(obj_faces))
    obj_voxel = obj_trimesh.voxelized(pitch=pitch)
    obj_voxel_points = obj_voxel.points
    inside = hand_trimesh.contains(obj_voxel_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


def solid_intersection_volume(hand_verts, hand_faces, obj_vox_points, obj_tsl, obj_rot, obj_vox_el_vol):
    # first transf points to desired location
    # convert obj_rot to rotation matrix
    if obj_rot.shape == (3, 3):
        obj_rotmat = obj_rot
    else:
        obj_rotmat = SO3.exp(obj_rot).as_matrix()
    obj_vox_points_transf = (obj_rotmat @ obj_vox_points.T).T
    obj_vox_points_transf = obj_vox_points_transf + obj_tsl
    # create hand trimesh
    hand_trimesh = trimesh.Trimesh(vertices=np.asarray(hand_verts), faces=np.asarray(hand_faces))
    # _ = hand_trimesh.vertex_normals
    # _ = hand_trimesh.face_normals
    # hand_trimesh.fix_normals()
    # inside = hand_trimesh.contains(obj_vox_points_transf)
    inside = check_mesh_contains(hand_trimesh, obj_vox_points_transf)
    volume = inside.sum() * obj_vox_el_vol
    return volume, obj_vox_points_transf, inside

