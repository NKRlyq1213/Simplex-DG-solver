from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simplex_dg.mesh.manifold import ManifoldMesh
from simplex_dg.reference.operators import ReferenceCache


@dataclass(frozen=True)
class GeometryCache:
    element_vertices: np.ndarray

    X: np.ndarray
    Xr: np.ndarray
    Xs: np.ndarray

    normal: np.ndarray
    sqrt_g: np.ndarray

    g11: np.ndarray
    g12: np.ndarray
    g22: np.ndarray
    gdet: np.ndarray

    ginv11: np.ndarray
    ginv12: np.ndarray
    ginv22: np.ndarray

    grad_r: np.ndarray
    grad_s: np.ndarray

    X_face: np.ndarray
    face_tangent: np.ndarray
    face_jacobian: np.ndarray
    face_normal: np.ndarray
    face_conormal: np.ndarray


def _reference_shape_functions(r: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi1 = -(r + s) / 2.0
    phi2 = (1.0 + r) / 2.0
    phi3 = (1.0 + s) / 2.0

    return phi1, phi2, phi3


def _sphere_project_with_derivative(
    Y: np.ndarray,
    dY: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    normY = np.linalg.norm(Y, axis=-1, keepdims=True)

    if np.any(normY <= 0.0):
        raise ValueError("Projection to sphere received zero vector.")

    X = radius * Y / normY

    Y_dot_dY = np.sum(Y * dY, axis=-1, keepdims=True)

    dX = radius * (
        dY / normY
        - Y * Y_dot_dY / (normY**3)
    )

    return X, dX


def map_reference_to_sphere_element(
    rs: np.ndarray,
    vertices: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rs = np.asarray(rs, dtype=float)
    vertices = np.asarray(vertices, dtype=float)

    if rs.ndim != 2 or rs.shape[1] != 2:
        raise ValueError("rs must have shape (Np, 2).")

    if vertices.shape != (3, 3):
        raise ValueError("vertices must have shape (3, 3).")

    r = rs[:, 0]
    s = rs[:, 1]

    phi1, phi2, phi3 = _reference_shape_functions(r, s)

    v1 = vertices[0]
    v2 = vertices[1]
    v3 = vertices[2]

    Y = (
        phi1[:, None] * v1[None, :]
        + phi2[:, None] * v2[None, :]
        + phi3[:, None] * v3[None, :]
    )

    dYdr = 0.5 * (v2 - v1)
    dYds = 0.5 * (v3 - v1)

    X, Xr = _sphere_project_with_derivative(
        Y=Y,
        dY=np.broadcast_to(dYdr, Y.shape),
        radius=radius,
    )

    _, Xs = _sphere_project_with_derivative(
        Y=Y,
        dY=np.broadcast_to(dYds, Y.shape),
        radius=radius,
    )

    return X, Xr, Xs


def _face_direction_rs(face_id: int) -> tuple[float, float]:
    if face_id == 1:
        return -2.0, 2.0
    if face_id == 2:
        return 0.0, -2.0
    if face_id == 3:
        return 2.0, 0.0

    raise ValueError("face_id must be 1, 2, or 3.")


def map_reference_face_to_sphere_element(
    face_id: int,
    rs_face: np.ndarray,
    vertices: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xf, Xr_f, Xs_f = map_reference_to_sphere_element(
        rs=rs_face,
        vertices=vertices,
        radius=radius,
    )

    drdt, dsdt = _face_direction_rs(face_id)
    Xt = drdt * Xr_f + dsdt * Xs_f

    cross = np.cross(Xr_f, Xs_f)
    norm_cross = np.linalg.norm(cross, axis=1)

    if np.any(norm_cross <= 0.0):
        raise ValueError("Degenerate face normal encountered.")

    nf = cross / norm_cross[:, None]

    tangent_norm = np.linalg.norm(Xt, axis=1)

    if np.any(tangent_norm <= 0.0):
        raise ValueError("Degenerate face tangent encountered.")

    tf = Xt / tangent_norm[:, None]

    # For positively oriented surface patch and CCW local boundary direction,
    # outward co-normal is t_hat cross n_hat.
    conormal = np.cross(tf, nf)
    conormal_norm = np.linalg.norm(conormal, axis=1)

    if np.any(conormal_norm <= 0.0):
        raise ValueError("Degenerate face co-normal encountered.")

    conormal = conormal / conormal_norm[:, None]

    return Xf, Xt, nf, conormal


def build_geometry_cache(
    mesh: ManifoldMesh,
    ref: ReferenceCache,
    validate: bool = True,
) -> GeometryCache:
    vertices = np.asarray(mesh.vertices, dtype=float)
    elements = np.asarray(mesh.elements, dtype=int)
    element_vertices = vertices[elements]

    K = elements.shape[0]
    Np = ref.rs.shape[0]
    Nf = ref.edge_rules[1].n_points

    X = np.zeros((K, Np, 3), dtype=float)
    Xr = np.zeros((K, Np, 3), dtype=float)
    Xs = np.zeros((K, Np, 3), dtype=float)

    for k in range(K):
        X[k], Xr[k], Xs[k] = map_reference_to_sphere_element(
            rs=ref.rs,
            vertices=element_vertices[k],
            radius=mesh.radius,
        )

    cross = np.cross(Xr, Xs)
    sqrt_g = np.linalg.norm(cross, axis=2)

    if np.any(sqrt_g <= 0.0):
        raise ValueError("Degenerate surface metric encountered.")

    normal = cross / sqrt_g[:, :, None]

    g11 = np.sum(Xr * Xr, axis=2)
    g12 = np.sum(Xr * Xs, axis=2)
    g22 = np.sum(Xs * Xs, axis=2)

    gdet = g11 * g22 - g12 * g12

    if np.any(gdet <= 0.0):
        raise ValueError("Non-positive surface metric determinant encountered.")

    ginv11 = g22 / gdet
    ginv12 = -g12 / gdet
    ginv22 = g11 / gdet

    grad_r = ginv11[:, :, None] * Xr + ginv12[:, :, None] * Xs
    grad_s = ginv12[:, :, None] * Xr + ginv22[:, :, None] * Xs

    X_face = np.zeros((K, 3, Nf, 3), dtype=float)
    face_tangent = np.zeros((K, 3, Nf, 3), dtype=float)
    face_jacobian = np.zeros((K, 3, Nf), dtype=float)
    face_normal = np.zeros((K, 3, Nf, 3), dtype=float)
    face_conormal = np.zeros((K, 3, Nf, 3), dtype=float)

    for k in range(K):
        for face_id in (1, 2, 3):
            edge = ref.edge_rules[face_id]

            Xf, Xt, nf, cf = map_reference_face_to_sphere_element(
                face_id=face_id,
                rs_face=edge.rs,
                vertices=element_vertices[k],
                radius=mesh.radius,
            )

            f = face_id - 1

            X_face[k, f] = Xf
            face_tangent[k, f] = Xt
            face_jacobian[k, f] = np.linalg.norm(Xt, axis=1)
            face_normal[k, f] = nf
            face_conormal[k, f] = cf

    cache = GeometryCache(
        element_vertices=element_vertices,
        X=X,
        Xr=Xr,
        Xs=Xs,
        normal=normal,
        sqrt_g=sqrt_g,
        g11=g11,
        g12=g12,
        g22=g22,
        gdet=gdet,
        ginv11=ginv11,
        ginv12=ginv12,
        ginv22=ginv22,
        grad_r=grad_r,
        grad_s=grad_s,
        X_face=X_face,
        face_tangent=face_tangent,
        face_jacobian=face_jacobian,
        face_normal=face_normal,
        face_conormal=face_conormal,
    )

    if validate:
        validate_geometry_cache(mesh, ref, cache)

    return cache


def validate_geometry_cache(
    mesh: ManifoldMesh,
    ref: ReferenceCache,
    geom: GeometryCache,
    tol: float = 1e-9,
) -> None:
    K = mesh.elements.shape[0]
    Np = ref.rs.shape[0]
    Nf = ref.edge_rules[1].n_points

    if geom.X.shape != (K, Np, 3):
        raise ValueError("geom.X has wrong shape.")

    if geom.Xr.shape != (K, Np, 3):
        raise ValueError("geom.Xr has wrong shape.")

    if geom.Xs.shape != (K, Np, 3):
        raise ValueError("geom.Xs has wrong shape.")

    if geom.X_face.shape != (K, 3, Nf, 3):
        raise ValueError("geom.X_face has wrong shape.")

    r_volume = np.linalg.norm(geom.X, axis=2)
    r_face = np.linalg.norm(geom.X_face, axis=3)

    if not np.allclose(r_volume, mesh.radius, atol=tol, rtol=tol):
        raise ValueError("Volume geometry nodes are not on the sphere.")

    if not np.allclose(r_face, mesh.radius, atol=tol, rtol=tol):
        raise ValueError("Face geometry nodes are not on the sphere.")

    normal_norm = np.linalg.norm(geom.normal, axis=2)
    face_normal_norm = np.linalg.norm(geom.face_normal, axis=3)
    conormal_norm = np.linalg.norm(geom.face_conormal, axis=3)

    if not np.allclose(normal_norm, 1.0, atol=tol, rtol=tol):
        raise ValueError("Volume normals are not unit length.")

    if not np.allclose(face_normal_norm, 1.0, atol=tol, rtol=tol):
        raise ValueError("Face normals are not unit length.")

    if not np.allclose(conormal_norm, 1.0, atol=tol, rtol=tol):
        raise ValueError("Face co-normals are not unit length.")

    if np.any(geom.sqrt_g <= 0.0):
        raise ValueError("geom.sqrt_g must be positive.")

    if np.any(geom.face_jacobian <= 0.0):
        raise ValueError("geom.face_jacobian must be positive.")

    if np.max(np.abs(np.sum(geom.normal * geom.Xr, axis=2))) > 1e-8:
        raise ValueError("normal is not orthogonal to Xr.")

    if np.max(np.abs(np.sum(geom.normal * geom.Xs, axis=2))) > 1e-8:
        raise ValueError("normal is not orthogonal to Xs.")

    face_tangent_unit = geom.face_tangent / geom.face_jacobian[:, :, :, None]

    if np.max(np.abs(np.sum(face_tangent_unit * geom.face_normal, axis=3))) > 1e-8:
        raise ValueError("face tangent is not orthogonal to face normal.")

    if np.max(np.abs(np.sum(face_tangent_unit * geom.face_conormal, axis=3))) > 1e-8:
        raise ValueError("face tangent is not orthogonal to face co-normal.")

    if np.max(np.abs(np.sum(geom.face_normal * geom.face_conormal, axis=3))) > 1e-8:
        raise ValueError("face normal is not orthogonal to face co-normal.")


def dual_basis_residuals(geom: GeometryCache) -> dict[str, float]:
    rr = np.sum(geom.grad_r * geom.Xr, axis=2)
    rs = np.sum(geom.grad_r * geom.Xs, axis=2)
    sr = np.sum(geom.grad_s * geom.Xr, axis=2)
    ss = np.sum(geom.grad_s * geom.Xs, axis=2)

    return {
        "max_abs_grad_r_dot_Xr_minus_1": float(np.max(np.abs(rr - 1.0))),
        "max_abs_grad_r_dot_Xs": float(np.max(np.abs(rs))),
        "max_abs_grad_s_dot_Xr": float(np.max(np.abs(sr))),
        "max_abs_grad_s_dot_Xs_minus_1": float(np.max(np.abs(ss - 1.0))),
    }