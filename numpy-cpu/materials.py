from functools import partial

import numpy as np


def unit(v):
  return v / np.linalg.norm(v)

def random_vector():
  return np.random.random(size=3) * 2 - 1

def random_vector_range(min, max):
  return min + random_vector() * (max - min)

def random_in_unit_sphere():
  u, v, cbrt_r = np.random.random(size=3)
  theta = u * 2 * np.pi
  phi = np.arccos(2 * v - 1)
  r = np.cbrt(cbrt_r)
  return np.array([ 
      r * np.sin(phi) * np.cos(theta),
      r * np.sin(phi) * np.sin(theta),
      r * np.cos(phi)
    ], dtype=np.float32)

def random_unit_vector():
  return unit(random_in_unit_sphere())


def random_in_unit_disc():
  r2, theta_r = np.random.random(size=2)
  r = np.sqrt(r2)
  theta = theta_r * 2 * np.pi
  return np.array([r * np.cos(theta), r * np.sin(theta), 0.0])

def random_in_unit_disc_vector(N):
  r2, theta_r = np.random.random(size=(2, N))
  r = np.sqrt(r2)
  theta = theta_r * 2 * np.pi
  return np.stack([
    r * np.cos(theta),
    r * np.sin(theta),
    np.zeros(N, dtype=np.float32)
  ], axis=-1)

def reflect(v, n):
  return v - 2 * np.dot(v, n) * n

def metal_material(albedo, fuzz, ray_in, hit):
  ray_dir = ray_in[1]
  
  r = reflect(unit(ray_dir), unit(hit.normal))
  rr = r + fuzz * random_in_unit_sphere()
  return np.dot(r, hit.normal) > 0, \
    albedo, np.array([hit.p, rr])

def lambertian_material(albedo, ray_in, hit):
  normal = hit.normal + random_unit_vector()
  if np.min(np.abs(normal)) < 1e-8:
    normal = hit.normal
  
  return True, albedo, np.array([hit.p, normal])

def reflectance(cos_theta, refraction_ratio):
  r0 = (1 - refraction_ratio) / (1 + refraction_ratio)
  r0 = r0 * r0
  return r0 + (1 - r0) * np.power(1 - cos_theta, 5)

def refract(v, n, refraction_ratio):
  cos_theta = np.clip(np.dot(-v, n), -1, 1)
  r_out_perp = refraction_ratio * (v + cos_theta * n)
  r_out_parallel = -np.sqrt(np.abs(1 - r_out_perp.dot(r_out_perp))) * n
  return r_out_perp + r_out_parallel

def dialectric_material(refractive_index, ray_in, hit):
  color = np.ones(3, dtype=np.float32)
  refraction_ratio = 1.0 / refractive_index if hit.front_face else refractive_index

  udir = unit(ray_in[1])
  cos_theta = np.clip(np.dot(-udir, hit.normal), -1, 1)
  sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

  cannot_refract = refraction_ratio * sin_theta > 1.0
  if cannot_refract or reflectance(cos_theta, refraction_ratio) > np.random.random():
    dir = reflect(udir, hit.normal)
  else:
    dir = refract(udir, hit.normal, refraction_ratio)

  return True, color, np.array([hit.p, dir])

def create_metal_material(albedo, fuzz):
  return partial(metal_material, albedo, fuzz)

def create_lambertian_material(albedo):
  return partial(lambertian_material, albedo)

def create_dialectric_material(refractive_index):
  return partial(dialectric_material, refractive_index)

def create_material_by_type(material_type, *args):
  if material_type == 'metal':
    return partial(metal_material, *args)
  elif material_type == 'lambertian':
    return partial(lambertian_material, *args)
  elif material_type == 'dielectric':
    return partial(dialectric_material, *args)
  else:
    raise ValueError('Unknown material type: {}'.format(material_type))