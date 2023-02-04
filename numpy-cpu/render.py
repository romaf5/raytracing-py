import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from materials import random_vector, random_vector_range, unit
from materials import random_in_unit_disc, random_in_unit_disc_vector
from materials import create_dialectric_material, create_metal_material, create_lambertian_material


INFINITY = int(1e9)


class Camera:
  def __init__(self):
    self.aspect_ratio = 3.0 / 2.0
    self.look_from = np.array([13, 2, 3], dtype=np.float32)
    self.look_at = np.zeros(3, dtype=np.float32)
    self.vup = np.array([0, -1, 0], dtype=np.float32)
    self.dist_to_focus = 10.0
    self.aperture = 0.1
    self.vfov = 20.0

    self.build_camera()

  @staticmethod
  def degrees_to_radians(degrees):
    return degrees * np.pi / 180.0

  def build_camera(self): 
    theta = self.degrees_to_radians(self.vfov)
    h = np.tan(theta / 2.0)
    viewport_height = 2.0 * h
    viewport_width = self.aspect_ratio * viewport_height
    
    self.w = unit(self.look_from - self.look_at)
    self.u = unit(np.cross(self.vup, self.w))
    self.v = np.cross(self.w, self.u)

    self.origin = self.look_from
    self.horizontal = self.dist_to_focus * viewport_height * self.u
    self.vertical = self.dist_to_focus * viewport_width * self.v
    self.lower_left_corner = self.origin - self.horizontal * 0.5 - self.vertical * 0.5 - self.dist_to_focus * self.w

    self.lens_radius = self.aperture * 0.5

  def get_image(self, width):
    height = int(width / self.aspect_ratio)
    return np.zeros((height, width, 3), dtype=np.float32)

  def get_rays(self, xs, ys):
    N = xs.shape[0]

    random_disc = random_in_unit_disc_vector(N) * self.lens_radius
    offset = random_disc[..., 0: 1].repeat(3, axis=-1) * self.u[None] + \
      random_disc[..., 1: 2].repeat(3, axis=-1) * self.v[None]

    o = self.origin[None] + offset
    dir = self.lower_left_corner[None] + xs[..., None] * self.horizontal[None] + ys[..., None] * self.vertical[None] - o
    return np.stack((o, dir), axis=1)
    
  def get_ray(self, x, y):
    random_disc = random_in_unit_disc() * self.lens_radius
    offset = random_disc[0] * self.u + random_disc[1] * self.v
    return np.array([self.origin + offset,
            self.lower_left_corner + x * self.horizontal  + y * self.vertical - self.origin - offset])


@dataclass
class Intersection:
  t: np.float32
  p: np.ndarray
  normal: np.ndarray
  material: np.int32
  front_face: np.bool8


class MaterialsPool:
  def __init__(self) -> None:
    ground_material = create_lambertian_material(np.array([0.5, 0.5, 0.5],  dtype=np.float32))
    material1 = create_dialectric_material(1.5)
    material2 = create_lambertian_material(np.array([0.4, 0.2, 0.1], dtype=np.float32))
    material3 = create_metal_material(np.array([0.7, 0.6, 0.5], dtype=np.float32), 0.0)
    self.pool = [ground_material, material1, material2, material3]

  def generate_random_material(self, r_type=None):
    if r_type is None:
      r_type = np.random.choice(['lambertian', 'metal', 'dielectric'])
    if r_type == 'lambertian':
      albedo = random_vector() * random_vector();
      return create_lambertian_material(albedo)
    elif r_type == 'metal':
      albedo = random_vector_range(0.5, 1.0)
      fuzz = np.random.uniform(0.0, 0.5)
      return create_metal_material(albedo, fuzz)
    elif r_type == 'dielectric':
      # glass 
      ref_idx = 1.5 
      return create_dialectric_material(ref_idx)
    raise Exception('Unknown material type')

  def add_random_material(self):
    self.pool.append(self.generate_random_material())

  def __getitem__(self, idx):
    return self.pool[idx]

  def __len__(self):
    return len(self.pool)


class SphericalWorld:
  EPS = 1e-3

  def __init__(self):
    self.materials_pool = MaterialsPool()

    spheres = []
    spheres.append((np.array([0, -1000, 0], dtype=np.float32), 1000, 0))

    for a in range(-11, 11):
      for b in range(-11, 11):
        center = np.array([a + 0.9 * np.random.random(), 0.2, b + 0.9 * np.random.random()], dtype=np.float32)

        distance = np.linalg.norm(center - np.array([4, 0.2, 0], dtype=np.float32))
        if distance > 0.9:
          self.materials_pool.add_random_material()
          spheres.append((center, 0.2, len(self.materials_pool) - 1))

    spheres.append((np.array([0, 1, 0], dtype=np.float32), 1.0, 1))
    spheres.append((np.array([-4, 1, 0], dtype=np.float32), 1.0, 2))
    spheres.append((np.array([4, 1, 0], dtype=np.float32), 1.0, 3))

    self.radius = np.array([radius for _, radius, _ in spheres], dtype=np.float32)
    self.center = np.array([center for center, _, _ in spheres], dtype=np.float32)
    self.material_idx = np.array([material_idx for _, _, material_idx in spheres], dtype=np.int32)

  def intersect(self, ray):
    rorigin, rdir = ray
    oc = rorigin[None] - self.center
    rdir_shaped = rdir[None].repeat(self.center.shape[0], axis=0)
    a = (rdir_shaped * rdir_shaped).sum(-1)
    half_b = (oc * rdir_shaped).sum(-1)
    c = (oc * oc).sum(-1) - self.radius * self.radius
    discriminant = half_b * half_b - a * c
    status = np.array(discriminant >= 0.0)

    sqrtd = np.zeros_like(discriminant)
    sqrtd[discriminant >= 0.0] = np.sqrt(discriminant[discriminant >= 0.0])

    root_a = (-half_b - sqrtd) / a
    root_b = (-half_b + sqrtd) / a
    root_a[root_a <= self.EPS] = root_b[root_a <= self.EPS]
    status[root_a <= self.EPS] = False

    if (np.all(status == False)):
      return False, None

    root_a[~status] = INFINITY
    idx = np.argmin(root_a)
    t = root_a[idx]
    p = rorigin + t * rdir
    normal = (p - self.center[idx]) / self.radius[idx]
    material_idx = self.material_idx[idx]
    material = self.materials_pool[material_idx]
    front_face = np.dot(rdir, normal) < 0.0
    normal = normal if front_face else -normal
    return True, Intersection(t, p, normal, material, front_face)


def trace_ray(ray, world, depth):
  if depth <= 0:
    return np.zeros(3, dtype=np.float32)
  
  (status, hit) = world.intersect(ray)
  if status:
    (status, color, scattered) = hit.material(ray, hit)
    if status:
      return color * trace_ray(scattered, world, depth - 1)
    return np.zeros(3, dtype=np.float32)

  return np.ones(3, dtype=np.float32)

def get_all_pixels_idx(N, M):
  y = np.arange(N)[..., None].repeat(M, axis=-1)
  x = np.arange(M)[None].repeat(N, axis=0)
  return np.stack([x, y], axis=-1)

def render_slice(camera, world, image_shape, max_depth_of_recursion, ly, ry, lx, rx):
  h, w = ry - ly, rx - lx
  pix_idx = get_all_pixels_idx(h, w)
  pix_idx = pix_idx.reshape(-1, 2)
  pix_shift = np.array([lx, ly], dtype=np.float32)
  pix_idx = (pix_idx + pix_shift[None])

  pix_idx = pix_idx.reshape(-1, 2, 1).repeat(samples_per_pixel, axis=-1).transpose(0, 2, 1)
  pix_idx = pix_idx.reshape(-1, 2)

  uv_r = np.random.random(size=pix_idx.shape)
  pix_idx = pix_idx + uv_r
  pix_idx = pix_idx / np.array([image_shape[1] - 1, image_shape[0] - 1], dtype=np.float32)
  rays = camera.get_rays(pix_idx[..., 0], pix_idx[..., 1])
  values = np.array([trace_ray(ray, world, max_depth_of_recursion) for ray in rays])

  values = values.reshape(h, w, samples_per_pixel, 3).mean(axis=2)
  values = np.clip(values, 0.0, 1.0)
  return values


if __name__ == '__main__':
  camera = Camera()
  world = SphericalWorld()
  image = camera.get_image(128)
  
  samples_per_pixel = 500
  max_depth_of_recursion = 50

  start = time.time()
  pos = 0
  jobs = 62
  with ProcessPoolExecutor() as executor:
    futures = []
    step = (image.shape[0] + jobs - 1) // jobs
    for i in range(jobs):
      r = min(pos + step, image.shape[0])
      if pos >= r: break
      future = executor.submit(render_slice, camera, world, image.shape, max_depth_of_recursion, pos, r, 0, image.shape[1])
      futures.append((future, (pos, r, 0, image.shape[1])))
      pos += step

    for future, slice in futures:
      image[slice[0]:slice[1], slice[2]:slice[3]] = future.result() * 255.0

  tend = time.time()
  print(f'{tend - start} seconds elapsed')

  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  cv2.imwrite('render.png', image)
