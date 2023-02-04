import warp as wp
wp.init()

import numpy as np
import matplotlib.pyplot as plt


LAMBERTIAN_MATERIAL_TYPE = wp.constant(0)
METAL_MATERIAL_TYPE = wp.constant(1)
DIELECTRIC_MATERIAL_TYPE = wp.constant(2)


ZERO = wp.constant(0)
ONE = wp.constant(1)
EPS_T = wp.constant(1e-3)
EPS = wp.constant(1e-8)
INFINITY = wp.constant(1e9)


@wp.struct 
class Intersection:
  success: wp.array(dtype=wp.int32)
  t: wp.array(dtype=wp.float32)
  p: wp.array(dtype=wp.vec3)
  normal: wp.array(dtype=wp.vec3)
  material_idx: wp.array(dtype=wp.int32)
  front_face: wp.array(dtype=wp.int32)

@wp.struct
class Spheres:
  center: wp.array(dtype=wp.vec3)
  radius: wp.array(dtype=wp.float32)
  material_idx: wp.array(dtype=wp.int32)
  cnt: wp.int32

@wp.struct
class Materials:
  albedo: wp.array(dtype=wp.vec3)
  fuzz: wp.array(dtype=wp.float32)
  refractive_index: wp.array(dtype=wp.float32)
  material_type: wp.array(dtype=wp.int32)

@wp.struct 
class ApplyMaterialResult:
  success: wp.array(dtype=wp.int32)
  albedo: wp.array(dtype=wp.vec3)
  center: wp.array(dtype=wp.vec3)
  normal: wp.array(dtype=wp.vec3)

@wp.func
def unit(vec: wp.vec3):
  return vec / wp.length(vec)


@wp.struct 
class LocalState:
  seed: wp.array(dtype=wp.int32)


@wp.func
def next_random(state: LocalState):
  tid = wp.tid()
  seed = wp.rand_init(state.seed[tid], tid)
  value = wp.randf(seed)
  state.seed[tid] = state.seed[tid] + 1
  return value

@wp.func
def next_random_in_range(state: LocalState, min_value: float, max_value: float):
  return min_value + (max_value - min_value) * next_random(state)


@wp.func
def random_vec2(local_state: LocalState):
  x = next_random(local_state)
  y = next_random(local_state)
  return wp.vec2(x, y)
  
@wp.func
def random_vec3(state: LocalState):
  x = next_random_in_range(state, -1.0, 1.0)
  y = next_random_in_range(state, -1.0, 1.0)
  z = next_random_in_range(state, -1.0, 1.0)
  return wp.vec3(x, y, z)  
  
@wp.func
def random_unit_vector(state: LocalState):
  return unit(random_vec3(state))

@wp.func
def reflect(v: wp.vec3, n: wp.vec3):
  return v - 2.0 * wp.dot(v, n) * n

@wp.func
def random_in_unit_disc(state: LocalState):
  while True:
    x = next_random_in_range(state, -1.0, 1.0)
    y = next_random_in_range(state, -1.0, 1.0)
    p = wp.vec3(x, y, 0.0)
    if wp.length(p) <= 1.0:
      return p


@wp.func
def random_in_unit_sphere(state: LocalState):
  while True:
    p = random_vec3(state)
    if wp.length(p) <= 1.0:
      return p 

@wp.func
def reflectance(cos_theta: wp.float32, refraction_ratio: wp.float32):
  r0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio)
  r0 = r0 * r0
  return r0 + (1.0 - r0) * wp.pow(1.0 - cos_theta, 5.0)

@wp.func
def refract(v: wp.vec3, n: wp.vec3, refraction_ratio: float):
  cos_theta = wp.clamp(-wp.dot(v, n), -1.0, 1.0)
  r_out_perp = refraction_ratio * (v + cos_theta * n)
  r_out_parallel = -wp.sqrt(wp.abs(1.0 - wp.dot(r_out_perp, r_out_perp))) * n
  return r_out_perp + r_out_parallel

@wp.func
def wp_abs(vec: wp.vec3):
  return wp.vec3(wp.abs(vec[0]), wp.abs(vec[1]), wp.abs(vec[2]))

@wp.func
def wp_min(vec: wp.vec3):
  return wp.min(vec[0], wp.min(vec[1], vec[2]))



@wp.func
def intersect_spheres(
                    spheres: Spheres,
                    ro: wp.vec3, rd: wp.vec3, intersection: Intersection, rid: wp.int32):
  intersection.success[rid] = 0
  intersection.t[rid] = INFINITY
  for i in range(spheres.cnt):
    center = spheres.center[i]
    radius = spheres.radius[i]
    material_idx = spheres.material_idx[i]
    oc = ro - center
    a = wp.dot(rd, rd)
    half_b = wp.dot(oc, rd)
    c = wp.dot(oc, oc) - radius*radius

    discriminant = half_b * half_b - a * c
    if discriminant >= 0:
      sqrtd = wp.sqrt(discriminant)
      root_a = (-half_b - sqrtd) / a
      root_b = (-half_b + sqrtd) / a
      if root_a < 1e-3:
        root_a = root_b
      
      if not (root_a < 1e-3 or root_a >= intersection.t[rid]):
        intersection.success[rid] = 1
        intersection.t[rid] = root_a
        intersection.p[rid] = ro + root_a * rd
        intersection.normal[rid] = (intersection.p[rid] - center) / radius
        intersection.material_idx[rid] = material_idx
        
        front_face_bool = wp.dot(rd, intersection.normal[rid]) < 0
        intersection.front_face[rid] = 0
        if front_face_bool:
          intersection.front_face[rid] = 1
        if not front_face_bool:
          intersection.normal[rid] = -intersection.normal[rid]

  return True


@wp.func
def apply_material(rd: wp.vec3,
                intersections: Intersection,
                materials: Materials,
                result: ApplyMaterialResult,
                rid: wp.int32, local_state: LocalState):
  material_idx = intersections.material_idx[rid]
  material_type = materials.material_type[material_idx]

  if material_type == LAMBERTIAN_MATERIAL_TYPE: # lambertian
    albedo = materials.albedo[material_idx]
    normal = intersections.normal[rid] + random_unit_vector(local_state)
    if wp_min(wp_abs(normal)) < 1e-8:
      normal = intersections.normal[rid] 
    result.success[rid] = 1
    result.albedo[rid] = albedo
    result.normal[rid] = normal
    result.center[rid] = intersections.p[rid]
  elif material_type == METAL_MATERIAL_TYPE: # metal
    albedo = materials.albedo[material_idx]
    fuzz = materials.fuzz[material_idx]
    r = reflect(unit(rd), unit(intersections.normal[rid]))
    rr = r + fuzz * random_in_unit_sphere(local_state)
    success = wp.dot(r, intersections.normal[rid]) > 0.0
    if success:
      result.success[rid] = 1
    else:
      result.success[rid] = 0

    result.albedo[rid] = albedo
    result.center[rid] = intersections.p[rid]
    result.normal[rid] = rr
  elif material_type == DIELECTRIC_MATERIAL_TYPE: # dielectric
    refractive_index = materials.refractive_index[material_idx]
    color = wp.vec3(1.0, 1.0, 1.0)
    if intersections.front_face[rid] > 0:
      refraction_ratio = 1.0 / refractive_index
    else:
      refraction_ratio = refractive_index

    udir = unit(rd)
    cos_theta = -wp.dot(udir, intersections.normal[rid])
    cos_theta = wp.clamp(cos_theta, -1.0, 1.0)
    sin_theta = wp.sqrt(1.0 - cos_theta*cos_theta)
    cannot_refract = refraction_ratio * sin_theta > 1.0
    if cannot_refract or reflectance(cos_theta, refraction_ratio) > next_random(local_state):
      dir = reflect(udir, intersections.normal[rid])
    else:
      dir = refract(udir, intersections.normal[rid], refraction_ratio)
    result.success[rid] = 1
    result.albedo[rid] = color
    result.center[rid] = intersections.p[rid]
    result.normal[rid] = dir
  else:
    return False
  
  return True

@wp.kernel
def trace_ray(
        spheres: Spheres,
        cam_pos: wp.vec3,
        width: int,
        height: int,
        samples_per_pixel: int,
        max_depth: int,
        uvs: wp.array(dtype=wp.vec2),
        materials: Materials,
        intersections: Intersection,
        local_state: LocalState,
        apply_material_result: ApplyMaterialResult,
        pixels: wp.array(dtype=wp.vec3)):
  
  rid = wp.tid()
 
  uv = uvs[rid]
  max_depth_o = max_depth
  fsamples_per_pixel = wp.float32(samples_per_pixel)
  for _ in range(samples_per_pixel):
    uv_i = uv + random_vec2(local_state)
    x = uv_i[0]
    y = uv_i[1]
    sx = 2.0*float(x)/float(height) - 1.0
    sy = 2.0*float(y)/float(height) - 1.0

    ro = cam_pos
    rd = wp.normalize(wp.vec3(sx, sy, -1.0))

    max_depth = max_depth_o
    color = wp.vec3(1.0/fsamples_per_pixel, 1.0/fsamples_per_pixel, 1.0/fsamples_per_pixel)
    
    while (max_depth > 0):
      intersect_spheres(spheres, ro, rd, intersections, rid)
      if intersections.success[rid] > 0:
        apply_material(rd, intersections, materials, apply_material_result, rid, local_state)
        color_comp = apply_material_result.albedo[rid]
        ro = apply_material_result.center[rid]
        rd = apply_material_result.normal[rid]
        status = apply_material_result.success[rid]
        if status > 0:
          color = wp.vec3(
            color[0] * color_comp[0],
            color[1] * color_comp[1],
            color[2] * color_comp[2]
          )
          max_depth -= 1
        else:
          color = wp.vec3(0.0, 0.0, 0.0)
          max_depth = 0
      else:
        max_depth = 0

    pixels[rid] = pixels[rid] + color
    

class Scene:
  def __init__(self, width=1024, height=800, samples_per_pixel=500, max_depth=50, random_materials=50):
    self.device = wp.get_preferred_device()
    self.cam_pos = wp.vec3(0, 1, 2)
    self.width = width
    self.height = height
    self.samples_per_pixel = samples_per_pixel
    self.max_depth = max_depth
    self.pixels = wp.zeros(width * height, dtype=wp.vec3, device=self.device)

    # uvs
    self.uvs = None
    self.initialize_uvs()

    # materials
    self.materials = None
    self.initialize_materials(random_materials)

    # scene 
    self.scene = None
    self.initialize_scene()

    # initialize outputs
    self.local_state = None
    self.intersections = None
    self.apply_material_results = None
    self.initialize_outputs_memory()

  def initialize_outputs_memory(self):
    size = len(self.uvs)

    inter = Intersection()
    inter.success = wp.zeros(size, dtype=wp.int32, device=self.device)
    inter.t = wp.zeros(size, dtype=wp.float32, device=self.device)
    inter.p = wp.zeros(size, dtype=wp.vec3, device=self.device)
    inter.normal = wp.zeros(size, dtype=wp.vec3, device=self.device)
    inter.material_idx = wp.zeros(size, dtype=wp.int32, device=self.device)
    inter.front_face = wp.zeros(size, dtype=wp.int32, device=self.device)
    self.intersections = inter

    apply_material_result = ApplyMaterialResult()
    apply_material_result.success = wp.zeros(size, dtype=wp.int32, device=self.device)
    apply_material_result.albedo = wp.zeros(size, dtype=wp.vec3, device=self.device)
    apply_material_result.center = wp.zeros(size, dtype=wp.vec3, device=self.device)
    apply_material_result.normal = wp.zeros(size, dtype=wp.vec3, device=self.device)
    self.apply_material_results = apply_material_result

    local_state = LocalState()
    local_state.seed = wp.array(np.random.randint(0, 2**30, size=size), dtype=wp.int32, device=self.device)
    self.local_state = local_state

  def initialize_uvs(self):
    y = np.arange(self.height, dtype=np.float32)
    x = np.arange(self.width, dtype=np.float32)
    y = y.reshape(self.height, 1).repeat(self.width, axis=1)
    x = x.reshape(1, self.width).repeat(self.height, axis=0)
    uv = np.stack((x, y), axis=-1).reshape(-1, 2)
    self.uvs = wp.array(uv, device=self.device, dtype=wp.vec2)

  def initialize_materials(self, N):
    albedos = [wp.vec3(0.5, 0.5, 0.5), wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.4, 0.2, 0.1), wp.vec3(0.7, 0.6, 0.5)]
    material_type = [0, 2, 0, 1]
    refractive_index = [0, 1.5, 0, 0]
    fuzz = [0.0, 0.0, 0.0, 0.0]

    for _ in range(N):
      type = np.random.choice([0, 0, 0, 0, 1, 1, 1, 1, 2])
      material_type.append(type)
      albedos.append(wp.vec3(*np.random.random(size=3)))
      refractive_index.append(1.5)
      fuzz.append(np.random.uniform(0.0, 0.5))
    
    self.materials = Materials()
    self.materials.albedo = wp.array(albedos, dtype=wp.vec3, device=self.device)
    self.materials.material_type = wp.array(material_type, dtype=wp.int32, device=self.device)
    self.materials.refractive_index = wp.array(refractive_index, dtype=wp.float32, device=self.device)
    self.materials.fuzz = wp.array(fuzz, dtype=wp.float32, device=self.device)

  def initialize_scene(self):
    spheres = []
    spheres.append((np.array([0, -1000, 0], dtype=np.float32), 1000, 0))

    for a in range(-11, 11):
      for b in range(-11, 11):
        center = np.array([a + 0.9 * np.random.random(), 0.2, b + 0.9 * np.random.random()], dtype=np.float32)

        distance = np.linalg.norm(center - np.array([4, 0.2, 0], dtype=np.float32))
        if distance > 0.9:
          random_material = np.random.randint(0, len(self.materials.albedo))
          spheres.append((center, 0.2, random_material))

    spheres.append((np.array([0, 1, 0], dtype=np.float32), 1.0, 1))
    spheres.append((np.array([-4, 1, 0], dtype=np.float32), 1.0, 2))
    spheres.append((np.array([4, 1, 0], dtype=np.float32), 1.0, 3))

    self.scene = Spheres()
    self.scene.radius = wp.array([radius for _, radius, _ in spheres], dtype=wp.float32, device=self.device)
    self.scene.center = wp.array([center for center, _, _ in spheres], dtype=wp.vec3, device=self.device)
    self.scene.material_idx = wp.array([material_idx for _, _, material_idx in spheres], dtype=wp.int32, device=self.device)
    self.scene.cnt = len(spheres)

  def render(self):
    with wp.ScopedTimer('Render'):
      wp.launch(
        kernel=trace_ray,
        dim=len(self.uvs),
        inputs=[
          self.scene,
          self.cam_pos,
          self.width,
          self.height,
          self.samples_per_pixel,
          self.max_depth,
          self.uvs,
          self.materials,
          self.intersections,
          self.local_state,
          self.apply_material_results,
          self.pixels
        ],
        device=self.device
      )
      wp.synchronize()

    out_img = self.pixels.numpy().reshape((self.height, self.width, 3))
    out_img = out_img[::-1]
    cv2.imwrite('out.png', out_img * 255.0)

    plt.imshow(out_img, origin="lower", interpolation="antialiased")
    plt.show()

  
if __name__ == '__main__':
  scene = Scene()
  scene.render()
