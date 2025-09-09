/*
try representing rays as a continuous wavefront? Infinite resolution from a finite math equation

the camera sends a ray for each pixel, and where that pixel collides the camera could either 1 or 2
1: uses the wavefront or sampling to find more further collisions and sample their caches
2: simply lookup the object's cache

each object in the scene has a cache. A background task runs through each object and either traces rays
  or wavefront and samples the colliding object's caches. The repetition of this should enable accurate
  but efficient GI.

object caches are represented as a tree allowing for progressive refinement. After a certain number of
  iterations the whole object could get subdivided. Any samples are interpolated. This allows an efficient
  but also quick approximation.
Another option might be to only subdivide regions that have significant change or detail under the assumption
  that non-detailed regions don't need additional caching. The only concern would be not correctly
  identifying a detailed section.
Another issue is ghost lighting. When a new sample is passed it needs to weight that result might higher.
  This should hopefully allow the light to quickly dissipate even though object A is still referencing object
  B before it gets to update it's own cache. My biggest concern would be mirrors. Another option might be
  referencing tracking where a list of objects owning each collision is given to allow for a direct removal
  of those collision on a change (movement or deletion) to prevent that issue. This would likely be
  more costly though and definitely would take up more memory usage.
Each iteration should double the number of bounces in terms of GI (2^iterations). Even if only the past 5
  bounces were stored that'd still be a depth of 16 at which any additional computation has diminishing
  returns. That could also help protect against ghosting, and storing 5 samples per cached point shouldn't
  be too terrible. That could be an option.

Actually, in theory any reflections be handled by the wavefront since it should be restoring the wave
  alongside reflections, refractions, and for specular while accounting for diffuse or colors. Because
  all cached surfaces also use the wavefront they too should be able to handle reflections causing indirect
  light to hit them. Hopefully this can handle all the behavior.


issues:
    view dependent rendering such as reflections would require spherical maps for each point unless a
      better method can be figured out (if each object stores references to others, maybe they could
      jump between each other?)

*/
use image::{RgbImage, Rgb};

use metal::*;

// the size of the output render
static WIDTH: u32 = 1080;// * 2;
static HEIGHT: u32 = 720;// * 2;

static OUTPUT_RENDER_FILE_NAME: &str = "output.png";


// this has to align with the version defined in msl
#[derive(Clone)]
#[repr(C, align(16))]
struct Object {
    // the extra padding is needed for metal's memory layout to actually align.......
    point_a: [f32; 4],
    point_b: [f32; 4],
    point_c: [f32; 4],
    surface_normal: [f32; 4],
    object_id: i32,
    material: Material,
}

impl Object {
    pub fn new(point_a: (f32, f32, f32), point_b: (f32, f32, f32), point_c: (f32, f32, f32), object_id: i32, material: Material) -> Self {
        let surface_normal = triangle_normal(point_a, point_b, point_c);
        Self { point_a: [point_a.0, point_a.1, point_a.2, 0.], point_b: [point_b.0, point_b.1, point_b.2, 0.],
            point_c: [point_c.0, point_c.1, point_c.2, 0.], surface_normal: [surface_normal.0, surface_normal.1, surface_normal.2, 0.],
            object_id, material
        }
    }
}

#[derive(Clone)]
#[repr(C, align(16))]
struct Material {
    roughness: f32,
    absorption: f32,
    transmittance: f32,
    index_of_refraction: f32,
    specularity: f32,
    scattering: f32,
    scattering_type: f32,
    _padding: [f32; 1],         // pad to 16 bytes
    emission: [f32; 4],
    color: [f32; 4],
}

impl Material {
    pub fn new (roughness: f32, absorption: f32, transmittance: f32, index_of_refraction: f32, specularity: f32, scattering: f32, g: f32, emission: (f32, f32, f32), color: (f32, f32, f32)) -> Self {
        Material {
            roughness, absorption, transmittance, specularity, index_of_refraction, scattering, scattering_type: g, emission: [emission.0, emission.1, emission.2, 0.], color: [color.0, color.1, color.2, 0.], _padding: [0.; 1],
        }
    }
}

type Vec3 = (f32, f32, f32);

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

fn length(v: Vec3) -> f32 {
    (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt()
}

fn normalize(v: Vec3) -> Vec3 {
    let len = length(v);
    if len > 0.0 {
        (v.0 / len, v.1 / len, v.2 / len)
    } else {
        (0.0, 0.0, 0.0)
    }
}

/// Compute the triangle normal (normalized).
fn triangle_normal(v0: Vec3, v1: Vec3, v2: Vec3) -> Vec3 {
    let edge1 = sub(v1, v0);
    let edge2 = sub(v2, v0);
    normalize(cross(edge1, edge2))
}

fn cube (center_x: f32, center_y: f32, center_z: f32, size_x: f32, size_y: f32, size_z: f32, material: Material, id: i32) -> Vec<Object> {
    // takes position, size, material, id  pub fn new(point_a: (f32, f32, f32), point_b: (f32, f32, f32), point_c: (f32, f32, f32), object_id: i32, material: Material) -> Self {
    vec![
        // top
        Object::new(
            (center_x - size_x * 0.5, center_y + size_y * 0.5, center_z - size_z * 0.5),
            (center_x - size_x * 0.5, center_y + size_y * 0.5, center_z + size_z * 0.5),
            (center_x + size_x * 0.5, center_y + size_y * 0.5, center_z + size_z * 0.5),
            id, material.clone()
        ),
        Object::new(
            (center_x - size_x * 0.5, center_y + size_y * 0.5, center_z - size_z * 0.5),
            (center_x + size_x * 0.5, center_y + size_y * 0.5, center_z - size_z * 0.5),
            (center_x + size_x * 0.5, center_y + size_y * 0.5, center_z + size_z * 0.5),
            id, material.clone()
        ),
        
        // bottom
        Object::new(
            (center_x - size_x * 0.5, center_y - size_y * 0.5, center_z - size_z * 0.5),
            (center_x - size_x * 0.5, center_y - size_y * 0.5, center_z + size_z * 0.5),
            (center_x + size_x * 0.5, center_y - size_y * 0.5, center_z + size_z * 0.5),
            id, material.clone()
        ),
        Object::new(
            (center_x - size_x * 0.5, center_y - size_y * 0.5, center_z - size_z * 0.5),
            (center_x + size_x * 0.5, center_y - size_y * 0.5, center_z - size_z * 0.5),
            (center_x + size_x * 0.5, center_y - size_y * 0.5, center_z + size_z * 0.5),
            id, material.clone()
        ),
        
        // left
        Object::new(
            (center_x - size_x * 0.5, center_y - size_y * 0.5, center_z - size_z * 0.5),
            (center_x - size_x * 0.5, center_y - size_y * 0.5, center_z + size_z * 0.5),
            (center_x - size_x * 0.5, center_y + size_y * 0.5, center_z + size_z * 0.5),
            id, material.clone()
        ),
        Object::new(
            (center_x - size_x * 0.5, center_y - size_y * 0.5, center_z - size_z * 0.5),
            (center_x - size_x * 0.5, center_y + size_y * 0.5, center_z - size_z * 0.5),
            (center_x - size_x * 0.5, center_y + size_y * 0.5, center_z + size_z * 0.5),
            id, material.clone()
        ),
        
        //right
        Object::new(
            (center_x + size_x * 0.5, center_y - size_y * 0.5, center_z - size_z * 0.5),
            (center_x + size_x * 0.5, center_y - size_y * 0.5, center_z + size_z * 0.5),
            (center_x + size_x * 0.5, center_y + size_y * 0.5, center_z + size_z * 0.5),
            id, material.clone()
        ),
        Object::new(
            (center_x + size_x * 0.5, center_y - size_y * 0.5, center_z - size_z * 0.5),
            (center_x + size_x * 0.5, center_y + size_y * 0.5, center_z - size_z * 0.5),
            (center_x + size_x * 0.5, center_y + size_y * 0.5, center_z + size_z * 0.5),
            id, material.clone()
        ),
        
        // front
        Object::new(
            (center_x - size_x * 0.5, center_y - size_y * 0.5, center_z + size_z * 0.5),
            (center_x + size_x * 0.5, center_y - size_y * 0.5, center_z + size_z * 0.5),
            (center_x + size_x * 0.5, center_y + size_y * 0.5, center_z + size_z * 0.5),
            id, material.clone()
        ),
        Object::new(
            (center_x - size_x * 0.5, center_y - size_y * 0.5, center_z + size_z * 0.5),
            (center_x - size_x * 0.5, center_y + size_y * 0.5, center_z + size_z * 0.5),
            (center_x + size_x * 0.5, center_y + size_y * 0.5, center_z + size_z * 0.5),
            id, material.clone()
        ),
        
        // back
        Object::new(
            (center_x - size_x * 0.5, center_y - size_y * 0.5, center_z - size_z * 0.5),
            (center_x + size_x * 0.5, center_y - size_y * 0.5, center_z - size_z * 0.5),
            (center_x + size_x * 0.5, center_y + size_y * 0.5, center_z - size_z * 0.5),
            id, material.clone()
        ),
        Object::new(
            (center_x - size_x * 0.5, center_y - size_y * 0.5, center_z - size_z * 0.5),
            (center_x - size_x * 0.5, center_y + size_y * 0.5, center_z - size_z * 0.5),
            (center_x + size_x * 0.5, center_y + size_y * 0.5, center_z - size_z * 0.5),
            id, material.clone()
        )
    ]
}

fn main() {
    let device = Device::system_default().expect("No Metal device found");
    
    // Write shader source as a raw string
    let src = std::fs::read_to_string("shaders/shader.metal")
        .expect("Failed to read shader file");
    
    // Input data
    let base_scene = vec![
        // floor
        Object::new((-3., -2., -2.), (3., -2., -2.),(3., -2., -15.), 1,
            Material::new(0.0, 0., 0., 1.003, 0., 0., 0., (0., 0., 0.), (0., 0., 0.))),
        Object::new((-3., -2., -2.), (-3., -2., -15.),(3., -2., -15.), 1,
            Material::new(0.0, 0., 0., 1.003, 0., 0., 0., (0., 0., 0.), (0., 0., 0.))),
        
        // ceiling
        Object::new((-3., 12., -2.), (3., 12., -2.),(3., 12., -15.), 1,
            Material::new(1., 0.5, 0., 1.003, 0., 0., 0., (0., 0., 0.), (2., 0.0, 0.0))),
        Object::new((-3., 12., -2.), (-3., 12., -15.),(3., 12., -15.), 1,
            Material::new(1., 0.5, 0., 1.003, 0., 0., 0., (0., 0., 0.), (0.9, 0.9, 0.9))),
        
        // backlight
        Object::new((-3., -2., -15.), (-3., 12., -15.),(3., 12., -15.), 1,
            Material::new(1., 1., 0., 1.003, 0., 0., 0., (9. * 3., 9. * 3., 9. * 3.), (0.9, 0.9, 0.9))),
        Object::new((-3., -2., -15.), (3., -2., -15.),(3., 12., -15.), 1,
            Material::new(1., 1., 0., 1.003, 0., 0., 0., (9. * 3., 9. * 3., 9. * 3.), (0.9, 0.9, 0.9))),
        
        // wall left
        Object::new((-3., -2., -2.), (-3., -2., -15.),(-3., 12., -15.), 1,
            Material::new(0.8, 0.2, 0., 1.003, 0., 0., 0., (0., 0., 0.), (0.9, 0.2, 0.7))),
        Object::new((-3., -2., -2.), (-3., 12., -2.),(-3., 12., -15.), 1,
            Material::new(0.3, 0.3, 0., 1.003, 0., 0., 0., (0., 0., 0.), (0.9, 0.3, 0.7))),
        
        // wall right
        Object::new((3., -2., -2.), (3., -2., -15.),(3., 12., -15.), 1,
            Material::new(1., 0.3, 0., 1.003, 0., 0., 0., (0., 0., 0.), (0.9, 0.9, 0.9))),
        Object::new((3., -2., -2.), (3., 12., -2.),(3., 12., -15.), 1,
            Material::new(0.8, 0.4, 0., 1.003, 0., 0., 0., (0., 0., 0.), (0.9, 0.9, 0.9))),
        
        // front panel
        Object::new((-3., -2., -2.), (-3., 12., -2.),(3., 12., -2.), 2,
            Material::new(0.01, 0.9, 1., 1.4, 0., 0., 0., (0., 0., 0.), (0.9, 0., 0.))),
        Object::new((-3., -2., -8.), (3., -2., -8.),(3., 12., -8.), 2,
            Material::new(0., 0.8, 1., 3.0, 0., 0., 0., (0., 0., 0.), (0., 0., 0.))),
    ];
    let input = vec![
        base_scene,
        cube(0., 4., 1.5, 6., 6., 2.25, Material::new(
            0., 0.8, 0.9, 1.3, 0., 0., 0., (0., 0., 0.), (1., 1., 1.)
        ), 3),
        // making everything slightly foggy
        // 1 / σt, or ∫ over [0, 1) of -ln(ξ)/σt, is the average distance a ray of light can travel through the medium. For 0.01 (scatter * absorption e.g. 0.2 * 0.05), the mean distance is 100.
        cube(0., 0., 0., 75., 75., 75., Material::new(
            0., 0.5, 1., 1.003, 0., 1.4, 0., (0., 0., 0.), (0., 0., 0.)
        ), 4),
        
        /*cube(0., 0., 25., 550., 550., 1., Material::new(
            0., 0., 0., 1.003, 0., (15., 15., 15.), (0., 0., 0.)
        ), 4),
        
        cube(0., -15., -5., 550., 1., 550., Material::new(
            0., 0., 0., 1.003, 0., (0., 0., 0.), (0.5, 0.5, 0.3)
        ), 5),
        cube(0., 20., -5., 550., 1., 550., Material::new(
            0., 0., 0., 1.003, 0., (0., 0., 0.), (0.3, 0.5, 0.5)
        ), 5),
        
        cube(-30., 0., -5., 1., 550., 550., Material::new(
            0.5, 0.4, 0., 1.003, 0., (0., 0., 0.), (0.3, 0.3, 0.3)
        ), 5),
        cube(30., 0., -5., 1., 550., 550., Material::new(
            0.5, 0.3, 0., 1.003, 0., (0., 0., 0.), (0.3, 0.3, 0.3)
        ), 5),*/
    ].concat();
    const COUNT: usize = (WIDTH * HEIGHT) as usize * 3;
    
    let mut final_results = Vec::with_capacity(COUNT);
    for _ in 0..COUNT {
        final_results.push(0f32);
    }
    let total_frames = 250;
    for frame in 0..total_frames {
        // Compile shader at runtime
        let opts = CompileOptions::new();
        let lib = device.new_library_with_source(&src, &opts)
            .expect("Failed to compile Metal shader");
        
        let func = lib.get_function("TraceRays", None).unwrap();
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&func));
        
        let pipeline_state = device
            .new_compute_pipeline_state(&desc)
            .unwrap();
        
        println!("Beginning iteration: {}", frame);
        let in_buf = device.new_buffer_with_data(
            input.as_ptr() as *const _,
            (input.len() * size_of::<Object>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (COUNT * size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&in_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        let width_buf = device.new_buffer_with_data(
            &WIDTH as *const _ as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        encoder.set_buffer(2, Some(&width_buf), 0);
        
        let width_buf = device.new_buffer_with_data(
            &HEIGHT as *const _ as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        encoder.set_buffer(3, Some(&width_buf), 0);
        
        let width_buf = device.new_buffer_with_data(
            &input.len() as *const _ as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        encoder.set_buffer(4, Some(&width_buf), 0);
        
        let width_buf = device.new_buffer_with_data(
            &frame as *const _ as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        encoder.set_buffer(5, Some(&width_buf), 0);
        
        let starting_volume = Material::new(
            0., 0.05, 1., 1.003, 0., 0.2, 0., (0., 0., 0.), (0.5, 0.5, 0.5)
        );
        let starting_volume_buf = device.new_buffer_with_data(
            &starting_volume as *const _ as *const _,
            size_of::<Material>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        encoder.set_buffer(6, Some(&starting_volume_buf), 0);
        
        
        let grid_size = MTLSize {
            width: WIDTH as NSUInteger,
            height: HEIGHT as NSUInteger,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        encoder.dispatch_threads(grid_size, threadgroup_size);
        
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        let out_ptr = out_buf.contents() as *const f32;
        if out_ptr.is_null() {  continue;  }
        let result = unsafe { std::slice::from_raw_parts(out_ptr, COUNT) };
        
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                //let (r, g, b) = sample_pixel_color_from_camera((x, y));
                let base_index = ( y * WIDTH + x) as usize * 3;
                let rgb = &result[base_index..base_index + 3];
                let (r, g, b) =  (
                    rgb[0],
                    rgb[1],
                    rgb[2]
                );
                final_results[base_index] += r;
                final_results[base_index + 1] += g;
                final_results[base_index + 2] += b;
            }
        }
    }
    
    let mut img = RgbImage::new(WIDTH, HEIGHT);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            //let (r, g, b) = sample_pixel_color_from_camera((x, y));
            let base_index = ( y * WIDTH + x) as usize * 3;
            let rgb = &final_results[base_index..base_index + 3];
            let (r, g, b) =  (
                f32::min(rgb[0] / total_frames as f32 * 255., 255.) as u8,
                f32::min(rgb[1] / total_frames as f32 * 255., 255.) as u8,
                f32::min(rgb[2] / total_frames as f32 * 255., 255.) as u8
            );
            img.put_pixel(x, HEIGHT - 1 - y, Rgb([r, g, b]));
        }
    }
    
    img.save(OUTPUT_RENDER_FILE_NAME).unwrap();
}

