use image::{RgbImage, Rgb};

use metal::*;

// the size of the output render
static MAX_LIGHT_BOUNCES: u32 = 128;
static WIDTH: u32 = 1080;// * 2;
static HEIGHT: u32 = 720;// * 2;

static OUTPUT_RENDER_FILE_NAME: &str = "output.png";


// this has to align with the version defined in msl
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Object {
    // the extra padding is needed for metal's memory layout to actually align.......
    point_a: [f32; 4],
    point_b: [f32; 4],
    point_c: [f32; 4],
    surface_normal: [f32; 4],
    object_id: u32,
    
    _padding: [f32; 3],
    
    material: Material,
}

impl Object {
    pub fn new(point_a: (f32, f32, f32), point_b: (f32, f32, f32), point_c: (f32, f32, f32), object_id: u32, material: Material) -> Self {
        let surface_normal = triangle_normal(point_a, point_b, point_c);
        Self { point_a: [point_a.0, point_a.1, point_a.2, 0.], point_b: [point_b.0, point_b.1, point_b.2, 0.],
            point_c: [point_c.0, point_c.1, point_c.2, 0.], surface_normal: [surface_normal.0, surface_normal.1, surface_normal.2, 0.],
            object_id, material, _padding: [0f32; 3],
        }
    }
    
    // just returns an empty instance
    pub fn null () -> Self {
        Self {
            point_a: [0., 0., 0., 0.],
            point_b: [0., 0., 0., 0.],
            point_c: [0., 0., 0., 0.],
            surface_normal: [0., 0., 0., 0.],
            object_id: 0,
            _padding: [0f32; 3],
            material: Material::new(
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                false,
                (0., 0., 0.),
                (0., 0., 0.),
            ),
        }
    }
}

#[derive(Clone, Copy)]
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
    
    is_volume: bool,
}

impl Material {
    pub fn new (roughness: f32, absorption: f32, transmittance: f32,
                index_of_refraction: f32, specularity: f32, scattering: f32,
                g: f32, is_volume: bool, emission: (f32, f32, f32), color: (f32, f32, f32)
    ) -> Self {
        Material {
            roughness, absorption, transmittance, specularity,
            index_of_refraction, scattering, scattering_type: g,
            emission: [emission.0, emission.1, emission.2, 0.],
            color: [color.0, color.1, color.2, 0.], is_volume,
            _padding: [0f32; 1],
        }
    }
}

type Vec3 = (f32, f32, f32);

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

fn sub_l (a: [f32; 4], b: [f32; 4]) -> (f32, f32, f32) {
    (a[0] - b[0], a[1] - b[1], a[2] - b[2])
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

fn cube (center_x: f32, center_y: f32, center_z: f32, size_x: f32, size_y: f32, size_z: f32, material: Material, id: u32) -> Vec<Object> {
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

fn create_mesh <const SIZE_X: usize, const SIZE_Y: usize> (material: Material, id: u32, start: (f32, f32, f32), size: (f32, f32), height_map: [[f32; SIZE_X]; SIZE_Y]) -> Vec<Object> {
    let mut mesh = vec![];
    let cell_size = (
        size.0 / (SIZE_X-1) as f32,
        size.1 / (SIZE_Y-1) as f32,
    );
    
    for cell_x in 0..SIZE_X-1 {
        for cell_y in 0..SIZE_Y-1 {
            let position_top_left = (
                cell_size.0 * cell_x as f32 + start.0,
                start.1 + height_map[cell_y as usize][cell_x as usize],
                cell_size.1 * cell_y as f32 + start.2,
            );
            let position_top_right = (
                cell_size.0 * (cell_x + 1) as f32 + start.0,
                start.1 + height_map[cell_y as usize][cell_x as usize + 1],
                cell_size.1 * cell_y as f32 + start.2,
            );
            let position_bottom_left = (
                cell_size.0 * cell_x as f32 + start.0,
                start.1 + height_map[cell_y as usize + 1][cell_x as usize],
                cell_size.1 * (cell_y + 1) as f32 + start.2,
            );
            let position_bottom_right = (
                cell_size.0 * (cell_x + 1) as f32 + start.0,
                start.1 + height_map[cell_y as usize + 1][cell_x as usize + 1],
                cell_size.1 * (cell_y + 1) as f32 + start.2,
            );
            mesh.push(Object::new(
                position_top_left, position_top_right, position_bottom_right, id, material.clone()
            ));
            mesh.push(Object::new(
                position_top_left, position_bottom_left, position_bottom_right, id, material.clone()
            ));
        }
    } mesh
}

fn generate_height_map <const SIZE_X: usize, const SIZE_Y: usize> () -> [[f32; SIZE_X]; SIZE_Y] {
    let mut table = [[0.0f32; SIZE_X]; SIZE_Y];
    for x in 0..SIZE_X {
        for y in 0..SIZE_Y {
            table[y][x] = f32::sin(x as f32 / SIZE_X as f32 * 48. + y as f32 * 0.9) * 0.4;
        }
    } table
}

#[repr(C, align(16))]
pub struct Light {
    index: u32,
    area_weight: f32,
    area: f32,
}


// stores raw pointers to make this and the gpu end structs line up in size without specifying anything
#[repr(C, align(16))]
struct PointedBVH <const NODE_STACK_SIZE_MULTIPLE_OF_FOUR: usize> {
    object_indexes: *const [u32; NODE_STACK_SIZE_MULTIPLE_OF_FOUR],
    children_aabb: *const [[f32; 4]; 2],
    children: *const [u32; 4],
    objects: *const Object,
}

// generates the bvh for the scene
#[repr(C, align(16))]
struct BoundingVolumeHierarchy<
    const NODE_STACK_SIZE_MULTIPLE_OF_FOUR: usize,  // if this is aligned to 16 bytes (multiple of four), everything else should automatically be aligned as each member is by default 16 bytes
> {
    // each index corresponds to the child's index     stores the index to the children
    pub object_indexes: Box<[[u32; NODE_STACK_SIZE_MULTIPLE_OF_FOUR]]>,
    // each index corresponds to the child's index (the bounding box; index 1 == position, index 2 == size)   the fourth dimension is just for padding
    // min, max
    pub children_aabb: Box<[[[f32; 4]; 2]]>,
    // binary tree's have 2 children each (indexes to the next child)    the extra two elements are for padding
    // all 0s means it's a leaf with no children
    pub children: Box<[[u32; 4]]>,
    // all the objects (the object struct is already padded)
    pub objects: Box<[Object]>,
}

impl BvhStructure {
    // simply returns a structure containing pointers to each member to simplify the memory layout on the gpu end
    pub fn as_ptr (&self) -> PointedBVH<NODE_STACK_SIZE_MULTIPLE_OF_FOUR> {
        PointedBVH {
            object_indexes: self.object_indexes.as_ptr(),
            children_aabb: self.children_aabb.as_ptr(),
            children: self.children.as_ptr(),
            objects: self.objects.as_ptr(),
        }
    }
    
    // takes ownership of Object to consume it while pushing it into the tree
    pub fn new (mut objects: Vec<Object>) -> Self {
        let mut tree = BvhStructure {
            object_indexes: vec![[0u32; NODE_STACK_SIZE_MULTIPLE_OF_FOUR]; NUM_NODES].into_boxed_slice(),
            children_aabb: vec![[[0.; 4]; 2]; NUM_NODES].into_boxed_slice(),
            children: vec![[0; 4]; NUM_NODES].into_boxed_slice(),
            objects: vec![Object::null(); NUM_OBJECTS].into_boxed_slice(),
        };
        
        // tracking the node index (the top of the buffer/stack)
        let mut current_stack_index = 1;  // 0 is the root node, so it should start at 1
        // creating the root node (just a basic bounding box, nothing fancy)
        let (min_bound, max_bound) = Self::get_bounds(&objects);
        tree.children_aabb[0] = [[min_bound.0, min_bound.1, min_bound.2, 0.], [max_bound.0, max_bound.1, max_bound.2, 0.]];
        // the children have to be the next two (makes it easy for this iteration
        //tree.children[0] = [1, 2, 0, 0];
        // the root node will not have any geometry unless pushed into it by the splitting process (which would happen in the recursive section)
        
        // taking the objects and pushing them into bins
        // calling a recursive function to start splitting the nodes into individual bins slowly
        // objects will only be pushed to the internal array when they're confirmed in the place in a bin (allowing easy splitting but also without cloning at all)
        
        // either splits the space or concludes the child position as a leaf node
        // each split half takes its respective objects and continues the recursion
        let mut object_count = 0;
        Self::get_recursive_nodes(&mut tree, &mut current_stack_index, objects, (min_bound, max_bound), 1, &mut object_count);
        
        tree
    }
    
    fn get_recursive_nodes (tree: &mut Self,
                            current_stack_index: &mut usize,
                            objects: Vec<Object>,
                            (min_bound, max_bound): ((f32, f32, f32), (f32, f32, f32)),
                            depth: u32,
                            object_count: &mut usize
    ) {
        // assigning the node it's aabb (bounding box)
        tree.children_aabb[*current_stack_index][0] = [min_bound.0, min_bound.1, min_bound.2, 0.];
        tree.children_aabb[*current_stack_index][1] = [max_bound.0, max_bound.1, max_bound.2, 0.];
        
        if depth >= BVH_DEPTH || objects.len() < MIN_OBJECT_SPLIT_COUNT {
            //println!("Depth: {} Objects: {}", depth, objects.len());
            // create a node at current_stack_index (including the bounding box; which is the same as the inputted min/max)
            // save all objects into the current position
            // increment the counter
            
            // leaf nodes don't really have any special parameters so setup should only include objects (and aabb)
            let mut index = 0;
            for object in objects {  // consumes the bin
                tree.object_indexes[*current_stack_index][index] = *object_count as u32;
                tree.objects[*object_count] = object;
                *object_count += 1;
                index += 1;
            }
            *current_stack_index += 1;
            
            return;
        }
        // saving the current stack index - 1 as parent_index
        let parent_index = *current_stack_index - 1;
        
        // splitting the volume in three bins (order: left, right, middle)
        let (left_bin, right_bin, middle_bin) = Self::split_volume(objects, (&min_bound, &max_bound));
        //println!("Split node #{} with {} / {} objects and {} in the middle.", current_stack_index, left_bin.len(), right_bin.len(), middle_bin.len());
        
        // taking all triangles caught inbetween and placing them into parent_index
        let mut index = 0;
        for object in middle_bin {  // consumes the bin
            tree.object_indexes[parent_index][index] = *object_count as u32;
            tree.objects[*object_count] = object;
            *object_count += 1;
            index += 1;
        }
        
        // creating a left child at current_stack_index and incrementing the counter
        // updating the first child in parent_index with the stack index before incrementing
        tree.children[parent_index][0] = *current_stack_index as u32;
        *current_stack_index += 1;
        let bounds = Self::get_bounds(&left_bin);
        Self::get_recursive_nodes(tree, current_stack_index, left_bin, bounds, depth + 1, object_count);
        
        
        // creating a right child at current_stack_index and incrementing the counter
        // updating the second child in parent_index with the stack index before incrementing
        // with that child, calling the recursive function again
        tree.children[parent_index][1] = *current_stack_index as u32;
        *current_stack_index += 1;
        let bounds = Self::get_bounds(&right_bin);
        Self::get_recursive_nodes(tree, current_stack_index, right_bin, bounds, depth + 1, object_count);
        
        // concluded
    }
    
    // left, right, overlapping    is the order of return vectors
    fn split_volume (objects: Vec<Object>, (min_bound, max_bound): (&(f32, f32, f32), &(f32, f32, f32))) -> (Vec<Object>, Vec<Object>, Vec<Object>) {
        // tracking the best point to split at
        let mut min_cost = f32::MAX;
        let mut split_point = ((0., 0., 0.), (0., 0., 0.));
        
        // looping over a set number of triangles to find the approximate best cost
        // max of 1000 objects to check (hopefully not too slow?)
        let count = usize::min(1000, objects.len());
        let ratio = objects.len() as f32 / count as f32;
        for i in 0..count {
            if min_cost < MIN_COST_BREAK {  break;  }
            // getting the approximate index to avoid too much bias when indexing large meshes
            let index = (i as f32 * ratio) as usize;
            let (point_a, point_b, point_c) = (&objects[index].point_a, &objects[index].point_b, &objects[index].point_c);
            // finding the farthest point to properly split the space without cutting the triangle in half (reducing variance some hopefully resulting in better costs overall)
            let max_triangle = (
                f32::max(f32::max(point_a[0], point_b[0]), point_c[0]) + 0.001,  // a small offset to prevent position checks from overlapping it
                f32::max(f32::max(point_a[1], point_b[1]), point_c[1]) + 0.001,  // a small offset to prevent position checks from overlapping it
                f32::max(f32::max(point_a[2], point_b[2]), point_c[2]) + 0.001,  // a small offset to prevent position checks from overlapping it
            );
            // getting the "cost" of the split
            let cost = (
                Self::split_cost(
                    &objects,
                    (min_bound, &(max_triangle.0, max_bound.1, max_bound.2)),
                    (&(max_triangle.0, min_bound.1, min_bound.2), max_bound),
                    (min_bound, max_bound)
                ),
                Self::split_cost(
                    &objects,
                    (min_bound, &(max_bound.0, max_triangle.1, max_bound.2)),
                    (&(min_bound.0, max_triangle.1, min_bound.2), max_bound),
                    (min_bound, max_bound)
                ),
                Self::split_cost(
                    &objects,
                    (min_bound, &(max_bound.0, max_bound.1, max_triangle.2)),
                    (&(min_bound.0, min_bound.1, max_triangle.2), max_bound),
                    (min_bound, max_bound)
            ));
            
            //println!("    * Checking triangle at index #{} with the max point at {:?} and costs of {:?}", index, max_triangle, cost);
            
            // updating the best point to split at
            let min_axis_cost = f32::min(f32::min(cost.0, cost.1), cost.2);
            if min_axis_cost >= min_cost {  continue;  }
            if cost.0 <= min_axis_cost + 0.001 {
                min_cost = cost.0;
                split_point = ((max_triangle.0, min_bound.1, min_bound.2), (max_triangle.0, max_bound.1, max_bound.2));
                continue;
            }
            if cost.1 <= min_axis_cost + 0.001 {
                min_cost = cost.1;
                split_point = ((min_bound.0, max_triangle.1, min_bound.2), (max_bound.0, max_triangle.1, max_bound.2));
                continue;
            }
            if cost.2 <= min_axis_cost + 0.001 {
                min_cost = cost.2;
                split_point = ((min_bound.0, min_bound.1, max_triangle.2), (max_bound.0, max_bound.1, max_triangle.2));
                continue;
            }
        }
        
        let box_area =
            (max_bound.0 - min_bound.0 * (max_bound.1 - min_bound.1)) * 2. +
            (max_bound.0 - min_bound.0)* (max_bound.2 - min_bound.2) * 2. +
            (max_bound.1 - min_bound.1) * (max_bound.2 - min_bound.2) * 2.;
        
        // taking the best split and finding all objects for each bin
        let mut bin_left = Vec::with_capacity(objects.len() / 2);
        let mut bin_right = Vec::with_capacity(objects.len() / 2);
        let mut bin_middle = Vec::with_capacity(16);  // this hopefully won't have very many
        // now consuming the objects vector to place them into their respective containers
        //println!("Split volume; bounding boxes: {:?}, {:?} and {:?}, {:?} costing: {}", min_bound, &split_point, &split_point, max_bound, min_cost);
        for object in objects {
            let center = [
                (object.point_a[0] + object.point_b[0] + object.point_c[0]) / 3.,
                (object.point_a[1] + object.point_b[1] + object.point_c[1]) / 3.,
                (object.point_a[2] + object.point_b[2] + object.point_c[2]) / 3.,
                0.
            ];
            // swapping between placement options based on the size of the triangle (tiny ones can overhang, but giant ones need to be centered
            let placement = {
                let triangle_area = 0.5 * length(cross(sub_l(object.point_b, object.point_a), sub_l(object.point_c, object.point_a)));
                if (triangle_area / box_area) < 0.01 || triangle_area < 0.05 {
                    Self::check_bin_placement(
                        [&center, &center, &center],
                        (min_bound, &split_point.1),
                        (&split_point.0, max_bound),
                    )
                } else {
                    Self::check_bin_placement(
                        [&object.point_a, &object.point_b, &object.point_c],
                        (min_bound, &split_point.1),
                        (&split_point.0, max_bound),
                    )
                }
            };
            match placement {
                (0, 0, 0) => {&mut bin_left},
                (1, 1, 1) => {&mut bin_right},
                _ => {&mut bin_middle},
            }.push(object);
        } (bin_left, bin_right, bin_middle)
    }
    
    fn check_bounds (point: &[f32; 4], (min_bound, max_bound): (&(f32, f32, f32), &(f32, f32, f32))) -> bool {
        point[0] >= min_bound.0 && point[0] <= max_bound.0 && point[1] >= min_bound.1 && point[1] <= max_bound.1 && point[2] >= min_bound.2 && point[2] <= max_bound.2
    }
    
    // 0 == left, 1 == right, 2 == error
    fn check_bin_placement (points: [&[f32; 4]; 3],
                            (min_bound_left, max_bound_left): (&(f32, f32, f32), &(f32, f32, f32)),
                            (min_bound_right, max_bound_right): (&(f32, f32, f32), &(f32, f32, f32)),
    ) -> (u8, u8, u8) {
        let placement1 = {
            match (
                Self::check_bounds(points[0], (min_bound_left, max_bound_left)),
                Self::check_bounds(points[0], (min_bound_right, max_bound_right)),
            ) {
                (true, false) => {0},
                (false, true) => {1},
                _ => {2}
            }
        };
        let placement2 = {
            match (
                Self::check_bounds(points[1], (min_bound_left, max_bound_left)),
                Self::check_bounds(points[1], (min_bound_right, max_bound_right)),
            ) {
                (true, false) => {0},
                (false, true) => {1},
                _ => {2}
            }
        };
        let placement3 = {
            match (
                Self::check_bounds(points[2], (min_bound_left, max_bound_left)),
                Self::check_bounds(points[2], (min_bound_right, max_bound_right)),
            ) {
                (true, false) => {0},
                (false, true) => {1},
                _ => {2}
            }
        };
        (placement1, placement2, placement3)
    }
    
    fn split_cost (objects: &Vec<Object>,
                   (min_bound_left, max_bound_left): (&(f32, f32, f32), &(f32, f32, f32)),
                   (min_bound_right, max_bound_right): (&(f32, f32, f32), &(f32, f32, f32)),
                   (min_bound_parent, max_bound_parent): (&(f32, f32, f32), &(f32, f32, f32))
    ) -> f32 {
        // generating the surface areas for each bounding box
        let mut total_area_left = 0.;
        let mut total_area_right = 0.;
        let mut total_area_middle = 0.;
        
        let mut left_count = 0;
        let mut right_count = 0;
        let mut middle_count = 0;
        
        for object in objects {
            // going through each object, finding which bin it belongs in, and adding it's surface area respectively
            let placement = Self::check_bin_placement(
                [&object.point_a, &object.point_b, &object.point_c],
                (min_bound_left, max_bound_left),
                (min_bound_right, max_bound_right),
            );
            let triangle_area = 0.5 * length(cross(sub_l(object.point_b, object.point_a), sub_l(object.point_c, object.point_a)));
            match placement {
                (0, 0, 0) => {total_area_left += triangle_area; left_count += 1;},
                (1, 1, 1) => {total_area_right += triangle_area; right_count += 1;},
                _ => {total_area_middle += triangle_area; middle_count += 1;},
            }
        }
        
        // getting the sizes of the bounding boxes
        let size_left = (
            max_bound_left.0 - min_bound_left.0,
            max_bound_left.1 - min_bound_left.1,
            max_bound_left.2 - min_bound_left.2,
        );
        let size_right = (
            max_bound_right.0 - min_bound_right.0,
            max_bound_right.1 - min_bound_right.1,
            max_bound_right.2 - min_bound_right.2,
        );
        let size_parent = (
            max_bound_parent.0 - min_bound_parent.0,
            max_bound_parent.1 - min_bound_parent.1,
            max_bound_parent.2 - min_bound_parent.2,
        );
        
        //println!("      - Split with {} units^2 on the left, {} units^2 on the right, and {} units^2 in the middle.", total_area_left, total_area_right, total_area_middle);
        
        // getting the surface area
        let surface_area_left = (size_left.0 * size_left.1 * 2.) + (size_left.1 * size_left.2 * 2.) + (size_left.0 * size_left.2 * 2.);  // x*y*2 + y*z*2 + x*z*2
        let surface_area_right = (size_right.0 * size_right.1 * 2.) + (size_right.1 * size_right.2 * 2.) + (size_right.0 * size_right.2 * 2.);
        let surface_area_parent = (size_parent.0 * size_parent.1 * 2.) + (size_parent.1 * size_parent.2 * 2.) + (size_parent.0 * size_parent.2 * 2.);
        
        if left_count == 0 || right_count == 0 {  return f32::INFINITY;  }
        
        const K: f32 = 0.25;  // the weight for triangles that are caught between
        // calculating the final cost
        surface_area_left / surface_area_parent * total_area_left + surface_area_right / surface_area_parent * total_area_right + K * total_area_middle + f32::abs(left_count as f32 - right_count as f32) * 10.
        // sleft / sparent * aleft + sright / sparent * aright + k(a const between 1 and 2 to taste) * amiddle
    }
    
    // gets the minimum and maximum bounds for a given set of objects
    fn get_bounds (objects: &Vec<Object>) -> ((f32, f32, f32), (f32, f32, f32)) {
        let mut min = (f32::MAX, f32::MAX, f32::MAX);
        let mut max = (f32::MIN, f32::MIN, f32::MIN);
        
        for object in objects {
            min.0 = f32::min(f32::min(min.0, object.point_a[0]), f32::min(object.point_b[0], object.point_c[0]));
            min.1 = f32::min(f32::min(min.1, object.point_a[1]), f32::min(object.point_b[1], object.point_c[1]));
            min.2 = f32::min(f32::min(min.2, object.point_a[2]), f32::min(object.point_b[2], object.point_c[2]));
            
            max.0 = f32::max(f32::max(max.0, object.point_a[0]), f32::max(object.point_b[0], object.point_c[0]));
            max.1 = f32::max(f32::max(max.1, object.point_a[1]), f32::max(object.point_b[1], object.point_c[1]));
            max.2 = f32::max(f32::max(max.2, object.point_a[2]), f32::max(object.point_b[2], object.point_c[2]));
        }
        (min, max)
    }
}

// Suppose these are the constants you want to fix for convenience
const BVH_DEPTH: u32 = 24;
const NUM_OBJECTS: usize = 250000;
const NODE_STACK_SIZE_MULTIPLE_OF_FOUR: usize = 32;  // make sure to update the version in metal defined at the top
const MIN_OBJECT_SPLIT_COUNT: usize = 16;
const MIN_COST_BREAK: f32 = 100.;

const NUM_NODES: usize = (u32::pow(2, BVH_DEPTH + 1) - 1) as usize;
// Create a type alias for BVH with those generics
const _: [(); 0] = [(); (NODE_STACK_SIZE_MULTIPLE_OF_FOUR % 4 == 0) as usize - 1];
type BvhStructure = BoundingVolumeHierarchy<NODE_STACK_SIZE_MULTIPLE_OF_FOUR>;

fn main() {
    let device = Device::system_default().expect("No Metal device found");
    
    // Write shader source as a raw string
    let src = std::fs::read_to_string("shaders/shader.metal")
        .expect("Failed to read shader file");
    
    // Input data
    /*let base_scene = vec![
        // floor
        Object::new((-3., -2., -2.), (3., -2., -2.),(3., -2., -25.), 1,
            Material::new(0.0, 0., 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0., 0., 0.))),
        Object::new((-3., -2., -2.), (-3., -2., -25.),(3., -2., -25.), 1,
            Material::new(0.0, 0., 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0., 0., 0.))),
        
        // ceiling
        Object::new((-3., 12., -2.), (3., 12., -2.),(3., 12., -25.), 1,
            Material::new(1., 0.5, 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (2., 0.0, 0.0))),
        Object::new((-3., 12., -2.), (-3., 12., -25.),(3., 12., -25.), 1,
            Material::new(1., 0.5, 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0.9, 0.9, 0.9))),
        
        // backlight
        Object::new((-3., -2., -15.), (-3., 12., -15.),(3., 12., -15.), 1,
            Material::new(1., 1., 0., 1.003, 0., 0., 0., false, (8. * 2.5, 8. * 2.5, 8. * 2.5), (0.9, 0.9, 0.9))),
        Object::new((-3., -2., -15.), (3., -2., -15.),(3., 12., -15.), 1,
            Material::new(1., 1., 0., 1.003, 0., 0., 0., false, (8. * 2.5, 8. * 2.5, 8. * 2.5), (0.9, 0.9, 0.9))),
        
        // shield for back wall
        Object::new((-3., -2., -15.1), (-3., 12., -15.1),(3., 12., -15.1), 1,
            Material::new(0., 1., 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0., 0., 0.))),
        Object::new((-3., -2., -15.1), (3., -2., -15.1),(3., 12., -15.1), 1,
            Material::new(0., 1., 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0., 0., 0.))),
        Object::new((-3., -2., -20.), (-3., 12., -20.),(3., 12., -20.), 1,
            Material::new(0., 1., 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0., 0., 0.))),
        Object::new((-3., -2., -20.), (3., -2., -20.),(3., 12., -20.), 1,
            Material::new(0., 1., 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0., 0., 0.))),
        
        // wall left
        Object::new((-3., -2., -2.), (-3., -2., -15.),(-3., 12., -15.), 1,
            Material::new(0.8, 0.2, 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0.9, 0.2, 0.7))),
        Object::new((-3., -2., -2.), (-3., 12., -2.),(-3., 12., -15.), 1,
            Material::new(0.3, 0.3, 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0.9, 0.3, 0.7))),
        
        // wall left for rays
        Object::new((-3., -2., -15.7), (-3., -2., -35.),(-3., 12., -35.), 1,
            Material::new(0.8, 1., 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0.9, 0.2, 0.7))),
        Object::new((-3., -2., -15.7), (-3., 12., -15.7),(-3., 12., -35.), 1,
            Material::new(0.3, 1., 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0.9, 0.3, 0.7))),

        // wall right
        Object::new((3., -2., -2.), (3., -2., -25.),(3., 12., -25.), 1,
            Material::new(1., 0.3, 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0.9, 0.9, 0.9))),
        Object::new((3., -2., -2.), (3., 12., -2.),(3., 12., -25.), 1,
            Material::new(0.8, 0.4, 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0.9, 0.9, 0.9))),
        
        // front panel
        Object::new((-3., -2., -2.), (-3., 12., -2.),(3., 12., -2.), 2,
            Material::new(0.01, 0.9, 1., 1.4, 0., 0., 0., false, (0., 0., 0.), (0.9, 0., 0.))),
        Object::new((-3., -2., -8.), (3., -2., -8.),(3., 12., -8.), 2,
            Material::new(0., 0.8, 1., 3.0, 0., 0., 0., false, (0., 0., 0.), (0., 0., 0.))),
        
        // back-light behind things
        Object::new((-2., 9., -16.), (4., 9., -16.),(4., 1., -20.), 1,
            Material::new(1., 1., 0., 1.003, 0., 0., 0., false, (700., 700., 700.), (1., 1., 1.))),
        Object::new((-2., 9., -16.), (-2., 1., -20.),(4., 1., -20.), 1,
            Material::new(1., 1., 0., 1.003, 0., 0., 0., false, (700., 700., 700.), (1., 1., 1.))),
    ];
    let starting_volume = Material::new(
        0., 0.1, 1., 1.003, 0., 0.1, 0.8, true, (0., 0., 0.), (0., 0., 0.)
    );
    
    let input = vec![
        base_scene,
        cube(0., 4., 1.5, 6., 6., 2.25, Material::new(
            0., 1., 1., 1.3, 0., 0., 0., true, (0., 0., 0.), (0., 0., 0.)
        ), 3),
        // making everything slightly foggy
        // 1 / σt, or ∫ over [0, 1) of -ln(ξ)/σt, is the average distance a ray of light can travel through the medium. For 0.01 (scatter * absorption e.g. 0.2 * 0.05), the mean distance is 100.
        //cube(0., 0., 0., 75., 25., 75., Material::new(
        //    0., 0.1, 1., 1.001, 0., 0.001, 0., true, (0., 0., 0.), (0.5, 0.5, 0.5)
        //), 4),
        
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
    ].concat();*/
    let total_frames = 2500;
    let starting_volume = Material::new(0., 0., 0., 1.003, 0., 0., 0., true, (0., 0., 0.), (0., 0., 0.));
    let wall_mat = Material::new(
        0.9, 0.4, 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0.9, 0.9, 1.)
    );
    let floor_mat = Material::new(
        0.9, 0.4, 0., 1.003, 0., 0., 0., false, (0., 0., 0.), (0.7, 0.9, 1.)
    );
    let light_map = Material::new(
        0., 1., 1., 1.003, 0., 0., 0., false, (500., 500., 500.), (0.9, 0.9, 1.)
    );
    let input = vec![
        // pool: -6, -15 (slopes to -20 as z -> -5), -25 -> 6, -5, -5
        vec![  // creating the walls of the swimming pool
            // north wall
           Object::new((-6., -15., -25.), (-6., -5., -25.), (6., -5., -25.), 1, wall_mat.clone()),
           Object::new((-6., -15., -25.), (6., -15., -25.), (6., -5., -25.), 1, wall_mat.clone()),
           // southern wall
           Object::new((-6., -20., 8.), (-6., -5., 8.), (6., -5., 8.), 1, wall_mat.clone()),
           Object::new((-6., -20., 8.), (6., -20., 8.), (6., -5., 8.), 1, wall_mat.clone()),
           
           // west wall
           Object::new((-6., -5., -25.), (-6., -5., 8.), (-6., -20., 8.), 1, wall_mat.clone()),
           Object::new((-6., -5., -25.), (-6., -15., -25.), (-6., -20., 8.), 1, wall_mat.clone()),
           // eastern wall
           Object::new((6., -5., -25.), (6., -5., 8.), (6., -20., 8.), 1, wall_mat.clone()),
           Object::new((6., -5., -25.), (6., -15., -25.), (6., -20., 8.), 1, wall_mat.clone()),
           
           // floor
           Object::new((-6., -15., -25.), (-6., -20., 8.), (6., -20., 8.), 1, floor_mat.clone()),
           Object::new((-6., -15., -25.), (6., -15., -25.), (6., -20., 8.), 1, floor_mat.clone()),
           
           // temporary light
           Object::new((-1., 20., -5.), (-1., 20., -7.5), (1.5, 20., -7.5), 1, light_map.clone()),
           Object::new((-1., 20., -5.), (1.5, 20., -5.), (1.5, 20., -7.5), 1, light_map.clone()),
        ],
        create_mesh::<256, 256>(
            Material::new(
                0., 0.8, 1., 1.333, 0., 0., 0., false, (0., 0., 0.), (0., 0.25, 0.35)
            ),
            2,
            (-6., -6., -25.),
            (12., 33.),
            // 26 x 26
            generate_height_map::<256, 256>(),
        ),
    ].concat();
    
    // generating the bvh
    let vector = input.clone();  // temporary while i still use the whole object vector
    println!("Constructing BVH for {} objects", vector.len());
    let start = std::time::Instant::now();
    let bvh = BvhStructure::new(vector);
    println!("Finished Constructing BVH in {} seconds", start.elapsed().as_secs_f64());
    
    // calculating all lights, their areas, and averaging
    // emmisive_triangle_area / total_emmisive_triangles_combined_area
    let mut total_area = 0f32;
    let mut light_sources: Vec<Light> = vec![];
    let mut light_source_indexes = vec![];
    for (index, object) in input.iter().enumerate() {
        if object.material.emission.iter().sum::<f32>() > 0.01 {
            // calculating the area
            let triangle_area = 0.5 * length(cross(sub_l(object.point_b, object.point_a), sub_l(object.point_c, object.point_a)));
            light_source_indexes.push((index as u32, triangle_area));
            total_area += triangle_area;
        }
    }
    for (index, area) in light_source_indexes {
        light_sources.push(Light {
            index, area_weight: area / total_area, area,
        });
    }
    println!("Number of detected lights: {}\nTotal surface area: {}", light_sources.len(), total_area);
    
    const COUNT: usize = (WIDTH * HEIGHT) as usize * 3;
    
    let mut final_results = Vec::with_capacity(COUNT);
    for _ in 0..COUNT {
        final_results.push(0f64);
    }
    
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
    
    let command_queue = device.new_command_queue();
    
    let in_buf = device.new_buffer_with_data(
        input.as_ptr() as *const _,
        (input.len() * size_of::<Object>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (COUNT * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let width_buf = device.new_buffer_with_data(
        &WIDTH as *const _ as *const _,
        size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let height_buff = device.new_buffer_with_data(
        &HEIGHT as *const _ as *const _,
        size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_size_buf = device.new_buffer_with_data(
        &input.len() as *const _ as *const _,
        size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mut frame = 0u32;
    let frame_count_buf = device.new_buffer_with_data(
        &frame as *const _ as *const _,
        size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let starting_volume_buf = device.new_buffer_with_data(
        &starting_volume as *const _ as *const _,
        size_of::<Material>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let lights_buf = device.new_buffer_with_data(
        &starting_volume as *const _ as *const _,
        size_of::<Light>() as u64 * light_sources.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let num_lights_buf = device.new_buffer_with_data(
        &light_sources.len() as *const _ as *const _,
        size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    
    let frame_start = std::time::Instant::now();
    while frame < total_frames {
        println!("Beginning iteration: {}", frame);
        
        let ptr = frame_count_buf.contents() as *mut u32;
        unsafe { *ptr = frame; }
        
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&in_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_buffer(2, Some(&width_buf), 0);
        encoder.set_buffer(3, Some(&height_buff), 0);
        encoder.set_buffer(4, Some(&input_size_buf), 0);
        encoder.set_buffer(5, Some(&frame_count_buf), 0);
        encoder.set_buffer(6, Some(&starting_volume_buf), 0);
        encoder.set_buffer(7, Some(&lights_buf), 0);
        encoder.set_buffer(8, Some(&num_lights_buf), 0);
        
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
                final_results[base_index] += r as f64;
                final_results[base_index + 1] += g as f64;
                final_results[base_index + 2] += b as f64;
            }
        }
        
        if frame % 200 == 0 {
            let mut img = RgbImage::new(WIDTH, HEIGHT);
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    //let (r, g, b) = sample_pixel_color_from_camera((x, y));
                    let base_index = ( y * WIDTH + x) as usize * 3;
                    let rgb = &final_results[base_index..base_index + 3];
                    let (r, g, b) =  (
                        f64::min(tone_map_uncharted2(rgb[0] / frame as f64) * 255., 255.) as u8,
                        f64::min(tone_map_uncharted2(rgb[1] / frame as f64) * 255., 255.) as u8,
                        f64::min(tone_map_uncharted2(rgb[2] / frame as f64) * 255., 255.) as u8
                    );
                    img.put_pixel(x, HEIGHT - 1 - y, Rgb([r, g, b]));
                }
            }
            
            img.save(OUTPUT_RENDER_FILE_NAME).unwrap();
        }
        
        frame += 1
    }
    let end = frame_start.elapsed().as_secs_f64();
    println!("Time: {}\nPer Ray AVG: {}", end, end / total_frames as f64);
    
    let mut img = RgbImage::new(WIDTH, HEIGHT);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            //let (r, g, b) = sample_pixel_color_from_camera((x, y));
            let base_index = ( y * WIDTH + x) as usize * 3;
            let rgb = &final_results[base_index..base_index + 3];
            let (r, g, b) =  (
                f64::min(tone_map_uncharted2(rgb[0] / total_frames as f64) * 255., 255.) as u8,
                f64::min(tone_map_uncharted2(rgb[1] / total_frames as f64) * 255., 255.) as u8,
                f64::min(tone_map_uncharted2(rgb[2] / total_frames as f64) * 255., 255.) as u8
            );
            img.put_pixel(x, HEIGHT - 1 - y, Rgb([r, g, b]));
        }
    }
    
    img.save(OUTPUT_RENDER_FILE_NAME).unwrap();
}

fn tone_map_uncharted2(x: f64) -> f64 {
    const A: f64 = 0.15;
    const B: f64 = 0.50;
    const C: f64 = 0.10;
    const D: f64 = 0.20;
    const E: f64 = 0.02;
    const F: f64 = 0.30;
    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
}

