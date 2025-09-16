// camera positioning and rotations
#define CAMERA_DIR_X 1.5708 * 0.  // facing forwards at a flat angle (angle 0 is to the right in trigonometry)
#define CAMERA_DIR_Y 1.3     // facing forwards at a flat angle (angle 0 is to the right in trigonometry)

#define CAMERA_POS_X 0.0     // facing forwards at a flat angle (angle 0 is to the right in trigonometry)
#define CAMERA_POS_Y 10.0     // facing forwards at a flat angle (angle 0 is to the right in trigonometry)
#define CAMERA_POS_Z 0.0     // facing forwards at a flat angle (angle 0 is to the right in trigonometry)

// camera view settings
#define FOV 1.57079632679           // 90º in radians
#define FOCAL_POINT 20.0      //
#define APERTURE 0.14         // the amount of shifting that's applied to the offset generated for the focal point

#define STACK_SIZE 8  // overflows are protected against, however it would result in artifacts and incorrect results; lower values may improve performance though
#define MAX_BOUNCES 8

#define NODE_STACK_SIZE_MULTIPLE_OF_FOUR 16  // this needs to be equal to the rust end
#define MAX_BVH_STACK_DEPTH 32  // needs to be 2 * max_depth to ensure no overflows, or lower with risk for performance

// ================================================ Random Number Generators and Helper Functions ================================================

// branchless minimum of a and b
inline float MinBranchless(float a, float b) {
    return a < b ? a : b;
}

// branchless maximum of a and b
inline float MaxBranchless(float a, float b) {
    return 0.5 * (a + b + metal::abs(a - b));
}

inline uint Hash32 (uint x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

// Convert to float [0,1)
inline float Rand (thread uint &state) {
    state = Hash32 (state);
    return (state & 0xFFFFFF) / float(0x1000000); // 24-bit fraction
}

inline float2 RandomPointInUnitDisk (thread uint &state) {
    float r = metal::sqrt(Rand(state));
    float theta = 2.0 * M_PI_F * Rand(state);
    return float2(r * metal::cos(theta), r * metal::sin(theta));
}

// ================================================ Defining the Memory Layout for Objects and Their Properties ================================================

struct Light {
    uint index;
    float area_weight;
    float area;
};

struct Material {
    float  roughness;  // represents how much incoming light it scattered (0 -> perfect mirror; 1 -> diffuse/matt surface
    float  absorption;  // how much light the material absorbs (the surface could be rough but reflect all light, or just be rough, or smooth but dull)
    float  transmittance;  // how much light will transmit through (maybe offset based on the roughness? or just blended)
    float  index_of_refraction;
    float  specularity;
    float  scattering;
    float  scattering_type;
    float3 emission;
    float3 color;
    bool is_volume;
};

struct Object {
    float3 point_a;
    float3 point_b;
    float3 point_c;
    float3 surface_normal;
    uint   object_id;
    Material material;
};

// ================================================ Gathering the Initial Ray Information ================================================

// generates the view angle based on the pixel coordinate
inline void GetViewRayDirection (uint2 gid, uint2 size, thread uint &state, thread float3 &direction, thread float3 &position) {
    // getting the uv coord (-1 to 1)
    // (pos - size/2) / (size / 2)
    float2 float_size = float2(size.x, size.y) * float2(0.5, 0.5);
    float2 uv_coord = (float2(gid.x, gid.y) - float_size) / float_size;

    // if the rotation code here and stuff doesn't work, blame chatGPT (I can't bother with the math)

    // basic pinhole camera ray in camera space
    float3 dir = metal::normalize(float3(uv_coord.x * metal::tan(FOV * 0.5),
                                  uv_coord.y * metal::tan(FOV * 0.5),
                                  -1.0));

    // apply yaw (around Y) and pitch (around X)
    float cy = metal::cos(CAMERA_DIR_X), sy = metal::sin(CAMERA_DIR_X);
    float cx = metal::cos(CAMERA_DIR_Y), sx = metal::sin(CAMERA_DIR_Y);

    // rotation matrices combined
    float3 rotated = float3(
        cy * dir.x + sy * dir.z,
        sx * (cy * dir.z - sy * dir.x) + cx * dir.y,
        cx * (cy * dir.z - sy * dir.x) - sx * dir.y
    );
    direction = metal::normalize(rotated);

    // calculating the focal point and depth of field

    float3 camera_vector;
    camera_vector.x = metal::cos(CAMERA_DIR_X) * metal::sin(CAMERA_DIR_Y);
    camera_vector.y = metal::sin(CAMERA_DIR_X);
    camera_vector.z = -metal::cos(CAMERA_DIR_X) * metal::cos(CAMERA_DIR_Y);
    camera_vector = metal::normalize(camera_vector);

    // getting the focal point and aperture offsets
    float3 world_up = float3(0.0, 1.0, 0.0); // always Y-up
    float3 camera_right = metal::normalize(metal::cross(world_up, camera_vector));
    float3 camera_up    = metal::cross(camera_vector, camera_right);

    float2 disk = RandomPointInUnitDisk(state);
    float3 lensOffset = camera_right * disk.x + camera_up * disk.y;

    float3 focal_point = float3(CAMERA_POS_X, CAMERA_POS_Y, CAMERA_POS_Z) + direction * float3(FOCAL_POINT, FOCAL_POINT, FOCAL_POINT);

    // offsetting the base position and calculating the angle to it
    position = float3(CAMERA_POS_X, CAMERA_POS_Y, CAMERA_POS_Z) + lensOffset * APERTURE;
    direction = metal::normalize(focal_point - position);
}

// ================================================ Collisions ================================================

inline void RayIntersectsTriangle_Branchless (
    thread const float3 &origin,
    thread const float3 &direction,
    float3 A,
    float3 B,
    float3 C,
    thread float &t_out,
    thread float3 &position_out,
    thread float &mask // 1.0 if intersecting, 0.0 otherwise
) {
    const float EPSILON = 0.0001;

    float3 edge1 = B - A;
    float3 edge2 = C - A;
    float3 h = metal::cross(direction, edge2);
    float det = metal::dot(edge1, h);

    float invDet = 1.0 / (det + (det == 0.0 ? EPSILON : 0.0)); // avoid division by zero

    float3 s = origin - A;
    float u = metal::dot(s, h) * invDet;
    float3 q = metal::cross(s, edge1);
    float v = metal::dot(direction, q) * invDet;
    float t = metal::dot(edge2, q) * invDet;

    // branchless masks
    float mask_det = metal::step(EPSILON, metal::abs(det));
    float mask_u = metal::step(0.0, u) * metal::step(u, 1.0);
    float mask_v = metal::step(0.0, v) * metal::step(u + v, 1.0);
    float mask_t = metal::step(EPSILON, t);

    mask = mask_det * mask_u * mask_v * mask_t;

    // compute intersection point only if mask != 0
    position_out = origin + direction * t * mask;
    t_out = t * mask;
}

inline void CheckCollisions (constant Object* objects,
                             constant uint& num_objects,
                             thread bool& collided,
                             thread uint& nearest_index,
                             thread float& nearest_distance,
                             thread float3& nearest_collision,
                             thread float3& position,
                             thread float3& direction
) {
    for (uint object_index = 0; object_index < num_objects; object_index++) {
        Object object = objects[object_index];
        float mask;
        float distance_to_collision;  // distance to collision
        float3 collision_position;
        RayIntersectsTriangle_Branchless(position, direction, object.point_a, object.point_b, object.point_c, distance_to_collision, collision_position, mask);
        bool collision_at_point = mask >= 0.99;
        collided = collided || collision_at_point;
        nearest_index = distance_to_collision < nearest_distance && collision_at_point ? object_index : nearest_index;
        nearest_collision = distance_to_collision < nearest_distance && collision_at_point ? collision_position : nearest_collision;
        nearest_distance = distance_to_collision < nearest_distance && collision_at_point ? distance_to_collision : nearest_distance;
    }
}

inline float3 SampleTrianglePoint(constant float3& A, constant float3& B, constant float3& C, thread uint &state) {
    float u = Rand(state);
    float v = Rand(state);

    float su = metal::sqrt(u);   // square root warp
    float b0 = 1.0 - su;
    float b1 = su * (1.0 - v);
    float b2 = su * v;

    return b0 * A + b1 * B + b2 * C;
}

// ================================================ BVH Stuff :( ================================================

// WHY can't I just pass this as a single pointer without breaking it up 20 times, and why can't I use multi-dimensional arrays?
struct BVH {
    constant uint *object_indexes;  // * NODE_STACK_SIZE_MULTIPLE_OF_FOUR
    constant uint *num_objects;
    constant float4 *children_aabb;  // * 2
    constant uint4 *children;
    constant Object *objects;
};

// slab test
inline bool CollideAABB (
    thread float3 &position,
    thread float3 &inverse_direction,
    thread float3 &box_min,
    thread float3 &box_max,
    thread float &out_distance
) {
    // Slab test
    float3 t0 = (box_min - position) * inverse_direction;
    float3 t1 = (box_max - position) * inverse_direction;

    float3 tmin3 = metal::min(t0, t1);
    float3 tmax3 = metal::max(t0, t1);

    // largest near, smallest far
    float tNear = metal::max(metal::max(tmin3.x, tmin3.y), tmin3.z);
    float tFar  = metal::min(metal::min(tmax3.x, tmax3.y), tmax3.z);

    out_distance   = metal::max(tNear, 0.);
    return (tNear <= tFar) && (tFar >= 0.0);
}

inline void RayIntersectionBVH (
    thread float3 &position,
    thread float3 &direction,
    thread float &out_distance,
    thread float3 &out_position,
    thread uint &out_index,  // the index of the object
    thread bool &collided,
    thread BVH *bvh,
    thread uint& hits
) {
    float3 inverse_direction = float3(1., 1., 1.) / direction;  // the inverse direction is used a lot and division is slow
    uint index_stack[MAX_BVH_STACK_DEPTH];
    uint stack_index = 0;

    // starting with the root
    float distance;
    float distance2;
    float3 box_min = bvh->children_aabb[0].xyz;
    float3 box_max = bvh->children_aabb[1].xyz;
    bool hit = CollideAABB(position, inverse_direction, box_min, box_max, distance);
    stack_index = hit ? 1 : 0;
    index_stack[0] = 0;  // 0 is the index for the root   (adding regardless, but the previous will cut the loop early if no collision; branchless!)

    out_distance = 99999999.;  // making it a large number to check minimums against

    // iterating
    Object object;
    float mask;
    bool child_1;
    bool child_2;
    float collision_distance;
    float3 collision_position;
    while (stack_index > 0) {
        //hits++;
        // popping off the stack
        stack_index--;
        uint index = index_stack[stack_index];

        // checking if it's a leaf node, and if it contains geometry
        uint4 children = bvh->children[index];

        // checking for intersections with any geometry (leaf or not, it could contain something)
        uint num_objects = bvh->num_objects[index];
        for (uint i = 0; i < num_objects; i++) {
            // checking collision with the object
            uint object_index = bvh->object_indexes[index * NODE_STACK_SIZE_MULTIPLE_OF_FOUR + i];
            object = bvh->objects[object_index];

            RayIntersectsTriangle_Branchless(position, direction, object.point_a, object.point_b, object.point_c, collision_distance, collision_position, mask);
            bool collided_at_point = mask >= 0.99 && collision_distance < out_distance;
            //hits += uint(mask >= 0.99);
            out_position = collided_at_point ? collision_position : out_position;
            out_distance = collided_at_point ? collision_distance : out_distance;
            out_index = collided_at_point ? object_index : out_index;
            collided = collided | collided_at_point;
        }

        // if it's a leaf node, continuing, otherwise adding any collided children
        if (children.x + children.y == 0) {  continue;  }

        // checking collisions with the children and adding them in order (farthest in first, closest second)
        // only add children if they are closer (to ensure minimal checks)
        box_min = bvh->children_aabb[children.x * 2    ].xyz;
        box_max = bvh->children_aabb[children.x * 2 + 1].xyz;
        child_1 = CollideAABB(position, inverse_direction, box_min, box_max, distance);
        child_1 = child_1 && distance < out_distance;  // making sure only children closure than the nearest current intersection are grabbed
        box_min = bvh->children_aabb[children.y * 2    ].xyz;
        box_max = bvh->children_aabb[children.y * 2 + 1].xyz;
        child_2 = CollideAABB(position, inverse_direction, box_min, box_max, distance2);
        child_2 = child_2 && distance2 < out_distance;  // making sure only children closure than the nearest current intersection are grabbed
        distance = child_1 ? distance : 999999999.;
        distance2 = child_2 ? distance2 : 999999999.;

        // ordering them
        bool2 hit_children = distance < distance2 ? bool2(child_2, child_1) : bool2(child_1, child_2);
        children = distance < distance2 ? uint4(children.y, children.x, 0, 0) : uint4(children.x, children.y, 0, 0);
        // adding each child (if they're hit)    please don't take some of these comments out of context.....
        // this is all branchless (or at least should mostly be)!
        index_stack[stack_index] = hit_children.x ? children.x : index_stack[stack_index];
        stack_index = hit_children.x ? stack_index + 1 : stack_index;
        index_stack[stack_index] = hit_children.y ? children.y : index_stack[stack_index];
        stack_index = hit_children.y ? stack_index + 1 : stack_index;
    }
}

// ================================================ Collision Sampling ================================================

inline float FresnelSchlickAngle(float cosTheta, float R0) {
    return R0 + (1.0 - R0) * metal::pow(1.0 - cosTheta, 5.0);
}

// Generates a cosine-weighted random direction over the hemisphere aligned with a normal
inline float3 CosineWeightedRandomHemisphere(float3 normal, thread uint &state) {
    float r1 = Rand(state);
    float r2 = Rand(state);
    float phi = 2.0 * M_PI_F * r1;
    float cosTheta = metal::sqrt(1.0 - r2);
    float sinTheta = metal::sqrt(r2);

    // local space sample
    float3 sample = float3(metal::cos(phi) * sinTheta, metal::sin(phi) * sinTheta, cosTheta);

    // create tangent space aligned with normal
    float3 up = metal::abs(normal.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
    float3 tangent = metal::normalize(metal::cross(up, normal));
    float3 bitangent = metal::cross(normal, tangent);

    // transform sample from tangent to world space
    return tangent * sample.x + bitangent * sample.y + normal * sample.z;
}

inline float3 SampleHG_World(float3 wi, float g, thread uint &state) {
    // Random numbers
    float xi1 = Rand(state);  // [0,1)
    float xi2 = Rand(state);  // [0,1)

    // Branchless cos(theta) computation
    float invG = 1.0 / g;
    float sqrTerm = (1.0 - g*g) / (1.0 - g + 2.0 * g * xi1);
    float cosTheta_aniso = (1.0 + g*g - sqrTerm*sqrTerm) * 0.5 * invG;

    // Isotropic case
    float cosTheta_iso = 1.0 - 2.0 * xi1;

    // Blend based on g != 0
    float gNonZero = metal::step(1e-3, metal::abs(g));       // 0 if g ~ 0, 1 if g != 0
    float cosTheta = gNonZero * cosTheta_aniso + (1.0 - gNonZero) * cosTheta_iso;

    // Spherical coordinates
    float sinTheta = metal::sqrt(metal::max(0.0, 1.0 - cosTheta*cosTheta));
    float phi = 2.0 * M_PI_F * xi2;

    // Local direction
    float3 localDir = float3(sinTheta * metal::cos(phi), sinTheta * metal::sin(phi), cosTheta);

    // Build orthonormal basis around wi
    float3 w = metal::normalize(wi);
    float3 a = metal::abs(w.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
    float3 v = metal::normalize(metal::cross(a, w));
    float3 u = metal::cross(w, v);

    // Rotate localDir to world space
    float3 worldDir = localDir.x * u + localDir.y * v + localDir.z * w;
    return worldDir;
}

inline void ScatterRays (thread bool &scattered, constant Material *last_material, thread const float &nearest_distance, thread uint &state, thread float3 &direction, thread float3 &position) {
    // t = −ln(ξ)/σt    ξ is [0, 1) random; σt is the scattering * absorption. (The distance to the next reflection is necessary, so put it in another function)
    float random_unit_state = Rand(state);
    float ot = last_material->scattering + last_material->absorption;
    //ot = ot < 0.001 ? 0.001 : ot;  // making sure it isn't zero incase something weird happens
    float scatter_distance = -metal::log(random_unit_state) / ot;

    // checking if the maximum distance is greater or not (if so scattering, otherwise continuing)
    scattered = scatter_distance < nearest_distance && last_material->scattering > 0.;
    float3 new_position = position + direction * float3(scatter_distance, scatter_distance, scatter_distance);
    position = scattered && last_material->scattering > 0. ? new_position : position;  // updating the position to be at the position calculated to be scattering
    float3 scattered_direction = SampleHG_World(direction, last_material->scattering_type, state);
    direction = scattered && last_material->scattering > 0. ? scattered_direction : direction;
}

inline float Checked (float numeral) {
    return metal::abs(numeral) < 0.00001 ? 0.00001 : numeral;
}

inline float GetWeightedMIS (constant Material* material, thread float3 &direction, thread float3 &direction_to_light, thread float3 &surface_normal, float distance_to_light) {
    // volumetric calculations
    bool is_volume = material->scattering > 0. ? 1. : 0.;
    float distance_attenuation = metal::exp(-(material->scattering + material->absorption) * distance_to_light);
    float gg = material->scattering_type*material->scattering_type;
    float greenstein_div = 4.*M_PI_F*metal::pow(1. + gg - 2. * material->scattering_type * metal::dot(direction, direction_to_light), 2. / 3.);
    float greenstein_factor = (1. - gg) / Checked(greenstein_div * is_volume * distance_attenuation);

    float3 h = (direction + direction_to_light) / Checked(metal::length(direction + direction_to_light));
    float surface_term1 = (1. - material->specularity) * metal::max(0., metal::dot(surface_normal, direction_to_light)) / M_PI_F;
    float h_norm = metal::dot(h, surface_normal);
    float asqur = (material->roughness*material->roughness);
    float factor = (h_norm * h_norm) * (asqur - 1.) + 1.;
    float denom_of_surf = M_PI_F * (factor*factor);
    float numerator = asqur / Checked(denom_of_surf);
    float surface_term2 = material->specularity * (numerator * h_norm) / Checked(4. * metal::dot(direction, h));
    float surface = surface_term1 + surface_term2;
    return surface * (1. - is_volume) + greenstein_factor;
}

// index_of_refraction_ration represents: current_medium_ior / ior_for_medium_being_entered
inline void BounceRay (thread float3 &direction,
                       thread float3 &position,
                       constant float3 &normal,
                       constant Object &object,
                       thread uint &state,
                       thread uint &index_for_refraction_stack,
                       thread float* indexes_of_refraction,
                       thread uint* object_ids,
                       constant Material** materials,
                       constant Material* current_material
) {
    Material material = object.material;

    // this should allow any object to be hit from any side and still correctly return the right value
    float sign = -metal::sign(metal::dot(normal, direction));  // making sure the normal is align regardless of which side is hit
    float3 surface_normal = normal * float3(sign, sign, sign);

    // getting the index of refraction
    float index_of_refraction_ration = material.index_of_refraction / indexes_of_refraction[index_for_refraction_stack];

    // calculating the reflection direction (also account for the fresnel coefficient)
    float3 reflected = metal::reflect(direction, surface_normal);

    float cosTheta = metal::dot(-direction, surface_normal);
    float reflectRatio = FresnelSchlickAngle(cosTheta, 1 - material.absorption);

    // calculating the refracted direction
    float3 refracted = metal::refract(direction, surface_normal, index_of_refraction_ration);
    refracted = metal::normalize(refracted * (1.0 - material.roughness) + CosineWeightedRandomHemisphere(refracted, state) * (material.roughness));

    float random_unit_state = Rand(state);
    // transparency:  0 -> opaque, 1 -> perfectly clear
    float3 unit_scattering_direction = CosineWeightedRandomHemisphere(surface_normal, state);

    bool refracted_condition = random_unit_state < material.transmittance;
    float3 refracted_direction = refracted_condition ? refracted : unit_scattering_direction;

    // choosing between the reflected and scattered vectors based on a random state and reflectivity
    random_unit_state = Rand(state);

    // a value of 0 in the refracted direction indicates a total internal reflection
    bool condition = random_unit_state < reflectRatio || (metal::abs(refracted.x) < 0.001 && metal::abs(refracted.y) < 0.001 && metal::abs(refracted.z) < 0.001);
    reflected = metal::normalize(reflected * (1.0 - material.roughness) + unit_scattering_direction * (material.roughness));
    direction = condition ? reflected : refracted_direction;
    bool condition_of_final_refraction = condition || !refracted_condition;
    float error_margin = condition_of_final_refraction ? 0.001 : -0.001;
    //current_index_of_refraction = condition || !refracted_condition ? current_index_of_refraction : new_index;
    position += surface_normal * float3(error_margin, error_margin, error_margin);  // making sure it's not actually on the object to prevent errors (when refracting it needs to do the opposite)

    // updating the refraction stack based on the material/object ids
    // (!condition_of_final_refraction && !condition && cur_id != id)   if the ray is entering a new material -> psh (new ior && object id)
    // (!condition_of_final_refraction && !condition && cur_id == id)   if the material == the current stack material -> pop (no new ior or id)
    // (condition_of_final_refraction)   if the ray isn't entering/exiting anything (reflecting/diffusing) -> nothing (no new ior or id)

    bool need_to_psh = !condition_of_final_refraction && !condition && object.object_id != object_ids[index_for_refraction_stack] && material.is_volume;
    bool need_to_pop = !condition_of_final_refraction && !condition && !need_to_psh && material.is_volume;
    bool nothing_needed = !need_to_psh && !need_to_pop;

    // calculating the new index for the stack
    index_for_refraction_stack = (index_for_refraction_stack + 1)*(need_to_psh) + (index_for_refraction_stack - 1)*(need_to_pop) + index_for_refraction_stack*(nothing_needed);
    index_for_refraction_stack = MinBranchless(STACK_SIZE - 1, index_for_refraction_stack);
    // updating the ior and id
    object_ids[index_for_refraction_stack] = need_to_psh ? object.object_id : object_ids[index_for_refraction_stack];
    indexes_of_refraction[index_for_refraction_stack] = need_to_psh ? material.index_of_refraction : indexes_of_refraction[index_for_refraction_stack];
    materials[index_for_refraction_stack] = need_to_psh ? current_material : materials[index_for_refraction_stack];
}

// ================================================ Tracking the Ray ================================================

// why does chatgpt lie so much?
inline void CheckForLights (thread float3& start_position,
                            thread float3& direction,
                            thread float3& brightness,
                            thread float& transmission,
                            constant Material* material,
                            constant float3& normal,
                            constant Light* lights,
                            constant uint& num_lights,
                            thread BVH* bvh,
                            thread uint& state,
                            thread uint& hits
) {
    // correcting the direction of the surface normal
    float sign = -metal::sign(metal::dot(direction, normal));
    float3 surface_normal = normal * float3(sign, sign, sign);
    // making sure there aren't any floating point precision errors
    float3 position = start_position + surface_normal * float3(0.001, 0.001, 0.001);

    float random_unit_state = Rand(state);
    uint random_index = uint(random_unit_state * num_lights);
    constant Light* light = &lights[random_index];
    constant Object* light_object = &bvh->objects[light->index];

    float3 light_position_sample = SampleTrianglePoint(light_object->point_a, light_object->point_b, light_object->point_c, state);
    float3 direction_to_light = metal::normalize(light_position_sample - position);

    // going through and checking for a collision
    bool collided = false;
    uint nearest_index = 0;
    float3 nearest_collision = float3(0., 0., 0.);
    float nearest_distance = 9999999.;  // big number; hopefully bigger than any real object's would have
    //CheckCollisions(objects, num_objects, collided, nearest_index, nearest_distance, nearest_collision, position, direction_to_light);
    RayIntersectionBVH(position, direction, nearest_distance, nearest_collision, nearest_index, collided, bvh, hits);

    float pdf = GetWeightedMIS(material, direction, direction_to_light, surface_normal, nearest_distance);
    float weight = light->area_weight * (1. / (nearest_distance * nearest_distance));
    weight /= metal::max(pdf, 0.0001);
    float3 final_emission = light_object->material.emission * float3(transmission, transmission, transmission) * float3(weight, weight, weight);
    brightness += collided && nearest_index != light->index || metal::isnan(weight) ? float3(0., 0., 0.) : final_emission;
}

inline uint TraceRay (
               thread   float3        &color,
               thread   float3        &direction,
               thread   float3        &position,
               thread   BVH           *bvh,
               //constant uint&          num_objects,
               thread   uint          &state,
               constant Material*      starting_volume,  // raw pointers on gpu's are just so wonderful...
               constant Light*         lights,
               constant uint          &num_lights
) {
    // color represents the color accumulation
    float transmission = 1.;  // transmission represents how much of the color can make it back to the camera
    float3 brightness = float3(0., 0., 0.);  // brightness represents how strong of light can actually make it back to the camera (brightness * color_acc = final color; color_acc += col * transmission)

    float indexes_of_refraction[STACK_SIZE];  // acting as a stack; hopefully nothing will overflow
    constant Material* materials[STACK_SIZE];
    materials[0] = starting_volume;
    indexes_of_refraction[0] = 1.0003;  // index 0 would be air/null
    uint index_for_refraction_stack = 0;

    uint object_ids[STACK_SIZE];  // acting as a stack; hopefully nothing will overflow
    object_ids[0] = 0;  // id 0 would be air/null
    //uint last_object_id_index = 0;  // this stack *should* line up with that of the refractive indexes

    uint hits = 0;

    for (uint depth = 0; depth < MAX_BOUNCES; depth++) {
        // emission sources should have an absorption of 100%; this prevents further light from accumulating while avoiding branch diversion from an early exit

        // going through and checking for a collision
        bool collided = false;
        uint nearest_index = 0;
        float3 nearest_collision = float3(0., 0., 0.);
        float nearest_distance = 9999999.;  // big number; hopefully bigger than any real object's would have
        //CheckCollisions(bvh, num_objects, collided, nearest_index, nearest_distance, nearest_collision, position, direction);
        RayIntersectionBVH(position, direction, nearest_distance, nearest_collision, nearest_index, collided, bvh, hits);

        float sun_angle_portion = metal::dot(metal::normalize(float3(0.4, 1.2, 0.3)), direction) * 0.5 + 0.5;
        float clamped = (sun_angle_portion > 1. ? 1. : sun_angle_portion);
        sun_angle_portion = (sun_angle_portion < 0. ? 0. : clamped);
        brightness += (collided ? float3(0., 0., 0.) : float3(4., 4., 5.2)) * float3(transmission * sun_angle_portion, transmission * sun_angle_portion, transmission * sun_angle_portion);
        color += (collided ? float3(0., 0., 0.) : float3(1., 1., 1.)) * float3(transmission * sun_angle_portion, transmission * sun_angle_portion, transmission * sun_angle_portion);

        // first checking if scattering needs to first happen
        bool scattered = false;
        constant Material* last_material = materials[index_for_refraction_stack];
        ScatterRays(scattered, last_material, nearest_distance, state, direction, position);
        brightness += scattered ? last_material->emission * float3(transmission, transmission, transmission) : float3(0., 0., 0.);
        color += scattered ? last_material->color * float3(transmission, transmission, transmission) : float3(0., 0., 0.);
        transmission *= scattered ? (1. - last_material->absorption) : 1.;

        // hopefully this won't cause too much divergence (volumes won't likely be common), but honestly either way the following are going to be redundant (and manually storing the state is complex)
        constant Object &object = bvh->objects[nearest_index];
        // the surf normal shouldn't be inverted as it should be passing through
        float3 corrected_position = position + object.surface_normal * metal::sign(metal::dot(object.surface_normal, direction)) * float3(0.01, 0.01, 0.01);
        position = object.material.scattering > 0. ? corrected_position : nearest_collision;  // in case it doesn't scatter and lands on the surface (preventing floating point errors)
        // checking for a nearby light source (MIS/importance sampling for better/quicker results)
        constant Material* material_collided = scattered ? materials[index_for_refraction_stack] : &bvh->objects[nearest_index].material;
        CheckForLights(position, direction, brightness, transmission, material_collided, object.surface_normal, lights, num_lights, bvh, state, hits);

        float random_unit_state = Rand(state);
        if ((!collided && !scattered) || random_unit_state > transmission) break;  // does seem to improve performance quite a bit in open scenes (enclosed ones won't get any benefits, but shouldn't be hurt)
        transmission = 1.;  // resetting it based on the probability (unsimplified it is transmission / transmission)
        if (scattered || object.material.scattering > 0.) {  continue;  }

        // updating the color based on the object's properties
        color += object.material.color * float3(transmission, transmission, transmission);
        brightness += object.material.emission * float3(transmission, transmission, transmission);
        transmission *= (1. - object.material.absorption * (1. - object.material.transmittance));

        BounceRay(direction, position, object.surface_normal, object, state, index_for_refraction_stack, indexes_of_refraction, object_ids, materials, &bvh->objects[nearest_index].material);
    }
    // brightness represents how much of each light made it back while the color represents the colors of impacted objects along the path of the light
    color *= brightness;// * float3(transmission, transmission, transmission);
    return hits;
}

// ================================================ Main Entry Kernel Function ================================================

inline float3 ToneMap_Uncharted2(float3 x) {
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

kernel void TraceRays (
    device  float*      output         [[ buffer(0 ) ]],
    constant uint&      width          [[ buffer(1 ) ]],
    constant uint&      height         [[ buffer(2 ) ]],
    constant uint&      frame          [[ buffer(3 ) ]],
    constant Material* starting_volume [[ buffer(4 ) ]],
    constant Light*     lights         [[ buffer(5 ) ]],
    constant uint&      num_lights     [[ buffer(6 ) ]],
    device float* debug_buffer         [[ buffer(7 ) ]],
    constant uint *bvh_object_indexes  [[ buffer(8 ) ]],
    constant uint *bvh_num_objects     [[ buffer(9 ) ]],
    constant float4 *bvh_children_aabb [[ buffer(10) ]],
    constant uint4 *bvh_children       [[ buffer(11) ]],
    constant Object *bvh_objects       [[ buffer(12) ]],

    uint2 gid [[ thread_position_in_grid ]]
) {
    BVH bvh;
    bvh.object_indexes = bvh_object_indexes;
    bvh.num_objects    = bvh_num_objects;
    bvh.children_aabb  = bvh_children_aabb;
    bvh.children       = bvh_children;
    bvh.objects        = bvh_objects;

    uint pixel_index = gid.y * width + gid.x;
    uint base_index = pixel_index * 3;

    // the state for random number generation
    uint state = pixel_index * 127 + frame * (width * height * 997);

    float3 ray_direction;
    float3 ray_position;
    GetViewRayDirection(gid, uint2(width, height), state, ray_direction, ray_position);

    float3 color = float3(0., 0., 0.);
    // ray position and float3 will be mutated so future references are not valid
    uint hits = TraceRay(color, ray_direction, ray_position, &bvh, state, starting_volume, lights, num_lights);
    //color = ToneMap_Uncharted2(color);

    // the color channels are between 0 and 1
    output[base_index + 0] = color.x; // R
    output[base_index + 1] = color.y; // G
    output[base_index + 2] = color.z; // B
}
