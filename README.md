# Path-Tracer-in-MSL
A basic path tracer written in Metal Shader Language.

The path tracer is entirely branchless (no longer true. Another branch was added for volumetrics to avoid manually saving the state, although realistically the calculations would be done either way so hpoefully it won't hurt performance too much; the bvh uses branches too) appart from one early exit from the stepping loop as it has no cost (regardless of if it exits the loop or not, the thread is still simulated for any additional steps) but potential gains in scenes with wide open spaces where the max depth will never be hit.

Supported Materials:
 * Diffuse
 * Glossy
 * Reflective
 * Refractive/transparent (working total internal reflections; theoretically caustics should work, but they haven't been tested)
 * Rough + Reflective/Refractive
 * Volumetric (in theory weighted and unweighted scattering directions depending on parameters)
 * Emissive
 * And probably more that I'm forgetting
 * *Note: currently the specular parameter doesn't actualy do anything (kinda, kinda not)*

Basic importance sampling was added for unoccluded lights within the same medium as the ray.
A BVH is now being generated (the generation is needs some optimizations), and is allowing for much more detailed scenes. Using serde, and bincode, the BVH is also being saved, if enabled, to allow for quicker load times.
