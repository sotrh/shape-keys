# WGPU shape key / morph target example

This example demonstrates a way to implement morph targets in wgpu. The gltf loading code makes some assumptions based on the model, so a more robust loader would be needed for models that differ in format.

## Uses

The most common use for shape keys is facial animations. While you can animate a persons face using bones, this can be tricky to get just right. If you only need a handful of expressions and/or you want to blend multiple expressions shape keys can give you more fine grained control.

## How it works

Shape keys work by storing the offset of the positions/normals for every vertex than adding to the vertex's position/normal to transform it into the desired position. The shader code for this is pretty simple operates entirely in the vertex shader.

```wgsl
@vertex
fn vs_main(v: Vertex, morphs: Morphs) -> VsData {
    var normal = v.normal;
    var position = v.position;

    normal += morphs.d_normal0 * uniforms.weights.x;
    normal += morphs.d_normal1 * uniforms.weights.y;
    normal = normalize(normal);

    position += morphs.d_position0 * uniforms.weights.x;
    position += morphs.d_position1 * uniforms.weights.y;
    
    let frag_pos = uniforms.mvp * vec4(position, 1.0);
    let eye_norm = (uniforms.normal_mat * vec4(normal, 0.0)).xyz;

    return VsData(frag_pos, eye_norm, normal * 0.5 + 0.5);
}
```
