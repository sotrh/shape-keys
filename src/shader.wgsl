struct Vertex {
    @location(0)
    position: vec3<f32>,
    @location(1)
    normal: vec3<f32>,
}

struct Morphs {
    @location(2)
    d_position0: vec3<f32>,
    @location(3)
    d_normal0: vec3<f32>,
    @location(4)
    d_position1: vec3<f32>,
    @location(5)
    d_normal1: vec3<f32>,
}

struct Uniforms {
    weights: vec4<f32>,
    normal_mat: mat4x4<f32>,
    mvp: mat4x4<f32>,
}

struct VsData {
    @builtin(position)
    frag_pos: vec4<f32>,
    @location(0)
    normal: vec3<f32>,
    @location(1)
    color: vec3<f32>,
}

@group(0)
@binding(0)
var<uniform> uniforms: Uniforms;

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

@fragment
fn fs_main(vd: VsData) -> @location(0) vec4<f32> {
    let ambient = 0.1;

    let l = normalize(vec3<f32>(1.0, 1.0, -1.0));
    let n = normalize(vd.normal);
    let diffuse = clamp(dot(l, n), 0.0, 1.0);

    // return vec4(vd.color * clamp(dot(l, n), 0.0, 1.0) + ambient, 1.0);
    // return vec4(n * 0.5 + 0.5, 1.0);
    return vec4(vec3(diffuse) + ambient, 1.0);
}