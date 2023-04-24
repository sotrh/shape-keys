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

struct Instance {
    @location(6)
    weights: vec4<f32>,
    @location(7)
    normal_mat0: vec3<f32>,
    @location(8)
    normal_mat1: vec3<f32>,
    @location(9)
    normal_mat2: vec3<f32>,
    @location(10)
    model_mat0: vec4<f32>,
    @location(11)
    model_mat1: vec4<f32>,
    @location(12)
    model_mat2: vec4<f32>,
    @location(13)
    model_mat3: vec4<f32>,
}

struct Uniforms {
    time: vec4<f32>,
    view: mat4x4<f32>,
    view_proj: mat4x4<f32>,
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
fn vs_main(v: Vertex, morphs: Morphs, instance: Instance) -> VsData {
    var normal = v.normal;
    var position = v.position;

    normal += morphs.d_normal0 * instance.weights.x;
    normal += morphs.d_normal1 * instance.weights.y;
    normal = normalize(normal);

    position += morphs.d_position0 * instance.weights.x;
    position += morphs.d_position1 * instance.weights.y;

    let normal_mat = mat3x3<f32>(
        instance.normal_mat0,
        instance.normal_mat1,
        instance.normal_mat2,
    );

    let model_mat = mat4x4<f32>(
        instance.model_mat0,
        instance.model_mat1,
        instance.model_mat2,
        instance.model_mat3,
    );
    
    // let frag_pos = uniforms.view_proj * vec4(position, 1.0);
    let frag_pos = uniforms.view_proj * model_mat * vec4(position, 1.0);
    let eye_norm = (uniforms.view * vec4(normal_mat * normal, 0.0)).xyz;

    return VsData(frag_pos, eye_norm, instance.normal_mat0 * 0.5 + 0.5);
    // return VsData(frag_pos, eye_norm, instance.weights.xyz);
}

@fragment
fn fs_main(vd: VsData) -> @location(0) vec4<f32> {
    let ambient = 0.1;

    let l = normalize(vec3<f32>(1.0, 1.0, -1.0));
    let n = normalize(vd.normal);
    let diffuse = clamp(dot(l, n), 0.0, 1.0);

    let col = vd.color * diffuse + ambient;

    return vec4(col, 1.0);
}

@group(1)
@binding(0)
var<storage, read_write> instances: array<Instance>;

@compute
@workgroup_size(64)
fn update_instances(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let total = arrayLength(&instances);
    let index = global_invocation_id.x;
    if (index >= total) {
        return;
    }

    var instance = instances[index];

    let angles = asin(instance.weights.xy) + uniforms.time.y;
    instance.weights.x = sin(angles.x) * 0.5 + 0.5;
    instance.weights.y = sin(angles.y) * 0.5 + 0.5;

    instances[index] = instance;
}