#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

layout(set = 0, binding = 1) buffer Data {
    uint data[][256][256];
};

layout(push_constant) uniform PushConstants {
    uvec2 resolution;
    vec3 camera_dir;
    vec3 rotation; 
    vec3 position;
    uint render_distance;
} constants;

float sdSphere(vec3 p, float d) { return length(p) - d; } 

float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) +
         length(max(d,0.0));
}
	

uint getVoxel(ivec3 c) {
    if (
        c.x <= 0 || c.x >= data.length() ||
        c.y <= 0 || c.y >= data[0].length() ||
        c.z <= 0 || c.z >= data[0][0].length()
    ) {
        return 0; 
    }
    return data[c.x][c.y][c.z];
}
vec2 rotate2d(vec2 v, float a) {
	float sinA = sin(a);
	float cosA = cos(a);
	return vec2(v.x * cosA - v.y * sinA, v.y * cosA + v.x * sinA);	
}

void main() {
	vec2 screenPos = (gl_GlobalInvocationID.xy / vec2(constants.resolution.x , constants.resolution.y)) * 2.0 - 1.0;
	vec3 cameraPlaneU = vec3(1.0, 0.0, 0.0);
	vec3 cameraPlaneV = vec3(0.0, 1.0, 0.0) * constants.resolution.y / constants.resolution.x;
	vec3 rayDir = constants.camera_dir + screenPos.x * cameraPlaneU + screenPos.y * cameraPlaneV;
	vec3 rayPos = constants.position;

    rayPos.yz = rotate2d(rayPos.yz, constants.rotation.x);
	rayDir.yz = rotate2d(rayDir.yz, constants.rotation.x);
    rayPos.xz = rotate2d(rayPos.xz, constants.rotation.y);
	rayDir.xz = rotate2d(rayDir.xz, constants.rotation.y);
    rayPos.xy = rotate2d(rayPos.xy, constants.rotation.z);
	rayDir.xy = rotate2d(rayDir.xy, constants.rotation.z);
	
	ivec3 mapPos = ivec3(floor(rayPos + 0.));

	vec3 deltaDist = abs(vec3(length(rayDir)) / rayDir);
	
	ivec3 rayStep = ivec3(sign(rayDir));

	vec3 sideDist = (sign(rayDir) * (vec3(mapPos) - rayPos) + (sign(rayDir) * 0.5) + 0.5) * deltaDist; 
	
	bvec3 mask = bvec3(false);

    uint u_voxel = 0;
	
	for (int i = 0; i <= constants.render_distance; i++) {
        uint voxel = getVoxel(mapPos);
		if (voxel != 0) {
            u_voxel = voxel;
            break;
        }
        if (i == constants.render_distance) {
            mask = bvec3(false);
            break;
        }
        if (sideDist.x < sideDist.y) {
            if (sideDist.x < sideDist.z) {
                sideDist.x += deltaDist.x;
                mapPos.x += rayStep.x;
                mask = bvec3(true, false, false);
            }
            else {
                sideDist.z += deltaDist.z;
                mapPos.z += rayStep.z;
                mask = bvec3(false, false, true);
            }
        }
        else {
            if (sideDist.y < sideDist.z) {
                sideDist.y += deltaDist.y;
                mapPos.y += rayStep.y;
                mask = bvec3(false, true, false);
            }
            else {
                sideDist.z += deltaDist.z;
                mapPos.z += rayStep.z;
                mask = bvec3(false, false, true);
            }
        }
	}
	
	vec3 color = vec3(0.1);
    if (mask.x) {
		color = vec3(0.25);
	}
    if (mask.y) {
		color = vec3(0.75);
	}
    if (mask.z) {
		color = vec3(0.5);
	}
    switch(u_voxel) {
        case 1: color.r += 0.25; break;
        case 2: color.g += 0.25; break;
        case 3: color.b += 0.25; break;
        case 4: color *= vec3(0.3, 0.4, 0.5); break;
        case 5: color *= vec3(0.6, 0.3, 0.9); break;
        case 6: color *= vec3(0.1, 0.4, 0.6); break;
        case 7: color *= vec3(0.8, 0.3, 0.6); break;
        case 8: color *= vec3(0.2, 0.9, 0.4); break;
        case 9: color *= vec3(0.1, 0.5, 0.8); break;
    }
    imageStore(img, ivec2(gl_GlobalInvocationID.xy), vec4(color, 1.0));
}
