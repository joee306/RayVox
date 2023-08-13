#version 450

layout(location = 0) out vec4 frag_color;

layout(location = 1) flat in ivec2 resolution;

const int MAX_RAY_STEPS = 640;

float sdSphere(vec3 p, float d) { return length(p) - d; } 

float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) +
         length(max(d,0.0));
}
	
bool getVoxel(ivec3 c) {
	vec3 p = vec3(c) + vec3(0.5);
    float d = min(max(-sdSphere(p, 35.0), sdBox(p, vec3(25.0))), -sdSphere(p, 100.0));
    return d < 0.0;
}

vec2 rotate2d(vec2 v, float a) {
	float sinA = sin(a);
	float cosA = cos(a);
	return vec2(v.x * cosA - v.y * sinA, v.y * cosA + v.x * sinA);	
}

void main() {
	vec2 screenPos = (gl_FragCoord.xy / resolution) * 2.0 - 1.0;
	vec3 cameraDir = vec3(0.0, 0.0, 0.8);
	vec3 cameraPlaneU = vec3(1.5, 0.0, 0.0);
	vec3 cameraPlaneV = vec3(0.0, 1.5, 0.0) * resolution.y / resolution.x;
	vec3 rayDir = cameraDir + screenPos.x * cameraPlaneU + screenPos.y * cameraPlaneV;
	vec3 rayPos = vec3(0.0, 0.0, -10.0);
	ivec3 mapPos = ivec3(floor(rayPos + 0.));

	vec3 deltaDist = abs(vec3(length(rayDir)) / rayDir);
	
	ivec3 rayStep = ivec3(sign(rayDir));

	vec3 sideDist = (sign(rayDir) * (vec3(mapPos) - rayPos) + (sign(rayDir) * 0.5) + 0.5) * deltaDist; 
	
	bvec3 mask;
    bool nothing = false;
	
	for (int i = 0; i <= MAX_RAY_STEPS; i++) {
		if (getVoxel(mapPos)) continue; 
        else if (i == MAX_RAY_STEPS) nothing = true;
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
	
	vec3 color = vec3(0.1, 0.1, 0.1);
    if (!nothing) {
        if (mask.x) {
            color = vec3(0.5);
        }
        if (mask.y) {
            color = vec3(1.0);
        }
        if (mask.z) {
            color = vec3(0.75);
        }
    }
	frag_color = vec4(color, 1.0);
}
