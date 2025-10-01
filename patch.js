import * as THREE from 'three';

export class Patch {
  constructor(data, globalUniforms) {
    this.uDeg = data.u;
    this.vDeg = data.v;
    this.controlPoints = data.cps; // flat array [x,y,z, x,y,z, ...]
    this.field = data.field;       // [ [timestep0...], [timestep1...] ]

    this.globalUniforms = globalUniforms; // { timestep, min_val, max_val }

    this.mesh = this.createMesh();
  }



  createGeometry(res = 16) {
    // Simple parametric geometry in JS (u,v grid), pass barycentric coords to shader
    const geometry = new THREE.BufferGeometry();

    const positions = [];
    const uvs = [];

    for (let i = 0; i <= res; i++) {
      for (let j = 0; j <= res; j++) {
        const u = i / res;
        const v = j / res;
        positions.push(u, v, 0); // fake position, will be displaced in shader
        uvs.push(u, v);
      }
    }

    const indices = [];
    for (let i = 0; i < res; i++) {
      for (let j = 0; j < res; j++) {
        const a = i * (res + 1) + j;
        const b = a + 1;
        const c = a + (res + 1);
        const d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);

    return geometry;
  }

  createMesh() {
    const geometry = this.createGeometry(30);

    const numCPs = (this.uDeg + 1) * (this.vDeg + 1);
    const texData = new Float32Array(numCPs * 4 * this.globalUniforms.num_timesteps);

    for (let t = 0; t < this.globalUniforms.num_timesteps; t++) {
      for (let i = 0; i < numCPs; i++) {
        texData[4 * (t * numCPs + i) + 0] = this.controlPoints[3 * i + 0];
        texData[4 * (t * numCPs + i) + 1] = this.controlPoints[3 * i + 1];
        texData[4 * (t * numCPs + i) + 2] = this.controlPoints[3 * i + 2];
        texData[4 * (t * numCPs + i) + 3] = this.field[t][i];
      }
    }

    const tex = new THREE.DataTexture(
      texData,
      numCPs,
      this.globalUniforms.num_timesteps,
      THREE.RGBAFormat,
      THREE.FloatType
    );
    tex.needsUpdate = true;

    const uniforms = {
      controlTex: { value: tex },
      timestep: this.globalUniforms.timestep,
      min_val: this.globalUniforms.min_val,
      max_val: this.globalUniforms.max_val,
      uDeg: { value: this.uDeg },
      vDeg: { value: this.vDeg },
      colormap: this.globalUniforms.colormap
    };

    const material = new THREE.ShaderMaterial({
  uniforms,
  vertexShader: /* glsl */`
    uniform sampler2D controlTex;
    uniform float timestep;
    uniform int uDeg, vDeg;
    varying vec2 vUv;
    varying float vScalar;

    float binomialCoeff(int n, int k) {
      float res = 1.0;
      for (int i=0; i<k; i++) res *= float(n - i) / float(i + 1);
      return res;
    }
    float bernstein(int i, int n, float t) {
      float coeff = binomialCoeff(n, i);
      float ti = (i == 0) ? 1.0 : pow(t, float(i));
      float omt = (n - i == 0) ? 1.0 : pow(1.0 - t, float(n - i));
      return coeff * ti * omt;
    }

    void main() {
      vUv = uv;
      vec3 pos = vec3(0.0);
      float val = 0.0;
      int idx = 0;

      for (int i=0; i<=16; i++) {
        if (i > uDeg) break;
        float Bu = bernstein(i, uDeg, uv.x);
        for (int j=0; j<=16; j++) {
          if (j > vDeg) break;
          float Bv = bernstein(j, vDeg, uv.y);
          vec4 cp = texelFetch(controlTex, ivec2(idx, int(timestep)), 0);
          pos += cp.rgb * (Bu * Bv);
          val += cp.a * (Bu * Bv);
          idx++;
        }
      }

      vScalar = val; // pass scalar to fragment
      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
  `,
  fragmentShader: /* glsl */`
    uniform float min_val, max_val;
    uniform int colormap;
    varying vec2 vUv;
    varying float vScalar;

    vec3 applyColormap(float t) {
  t = clamp(t, 0.0, 1.0);

  const float stops[5] = float[5](0.0, 0.2, 0.5, 0.8, 1.0);

  // rainbow anchors
  const vec3 rainbow[5] = vec3[5](
    vec3(0.0, 0.0, 1.0),
    vec3(0.0, 1.0, 1.0),
    vec3(0.0, 1.0, 0.0),
    vec3(1.0, 1.0, 0.0),
    vec3(1.0, 0.0, 0.0)
  );

  // cool-to-warm anchors
  const vec3 ctw[5] = vec3[5](
    vec3(60.0/255.0, 78.0/255.0, 194.0/255.0),
    vec3(155.0/255.0, 188.0/255.0, 255.0/255.0),
    vec3(220.0/255.0, 220.0/255.0, 220.0/255.0),
    vec3(246.0/255.0, 163.0/255.0, 133.0/255.0),
    vec3(180.0/255.0,   4.0/255.0,  38.0/255.0)
  );

  // blackbody anchors
  const vec3 bb[5] = vec3[5](
    vec3(0.0, 0.0, 0.0),
    vec3(120.0/255.0,   0.0,        0.0),
    vec3(230.0/255.0,  50.0/255.0,  0.0),
    vec3(1.0,           1.0,        0.0),
    vec3(1.0,           1.0,        1.0)
  );

  // grayscale anchors
  const vec3 gs[5] = vec3[5](
    vec3(0.0, 0.0, 0.0),
    vec3(64.0/255.0, 64.0/255.0, 64.0/255.0),
    vec3(127.0/255.0, 127.0/255.0, 128.0/255.0),
    vec3(191.0/255.0, 191.0/255.0, 191.0/255.0),
    vec3(1.0, 1.0, 1.0)
  );

  // find interval
  int idx = 0;
  for (int k = 0; k < 4; ++k) {
    if (t <= stops[k+1]) { idx = k; break; }
  }
  float alpha = (t - stops[idx]) / (stops[idx+1] - stops[idx]);

  vec3 c0, c1;
  if (colormap == 0) {
    c0 = rainbow[idx]; c1 = rainbow[idx+1];
  } else if (colormap == 1) {
    c0 = ctw[idx]; c1 = ctw[idx+1];
  } else if (colormap == 2) {
    c0 = bb[idx]; c1 = bb[idx+1];
  } else {
    c0 = gs[idx]; c1 = gs[idx+1];
  }

  return mix(c0, c1, alpha);
}



    void main() {
      float tval = (vScalar - min_val) / (max_val - min_val);
      vec3 color = applyColormap(tval);
      gl_FragColor = vec4(color, 1.0);
    }
  `,
  side: THREE.DoubleSide
});

    return new THREE.Mesh(geometry, material);
  }
}
