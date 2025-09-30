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
      return binomialCoeff(n, i) * pow(t, float(i)) * pow(1.0 - t, float(n - i));
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
      if (colormap == 0) {
        return vec3( sin(6.28318*t), sin(6.28318*t+2.094), sin(6.28318*t+4.188) )*0.5+0.5;
      }
      else if (colormap == 1) {
        return mix( mix(vec3(0,0,1), vec3(1,1,1), t), vec3(1,0,0), t );
      }
      else if (colormap == 2) {
        return vec3( clamp(3.0*t,0.0,1.0),
                     clamp(3.0*t-1.0,0.0,1.0),
                     clamp(3.0*t-2.0,0.0,1.0) );
      }
      else {
        return vec3(t);
      }
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
