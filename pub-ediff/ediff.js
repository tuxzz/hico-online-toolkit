import { HTMLBindable } from "./hico-htmlbindable.js";
import { hico_assert, hico_assert_intern } from "./hico-util.js";

const noise_size = 256;

const vert_shader_source = `#version 300 es
layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec2 in_tex_coord;

out vec2 tex_coord;
out vec2 noise_tex_coord;
uniform float noise_scale_fac;

void main() {
  gl_Position = vec4(in_pos.x, in_pos.y, in_pos.z, 1.0);
  tex_coord = in_tex_coord;
  noise_tex_coord = in_tex_coord * noise_scale_fac;
}`;

const frag_shader_srgb_source = `#version 300 es
in mediump vec2 tex_coord;
in mediump vec2 noise_tex_coord;
out mediump vec4 color;

uniform lowp int color_mode;
uniform mediump float lut_mix_ratio;
uniform mediump sampler2D img_tex;
uniform mediump sampler2D noise_tex;
uniform mediump sampler3D lut_a, lut_b;

const mediump float q = 1.0f / 255.0f;
const mediump float a = 0.055;

mediump vec3 dithering_8bit(mediump vec3 img) {
  mediump vec3 r_img = mod(img, q);
  mediump vec3 q_img = img - r_img;
  mediump vec3 noise = texture(noise_tex, noise_tex_coord).rgb;
  q_img += vec3(greaterThan(r_img, noise)) * q;
  return q_img;
}

mediump vec3 srgb_to_yuv(mediump vec3 x) {
  const mediump mat3 conv_mat = transpose(mat3(
    0.299, 0.587, 0.114,
    -0.168736, -0.331264, 0.5,
    0.5, -0.418688, -0.081312
  ));
  return conv_mat * x + vec3(0.0, 0.5, 0.5);
}

mediump vec3 yuv_to_srgb(mediump vec3 x) {
  const mediump mat3 conv_mat = transpose(mat3(
    1, -1.21889419e-06, 1.40199959e+00,
    1, -3.44135678e-01, -7.14136156e-01,
    1, 1.77200007e+00, 4.06298063e-07
  ));
  return conv_mat * (x - vec3(0.0, 0.5, 0.5));
}

mediump vec3 s2lin(mediump vec3 x) {
  mediump vec3 s = vec3(lessThanEqual(x, vec3(0.04045)));
  return s * (x / 12.92f) + (1.0f - s) * pow((x + a) * (1.0f / (1.0f + a)), vec3(2.4f));
}

mediump vec3 lin2s(mediump vec3 x) {
  mediump vec3 s = vec3(lessThanEqual(x, vec3(0.0031308049535603713)));
  return s * (x * 12.92f) + (1.0f - s) * ((1.0f + a) * pow(x, vec3(1.0f / 2.4f)) - a);
}

mediump vec3 lut_conv(mediump vec3 img) {
  return mix(texture(lut_a, img).rgb, texture(lut_b, img).rgb, lut_mix_ratio);
}

void main(){
  mediump vec4 t = texture(img_tex, tex_coord);
  if(color_mode == 0)
    t.xyz = lut_conv(t.zyx);
  else if(color_mode == 1)
    t.xyz = yuv_to_srgb(lut_conv(srgb_to_yuv(t.xyz).zyx));
  else if(color_mode == 2)
    t.xyz = lin2s(lut_conv(s2lin(t.xyz).zyx));
  color.rgb = dithering_8bit(t.rgb);
  color.a = t.a;
}`;

const plane_vertex = new Float32Array([
  /* x, y, z, u, v */
  -1.0, -1.0, 0.0, 0.0, 1.0,
  -1.0, 1.0, 0.0, 0.0, 0.0,
  1.0, -1.0, 0.0, 1.0, 1.0,
  1.0, 1.0, 0.0, 1.0, 0.0
]);

const noise_tex_data = new Float32Array(noise_size * noise_size * 3);
for(let i = 0; i < noise_size * noise_size * 3; ++i) {
  noise_tex_data[i] = (Math.random() * 0.95 + 0.05) / 255;
}

function create_shader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if(!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    let info = gl.getShaderInfoLog(shader);
    throw "Could not compile " + (type === gl.VERTEX_SHADER ? "vertex" : "fragment") + " shader:\n" + info;
  }
  return shader;
}

function create_shader_program(gl, vert_shader, frag_shader) {
  const shader_program = gl.createProgram();
  gl.attachShader(shader_program, vert_shader);
  gl.attachShader(shader_program, frag_shader);
  gl.linkProgram(shader_program);
  if(!gl.getProgramParameter(shader_program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(shader_program);
    throw "Could not link shader program:\n" + info;
  }
  return shader_program;
}

function create_texture_3d(gl, img, h, w, d, c, color_depth, wrap_mode, scale_filter, src_offset) {
  src_offset = src_offset | 0;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, tex);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, wrap_mode);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, wrap_mode);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, wrap_mode);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, scale_filter);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, scale_filter);
  let ifmt, fmt, type;
  if(c === 3 && color_depth === 16 && img.constructor === Uint16Array) {
    ifmt = gl.RGB16F;
    fmt = gl.RGB;
    type = gl.HALF_FLOAT;
  }
  else
    hico_assert_intern(false, "Invalid data type for Texture3D");
  gl.texImage3D(gl.TEXTURE_3D, 0, ifmt, w, h, d, 0, fmt, type, img, src_offset);
  gl.bindTexture(gl.TEXTURE_3D, null);
  return tex;
}

function create_texture_2d(gl, img, h, w, c, depth, wrap_mode, scale_filter, src_offset) {
  src_offset = src_offset | 0;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap_mode);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap_mode);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, scale_filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, scale_filter);
  let ifmt, fmt, type;
  if(c === 4 && depth === 8 && (img.constructor === Uint8Array || img.constructor === Uint8ClampedArray)) {
    if(img.constructor === Uint8ClampedArray)
      img = new Uint8Array(img.buffer);
    ifmt = gl.RGBA8;
    fmt = gl.RGBA;
    type = gl.UNSIGNED_BYTE;
  }
  else if(c === 3 && depth === 32 && img.constructor === Float32Array) {
    ifmt = gl.RGB32F;
    fmt = gl.RGB;
    type = gl.FLOAT;
  }
  else
    hico_assert_intern(false, "Invalid data type for Texture2D");
  gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, img, src_offset);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

function create_lut_cube_tex_list(gl, data, n_diffusion, n_s, n_p, n_t) {
  const l = [];
  for(let i_diff = 0; i_diff < n_diffusion; ++i_diff) {
    const offset_diff = i_diff * n_s * n_p * n_t * 3;
    const diffuse_cube = create_texture_3d(gl, data, n_s, n_p, n_t, 3, 16, gl.CLAMP_TO_EDGE, gl.LINEAR, offset_diff);
    l.push(diffuse_cube);
  }
  return l;
}

export class EDiff extends HTMLBindable {
  constructor(canvas) {
    if(!canvas instanceof HTMLCanvasElement)
      throw new TypeError("Argument of EDiff must be a instanceof HTMLCanvasElement");
    super(canvas);

    this._gl = null;
    this._gl_vbo = null;
    this._gl_vao = null;
    this._gl_program = null;
    this._gl_img_tex = null;
    this._gl_noise_tex = null;
    this._gl_loc_lut_mix_ratio = null;
    this._gl_tex_lut_list = null;
    this._gl_loc_color_mode = null;

    this._lut_cube = null;
    this._diffusion_strength = 0.5;
    this._color_mode = 0;
    this._dirty = false;
  }

  set_img(h, w, data) {
    hico_assert(data.constructor === Uint8ClampedArray || data.constructor === Uint8Array, "Bad data type for data");
    hico_assert(h > 0 && w > 0, "Bad input shape");
    hico_assert(data.length === h * w * 4, "Bad input data length");

    if(this._gl !== null)
      this.clear_img();

    const img = data;
    const draw = this.raw_html_element;
    const pixel_ratio = window.devicePixelRatio;

    draw.height = h;
    draw.width = w;
    draw.style.height = h / pixel_ratio + "px";
    draw.style.width = w / pixel_ratio + "px";
    
    const gl = draw.getContext("webgl2", {
      "alpha": true,
      "antialias": false,
      "depth": false,
      "stencil": false,
      "preserveDrawingBuffer": true,
    });
    hico_assert(gl !== null, "Unable to initialize WebGL 2.0.\nYour browser or machine may not support it.");

    gl.viewport(0, 0, draw.width, draw.height);
    
    const vert_shader = create_shader(gl, gl.VERTEX_SHADER, vert_shader_source);
    const frag_shader = create_shader(gl, gl.FRAGMENT_SHADER, frag_shader_srgb_source);
    const shader_program = create_shader_program(gl, vert_shader, frag_shader);
    gl.deleteShader(frag_shader);
    gl.deleteShader(vert_shader);
    
    const loc_noise_scale_fac = gl.getUniformLocation(shader_program, "noise_scale_fac");
    const loc_img_tex = gl.getUniformLocation(shader_program, "img_tex");
    const loc_noise_tex = gl.getUniformLocation(shader_program, "noise_tex");
    const loc_lut_a = gl.getUniformLocation(shader_program, "lut_a");
    const loc_lut_b = gl.getUniformLocation(shader_program, "lut_b");
    const loc_lut_mix_ratio = gl.getUniformLocation(shader_program, "lut_mix_ratio");
    const loc_color_mode = gl.getUniformLocation(shader_program, "color_mode");

    const vbo = gl.createBuffer();
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, plane_vertex, gl.STATIC_COPY, 0, plane_vertex.length);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 5 * Float32Array.BYTES_PER_ELEMENT, 0);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 5 * Float32Array.BYTES_PER_ELEMENT, 3 * Float32Array.BYTES_PER_ELEMENT);
    gl.enableVertexAttribArray(0);
    gl.enableVertexAttribArray(1);
    gl.bindVertexArray(null);
    
    const tex = create_texture_2d(gl, img, h, w, 4, 8, gl.CLAMP_TO_EDGE, gl.NEAREST);
    const noise_tex = create_texture_2d(gl, noise_tex_data, noise_size, noise_size, 3, 32, gl.MIRRORED_REPEAT, gl.NEAREST);

    gl.useProgram(shader_program);
    gl.uniform1f(loc_noise_scale_fac, (Math.max(draw.height, draw.width) / noise_size) | 0)
    gl.uniform1i(loc_img_tex, 0);
    gl.uniform1i(loc_noise_tex, 1);
    gl.uniform1i(loc_lut_a, 2);
    gl.uniform1i(loc_lut_b, 3);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, noise_tex);

    /* save necessary object */
    this._gl = gl;
    this._gl_vbo = vbo;
    this._gl_vao = vao;
    this._gl_program = shader_program;
    this._gl_img_tex = tex;
    this._gl_noise_tex = noise_tex;
    this._gl_loc_lut_mix_ratio = loc_lut_mix_ratio;
    this._gl_loc_color_mode = loc_color_mode;

    if(this._lut_cube) {
      this._gl_tex_lut_list = create_lut_cube_tex_list(gl, this._lut_cube.data, ...this._lut_cube.shape);
      this._dirty = true;
      this.request_redraw();
    }
  }

  clear_img() {
    if(this._gl === null)
      return;
    const gl = this._gl;

    if(this._gl_tex_lut_list !== null)
      this._gl_tex_lut_list.forEach((tex)=>{ gl.deleteTexture(tex); });
    gl.deleteTexture(this._gl_noise_tex);
    gl.deleteTexture(this._gl_img_tex);
    gl.deleteProgram(this._gl_program);
    gl.deleteVertexArray(this._gl_vao);
    gl.deleteBuffer(this._gl_vbo);
    
    this._gl = null;
    this._gl_vbo = null;
    this._gl_vao = null;
    this._gl_program = null;
    this._gl_img_tex = null;
    this._gl_noise_tex = null;
    this._gl_loc_lut_mix_ratio = null;
    this._gl_tex_lut_list = null;
    this._gl_loc_color_mode = null;
  }

  set_lut_cube_ref(data, n_diffusion, n_s, n_p, n_t) {
    hico_assert(data.constructor === Uint16Array, "Bad data type for data");
    hico_assert(n_diffusion > 0 && n_s > 0 && n_p > 0 && n_t > 0, "Bad input shape");
    hico_assert(data.length === n_diffusion * n_s * n_p * n_t * 3, "Bad input data length");
    
    const gl = this._gl;
    if(this._gl_tex_lut_list !== null)
      this.clear_lut_cube_ref();
    if(gl !== null)
      this._gl_tex_lut_list = create_lut_cube_tex_list(gl, data, n_diffusion, n_s, n_p, n_t);
    this._lut_cube = {
      "data": data,
      "shape": [n_diffusion, n_s, n_p, n_t],
    };

    if(gl !== null) {
      this._dirty = true;
      this.request_redraw();
    }
  }

  clear_lut_cube_ref() {
    if(this._lut_cube === null)
      return;
    if(this._gl_tex_lut_list !== null) {
      this._gl_tex_lut_list.forEach((tex)=>{ this._gl.deleteTexture(tex); });
      this._gl_tex_lut_list = null;
    }
    this._lut_cube = null;
  }

  redraw() {
    const gl = this._gl;
    if(gl === null || this._diffusion_strength === null || this._lut_cube === null || this._color_mode === null)
      return;
    
    gl.clearColor(0, 0, 0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const [n_diffusion, n_s, n_p, n_t] = this._lut_cube.shape;
    let i_diff_a, i_diff_b, r_diff;
    {
      const i_diff = this._diffusion_strength * (n_diffusion - 1);
      let f_diff = Math.floor(i_diff) | 0;
      let c_diff = Math.ceil(i_diff) | 0;
      r_diff = i_diff - f_diff;
      if(i_diff <= 0) {
        f_diff = 0;
        c_diff = 1;
        r_diff = 0;
      }
      else if(i_diff >= n_diffusion - 1) {
        f_diff = n_diffusion - 1;
        c_diff = n_diffusion - 1;
        r_diff = 1;
      }
      i_diff_a = f_diff;
      i_diff_b = c_diff;
    }

    gl.useProgram(this._gl_program);
    gl.uniform1i(this._gl_loc_color_mode, this._color_mode);
    gl.uniform1f(this._gl_loc_lut_mix_ratio, r_diff);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._gl_img_tex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this._gl_noise_tex);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_3D, this._gl_tex_lut_list[i_diff_a]);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_3D, this._gl_tex_lut_list[i_diff_b]);

    gl.bindVertexArray(this._gl_vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  request_redraw() {
    if(this._dirty) {
      const me = this;
      requestAnimationFrame(()=>{
        me.redraw();
        this._dirty = false;
      });
    }
  }

  set diffusion_strength(v) {
    hico_assert(v >= 0 && v <= 1, "Invalid diffusion strength");
    this._diffusion_strength = v;
    this._dirty = true;
    this.request_redraw();
  }

  set color_mode(v) {
    hico_assert(v >= 0 && v <= 2, "Invalid color");
    this._color_mode = v;
    this._dirty = true;
    this.request_redraw();
  }

  get diffusion_strength() {
    return this._diffusion_strength;
  }

  get color_mode() {
    return this._color_mode;
  }

  unbind() {
    this.clear_img();
    super.unbind();
  }
}