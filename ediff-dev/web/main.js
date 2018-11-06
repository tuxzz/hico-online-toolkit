(function(){
  const cube_file = "./cube_bundle.bin";
  const n_cube = 5;
  const cube_color_mode = new Uint8Array([0, 1, 0, 1, 1, 2]);
  const [n_diffusion, n_s, n_p, n_t, n_c] = [9, 16, 16, 16, 3];
  const noise_size = 256;
  let lut_cube = null;

  const pixel_ratio = window.devicePixelRatio || 1;
  const draw = document.getElementById("draw_area");
  const diffusion_slider = document.getElementById("diffusion_strength");
  const pattern_mode_select = document.getElementById("pattern_mode");
  const file_name = document.getElementById("file_name");
  const choose_file = document.getElementById("choose_file");
  const loading_span = document.getElementById("loading");

  let g_gl = null;
  let g_gl_vbo = null;
  let g_gl_vao = null;
  let g_gl_program = null;
  let g_gl_img_tex = null;
  let g_gl_noise_tex = null;
  let g_gl_loc_lut_mix_ratio = null;
  let g_gl_tex_lut_list = null;
  let g_gl_loc_color_mode = null;

  let g_selected_cube = 0;
  let g_diffusion_strength = 0.5;
  let g_color_mode = 0;

  {
    const req = new XMLHttpRequest();
    req.open("GET", cube_file, true);
    req.responseType = "arraybuffer";

    req.addEventListener("error", function(ev) {
      da_assert(false, "Failed to load LUT Cube: Network error");
    });

    req.addEventListener("load", function(ev) {
      const buffer = req.response;
      da_assert(req.status === 200, "Failed to load LUT Cube: HTTP code " + req.status);
      da_assert(buffer, "Failed to load LUT Cube: Empty response");
      const excepted_len = n_cube * n_diffusion * n_s * n_p * n_t * n_c * 2;
      da_assert(buffer.byteLength === excepted_len, "Bad LUT Cube: (got " + buffer.byteLength + " bytes, excepted " + excepted_len + " bytes");
      lut_cube = new Uint16Array(buffer);
      loading_span.innerText = "";
    });
    req.send();
  }
  
  const noise_tex_data = new Float32Array(noise_size * noise_size * 3);
  for(let i = 0; i < noise_size * noise_size * 3; ++i) {
    noise_tex_data[i] = (Math.random() * 0.95 + 0.05) / 255;
  }

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

  mediump vec4 dithering_8bit(mediump vec4 img) {
    mediump vec4 r_img = mod(img, q);
    mediump vec4 q_img = img - r_img;
    mediump vec4 noise = texture(noise_tex, noise_tex_coord);
    q_img += vec4(greaterThan(r_img, noise)) * q;
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
    mediump vec3 t = vec3(0.0, tex_coord.s, 0.0);
    t = texture(img_tex, tex_coord).rgb;
    if(color_mode == 0)
      t = lut_conv(t.zyx);
    else if(color_mode == 1)
      t = yuv_to_srgb(lut_conv(srgb_to_yuv(t).zyx));
    else if(color_mode == 2)
      t = lin2s(lut_conv(s2lin(t).zyx));
    color = dithering_8bit(vec4(t.x, t.y, t.z, 1.0f));
  }`;
  

  const plane_vertex = new Float32Array([
    /* x, y, z, u, v */
    -1.0, -1.0, 0.0, 0.0, 1.0,
    -1.0, 1.0, 0.0, 0.0, 0.0,
    1.0, -1.0, 0.0, 1.0, 1.0,
    1.0, 1.0, 0.0, 1.0, 0.0
  ]);

  function da_assert(cond, msg) {
    if(!cond) {
      alert(msg);
      throw msg;
    }
  }

  function select_and_load_image(cb) {
    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = "image/*";
    picker.onchange = function(evt) {
      let files = evt.target.files;
      if(files.length <= 0)
        return;
      let f = files[0];
      let reader = new FileReader();
      reader.onload = function(e) {
        let img = document.createElement("img");
        img.onload = function() {
          let o = load_image_from_img(img);
          o["filename"] = f.name
          cb(o);
        }
        img.src = e.target.result;
        img.title = f.name;
        
      };
      reader.readAsDataURL(f);
    };
    picker.click();
  }
  
  function load_image_from_img(img) {
    const canvas = document.createElement("canvas"); 
    canvas.height = img.height;
    canvas.width = img.width;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    return {
      "height": canvas.height,
      "width": canvas.width,
      "data": data,
    };
  }

  function createShaderObject(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if(!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      let info = gl.getShaderInfoLog(shader);
      throw "Could not compile " + (type === gl.VERTEX_SHADER ? "vertex" : "fragment") + " shader.\n\n" + info;
    }
    return shader;
  }

  function createShaderProgramObject(gl, vert_shader, frag_shader) {
    const shader_program = gl.createProgram();
    gl.attachShader(shader_program, vert_shader);
    gl.attachShader(shader_program, frag_shader);
    gl.linkProgram(shader_program);
    if(!gl.getProgramParameter(shader_program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(shader_program);
      throw "Could not link shader program. \n\n" + info;
    }
    return shader_program;
  }

  function createTexture3DObject(gl, img, h, w, d, c, color_depth, wrap_mode, scale_filter, src_offset) {
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
      throw "Fuck";
    gl.texImage3D(gl.TEXTURE_3D, 0, ifmt, w, h, d, 0, fmt, type, img, src_offset);
    gl.bindTexture(gl.TEXTURE_3D, null);
    return tex;
  }

  function createTexture2DObject(gl, img, h, w, c, depth, wrap_mode, scale_filter, src_offset) {
    src_offset = src_offset | 0;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap_mode);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap_mode);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, scale_filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, scale_filter);
    let ifmt, fmt, type;
    if(c === 3 && depth === 8 && (img.constructor === Uint8Array || img.constructor === Uint8ClampedArray)) {
      ifmt = gl.RGB8;
      fmt = gl.RGB;
      type = gl.UNSIGNED_BYTE;
    }
    else if(c === 4 && depth === 8 && (img.constructor === Uint8Array || img.constructor === Uint8ClampedArray)) {
      ifmt = gl.RGBA8;
      fmt = gl.RGBA;
      type = gl.UNSIGNED_BYTE;
    }
    else if(c === 3 && depth === 32 && img.constructor === Float32Array) {
      ifmt = gl.RGB32F;
      fmt = gl.RGB;
      type = gl.FLOAT;
    }
    else if(c === 4 && depth === 32 && img.constructor === Float32Array) {
      ifmt = gl.RGBA32F;
      fmt = gl.RGBA;
      type = gl.FLOAT;
    }
    else
      throw "Fuck";
    gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, img, src_offset);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  function setup_gl(h, w, img) {
    da_assert(img.constructor === Uint8ClampedArray || img.constructor === Uint8Array, "Fuck bad img type!");
    h = h | 0;
    w = w | 0;
    draw.height = h;
    draw.width = w;
    draw.style.height = h / pixel_ratio + "px";
    draw.style.width = w / pixel_ratio + "px";

    const gl = draw.getContext("webgl2", {
      antialias: false,
      depth: false,
    });
    da_assert(gl !== null, "Unable to initialize WebGL 2.0. Your browser or machine may not support it.")

    gl.viewport(0, 0, draw.width, draw.height);
    
    const vert_shader = createShaderObject(gl, gl.VERTEX_SHADER, vert_shader_source);
    const frag_shader = createShaderObject(gl, gl.FRAGMENT_SHADER, frag_shader_srgb_source);
    const shader_program = createShaderProgramObject(gl, vert_shader, frag_shader);
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
    
    const tex = createTexture2DObject(gl, img, h, w, 4, 8, gl.CLAMP_TO_EDGE, gl.NEAREST);
    const noise_tex = createTexture2DObject(gl, noise_tex_data, noise_size, noise_size, 3, 32, gl.MIRRORED_REPEAT, gl.NEAREST);

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
    g_gl = gl;
    g_gl_vbo = vbo;
    g_gl_vao = vao;
    g_gl_program = shader_program;
    g_gl_img_tex = tex;
    g_gl_noise_tex = noise_tex;
    g_gl_loc_lut_mix_ratio = loc_lut_mix_ratio;
    g_gl_loc_color_mode = loc_color_mode;
  }

  function select_lut_cube(i_cube) {
    i_cube = i_cube | 0;
    g_selected_cube = null;

    da_assert(i_cube >= 0 && i_cube < n_cube, "Internal error: Invalid cube index " + i_cube);
    da_assert(g_gl !== null, "Internal error: WebGL context is not created.");
    const gl = g_gl;
    
    if(g_gl_tex_lut_list !== null) {
      const n = g_gl_tex_lut_list.length;
      for(let i = 0; i < n; ++i)
        gl.deleteTexture(g_gl_tex_lut_list[i]);
    }
    g_gl_tex_lut_list = null;

    const l = [];
    const offset_cube = i_cube * n_diffusion * n_s * n_p * n_t * n_c;
    for(let i_diff = 0; i_diff < n_diffusion; ++i_diff) {
      const offset_diff = offset_cube + i_diff * n_s * n_p * n_t * n_c;
      const diffuse_cube = createTexture3DObject(gl, lut_cube, n_s, n_p, n_t, 3, 16, gl.CLAMP_TO_EDGE, gl.LINEAR, offset_diff);
      l.push(diffuse_cube);
    }

    g_gl_tex_lut_list = l;
    g_selected_cube = i_cube;
  }

  function redraw_gl() {
    da_assert(g_gl !== null, "Internal error: WebGL context is not created.");
    const gl = g_gl;

    da_assert(g_diffusion_strength >= 0.0 && g_diffusion_strength <= 1.0, "Fuck bad strength.");
    da_assert(lut_cube !== null, "Fuck no cube!");
    da_assert(g_gl_tex_lut_list !== null, "Fuck no cube selected!");
    let i_diff_a, i_diff_b, r_diff;
    {
      const i_diff = g_diffusion_strength * (n_diffusion - 1);
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

    gl.useProgram(g_gl_program);
    gl.uniform1i(g_gl_loc_color_mode, g_color_mode);
    gl.uniform1f(g_gl_loc_lut_mix_ratio, r_diff);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, g_gl_img_tex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, g_gl_noise_tex);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_3D, g_gl_tex_lut_list[i_diff_a]);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_3D, g_gl_tex_lut_list[i_diff_b]);

    gl.bindVertexArray(g_gl_vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
    gl.bindVertexArray(null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  function destroy_gl() {
    da_assert(g_gl !== null, "Internal error: WebGL context is not created.");
    const gl = g_gl;

    if(g_gl_tex_lut_list !== null) {
      const n = g_gl_tex_lut_list.length;
      for(let i = 0; i < n; ++i)
        gl.deleteTexture(g_gl_tex_lut_list[i]);
    }
    gl.deleteTexture(g_gl_noise_tex);
    gl.deleteTexture(g_gl_img_tex);
    gl.deleteProgram(g_gl_program);
    gl.deleteVertexArray(g_gl_vao);
    gl.deleteBuffer(g_gl_vbo);
    
    g_gl = null;
    g_gl_vbo = null;
    g_gl_vao = null;
    g_gl_program = null;
    g_gl_img_tex = null;
    g_gl_noise_tex = null;
    g_gl_loc_lut_mix_ratio = null;
    g_gl_loc_color_mode = null;
    g_gl_tex_lut_list = null;
  }

  diffusion_slider.value = (g_diffusion_strength * 100) | 0;
  diffusion_slider.addEventListener("input", function(ev) {
    g_diffusion_strength = parseInt(diffusion_slider.value) / 100.0;
    requestAnimationFrame(function(){
      redraw_gl();
    });
  });

  pattern_mode_select.value = g_selected_cube;
  pattern_mode_select.addEventListener("change", function(ev) {
    const i_cube = parseInt(pattern_mode_select.value);
    g_color_mode = cube_color_mode[i_cube];
    if(i_cube === 5)
      select_lut_cube(0); // use sRGB cube for linRGB
    else
      select_lut_cube(i_cube);
    redraw_gl();
  });

  choose_file.addEventListener("click", function(ev) {
    select_and_load_image(function(d) {
      file_name.innerText = d.filename;
      if(g_gl !== null)
        destroy_gl();
      setup_gl(d.height, d.width, new Uint8Array(d.data.buffer));
      select_lut_cube(g_selected_cube);
      redraw_gl();
    });
  });
})()