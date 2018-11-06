import { Rect } from "./hico-rect.js";
import { HTMLBindable } from "./hico-htmlbindable.js";
import { hico_assert_intern, remove_one_from_list, find_one_from_list } from "./hico-util.js";

const c_move_button = 1;
const c_sub_move_button = 0;
const c_scale_button = 2;
const c_scale_content_size_limit = 16;

let g_scale_button_pressed = null;
let g_move_button_pressed = null;
let g_first_mouse_pos = null;
let g_first_touch_list = null;

let g_transcation_flickable = null;
let g_orig_viewport = null;
let g_uncommited_movement = null;
let g_uncommited_scale_movement = null;
let g_uncommited_scale = null;

function transcation_viewport(flickable) {
  hico_assert_intern(!g_transcation_flickable, "Transcation is already created");
  g_transcation_flickable = flickable;
  g_orig_viewport = flickable.viewport;
  g_uncommited_movement = [0, 0];
  g_uncommited_scale_movement = [0, 0];
  g_uncommited_scale = [1.0, 1.0];
}

function rollback_viewport() {
  hico_assert_intern(g_transcation_flickable, "Transcation is not created");

  g_transcation_flickable.viewport = g_orig_viewport;

  g_transcation_flickable = null;
  g_orig_viewport = null;
  g_uncommited_movement = null;
  g_uncommited_scale_movement = null;
  g_uncommited_scale = null;
}

function preview_viewport() {
  hico_assert_intern(g_transcation_flickable, "Transcation is not created");

  const flickable = g_transcation_flickable;
  flickable.viewport = g_orig_viewport;
  const [umy, umx] = flickable.map_screen_to_viewport_delta(...g_uncommited_movement);
  const [usmy, usmx] = flickable.map_screen_to_viewport_delta(...g_uncommited_scale_movement);
  const [usy, usx] = g_uncommited_scale;
  const o = g_orig_viewport;
  const vp = Rect.from_yxhw(o.y + umy + usmy, o.x + umx + usmx, o.h * usy, o.w * usx);
  flickable.viewport = vp;
  return vp;
}

function commit_viewport() {
  hico_assert_intern(g_transcation_flickable, "Transcation is not created");

  preview_viewport();

  g_transcation_flickable = null;
  g_orig_viewport = null;
  g_uncommited_movement = null;
  g_uncommited_scale_movement = null;
  g_uncommited_scale = null;
}

function move_flickable(dy, dx) {
  hico_assert_intern(g_transcation_flickable, "Transcation is not created");
  const [umy, umx] = g_uncommited_movement;
  g_uncommited_movement = [umy - dy, umx - dx];
}

function set_flickable_movement(my, mx) {
  hico_assert_intern(g_transcation_flickable, "Transcation is not created");
  g_uncommited_movement = [-my, -mx];
}

function scale_flickable(ey, ex, fy, fx) {
  hico_assert_intern(g_transcation_flickable, "Transcation is not created");
  fx = fx === "number" ? fx : fy;

  const [usmy, usmx] = g_uncommited_scale_movement;
  const [usy, usx] = g_uncommited_scale;
  const [sy, sx] = [usy * fy, usx * fx];
  const [ch, cw] = g_transcation_flickable.content_size;
  if(ch * ch / (g_orig_viewport.h * sy) < c_scale_content_size_limit || cw * cw / (g_orig_viewport.w * sx) < c_scale_content_size_limit)
    return;
  g_uncommited_scale_movement = [usmy + ey * (1.0 - fy), usmx + ex * (1.0 - fx)];
  g_uncommited_scale = [sy, sx];
}

function get_flickable_scale_limit() {
  const [ch, cw] = g_transcation_flickable.content_size;
  return [(ch * ch) / (c_scale_content_size_limit * g_orig_viewport.h), (cw * cw) / (c_scale_content_size_limit * g_orig_viewport.w)];
}

function set_flickable_scale(ey, ex, sy, sx) {
  hico_assert_intern(g_transcation_flickable, "Transcation is not created");
  sx = sx === "number" ? sx : sy;

  const [sy_limit, sx_limit] = get_flickable_scale_limit();
  hico_assert_intern(sy <= sy_limit && sx <= sx_limit, "Scale factor reached limit");

  g_uncommited_scale_movement = [ey * (1.0 - sy), ex * (1.0 - sx)];
  g_uncommited_scale = [sy, sx];
}

function on_mousedown(ev, flickable) {
  if(g_first_touch_list !== null)
    return;
  const button = ev.button;
  const [ey, ex] = flickable.map_screen_to_element(ev.clientY, ev.clientX);
  if((button === c_move_button || button === c_sub_move_button) && g_scale_button_pressed === null && (g_move_button_pressed === null || find_one_from_list(g_move_button_pressed, button) === null)) {
    if(g_move_button_pressed === null)
    {
      transcation_viewport(flickable);
      g_move_button_pressed = [];
      g_first_mouse_pos = [ey, ex];
    }
    g_move_button_pressed.push(button);

    ev.preventDefault();
  }
  else if(button === c_scale_button && g_move_button_pressed === null && (g_scale_button_pressed === null || find_one_from_list(g_scale_button_pressed, button) === null)) {
    if(g_scale_button_pressed === null) {
      g_scale_button_pressed = [];
      transcation_viewport(flickable);
      g_first_mouse_pos = [ey, ex];
    }
    g_scale_button_pressed.push(button);
    ev.preventDefault();
  }
}

function on_wheel(ev, flickable) {
  if(g_scale_button_pressed !== null)
    return;
  const [ey, ex] = flickable.map_screen_to_element(ev.clientY, ev.clientX);
  const dwy = ev.deltaY;

  const fac = dwy > 0 ? 1.25 : 0.8;
  let self_transcation = false;
  if(!g_transcation_flickable) {
    transcation_viewport(flickable);
    self_transcation = true;
  }
  scale_flickable(ey, ex, fac);
  if(self_transcation)
    commit_viewport();
  else
  {
    commit_viewport();
    transcation_viewport(flickable);
    if(g_move_button_pressed)
      g_first_mouse_pos = [ey, ex];
  }
  ev.preventDefault();
}

window.addEventListener("mousemove", (ev)=>{
  if(g_move_button_pressed) {
    const [ey, ex] = g_transcation_flickable.map_screen_to_element(ev.clientY, ev.clientX);
    const content = g_transcation_flickable.flick_content;
    if(content === null)
      return;
    const [ly, lx] = g_first_mouse_pos;
    const [my, mx] = [ey - ly, ex - lx];

    set_flickable_movement(my, mx);
    preview_viewport();

    ev.preventDefault();
  }
  else if(g_scale_button_pressed) {
    const [ey, ex] = g_transcation_flickable.map_screen_to_element(ev.clientY, ev.clientX);
    const [ly, lx] = g_first_mouse_pos;
    const mx = ex - lx;
    const d = Math.abs(mx);
    const dfac = d * 0.0025;
    let f;
    if(mx < 0)
      f = Math.min(...get_flickable_scale_limit(), 1.0 + dfac);
    else
      f = Math.max(1e-5, 1 / (1.0 + dfac));
    set_flickable_scale(ly, lx, f);
    preview_viewport();
    ev.preventDefault();
  }
});

window.addEventListener("click", (ev)=>{
  if(g_move_button_pressed !== null || g_scale_button_pressed !== null)
    ev.preventDefault();
});

window.addEventListener("mouseup", (ev)=>{
  if(g_move_button_pressed !== null) {
    remove_one_from_list(g_move_button_pressed, ev.button);
    if(g_move_button_pressed.length === 0) {
      g_move_button_pressed = null;
      commit_viewport();
    }
    ev.preventDefault();
  }
  else if(g_scale_button_pressed !== null) {
    remove_one_from_list(g_scale_button_pressed, ev.button);
    if(g_scale_button_pressed.length === 0) {
      g_scale_button_pressed = null;
      commit_viewport();
    }
    ev.preventDefault();
  }
});

function preprocess_touch_list(flickable, touch_list) {
  const l = [];
  const n_touch = touch_list.length;
  for(let i_touch = 0; i_touch < n_touch; ++i_touch) {
    const touch = touch_list[i_touch];
    const id = touch.identifier;
    l.push({"id": id, "pos": flickable.map_screen_to_element(touch.clientY, touch.clientX)});
  }
  return l;
}

function on_touchstart(ev, flickable) {
  if(g_move_button_pressed !== null || g_scale_button_pressed !== null)
    return;
  if(g_first_touch_list !== null)
    commit_viewport();
  transcation_viewport(flickable);
  g_first_touch_list = preprocess_touch_list(flickable, ev.touches);
  ev.preventDefault();
}

window.addEventListener("touchmove", (ev)=>{
  if(g_first_touch_list === null)
    return;
  const flickable = g_transcation_flickable;
  const touch_list = ev.touches;
  const n_touch = touch_list.length;
  const new_touch_list = preprocess_touch_list(flickable, touch_list);
  const delta_touch_list = [];
  new_touch_list.forEach((v)=>{
    const [ty, tx] = v.pos;
    const id = v.id;
    const i_first = find_one_from_list(g_first_touch_list, (w)=>{ return id === w.id; });
    if(i_first === null)
      return;
    const [lty, ltx] = g_first_touch_list[i_first].pos;
    delta_touch_list.push({
      "id": v.id,
      "pos": [ty - lty, tx - ltx],
    });
  });

  if(n_touch === 1) {
    const [dy, dx] = delta_touch_list[0].pos;
    set_flickable_movement(dy, dx);
    preview_viewport();
    ev.preventDefault();
  }
  else if(n_touch == 2) {
    const [dy0, dx0] = delta_touch_list[0].pos;
    const [dy1, dx1] = delta_touch_list[1].pos;

    const [ly0, lx0] = g_first_touch_list[0].pos;
    const [ly1, lx1] = g_first_touch_list[1].pos;
    const [ny0, nx0] = new_touch_list[0].pos;
    const [ny1, nx1] = new_touch_list[1].pos;
    
    const [cy, cx] = [(ly0 + ly1) * 0.5, (lx0 + lx1) * 0.5];

    const ld = Math.sqrt((ly0 - ly1) * (ly0 - ly1) + (lx0 - lx1) * (lx0 - lx1));
    const nd = Math.sqrt((ny0 - ny1) * (ny0 - ny1) + (nx0 - nx1) * (nx0 - nx1));
    const dd = nd - ld;
    const dfac = Math.abs(dd) * 0.01;
    let f;
    if(dd < 0)
      f = Math.min(...get_flickable_scale_limit(), 1.0 + dfac);
    else
      f = Math.max(1e-5, 1 / (1.0 + dfac));
    set_flickable_scale(cy, cx, f);
    preview_viewport();
    ev.preventDefault();
  }
});

window.addEventListener("touchend", (ev)=>{
  if(g_first_touch_list === null)
    return;
  const flickable = g_transcation_flickable;
  commit_viewport();
  if(ev.touches.length <= 0)
    g_first_touch_list = null;
  else {
    g_first_touch_list = preprocess_touch_list(flickable, ev.touches);
    transcation_viewport(flickable);
  }
  ev.preventDefault();
});

export class Flickable extends HTMLBindable {
  constructor(html_element) {
    if(!html_element instanceof EventTarget)
      throw new TypeError("Argument of Flickable must be a HTMLElement + EventTarget");
    super(html_element);

    [this._last_view_h, this._last_view_w] = [Math.max(1, html_element.clientHeight), Math.max(1, html_element.clientWidth)];
    this._flick_content = null;
    this._viewport = Rect.from_yxhw(0, 0, 0, 0);

    const me = this;
    this._on_mousedown = (ev)=>{ on_mousedown(ev, me); };
    this._on_wheel = (ev)=>{ on_wheel(ev, me); };
    this._on_touchstart = (ev)=>{ on_touchstart(ev, me); };
    this._on_contextmenu = (ev)=>{ ev.preventDefault(); };
    this._saved_transform = null;
    this._saved_transformOrigin = null;

    html_element.addEventListener("mousedown", this._on_mousedown);
    html_element.addEventListener("wheel", this._on_wheel);
    html_element.addEventListener("touchstart", this._on_touchstart);
    html_element.addEventListener("contextmenu", this._on_contextmenu);
  }

  set flick_content(e) {
    if(this._flick_content !== null) {
      this._flick_content.style.transform = this._saved_transform;
      this._flick_content.style.transformOrigin = this._saved_transformOrigin;
    }
    if(e !== null) {
      this._saved_transform = e.style.transform;
      this._saved_transformOrigin = e.style.transformOrigin;
    }
    this._flick_content = e;
  }

  get flick_content() {
    return this._flick_content;
  }

  get viewport() {
    const flickable = this.raw_html_element;
    const [vh, vw] = [Math.max(1, flickable.clientHeight), Math.max(1, flickable.clientWidth)];
    if(vh != this._last_view_h) {
      this._viewport.h *= vh / this._last_view_h;
      this._last_view_h = vh;
    }
    if(vw != this._last_view_w) {
      this._viewport.w *= vw / this._last_view_w;
      this._last_view_w = vw;
    }
    return this._viewport.clone();
  }

  set viewport(r) {
    this._viewport = r.normalized();
    const e = this.flick_content;
    if(e === null)
      return;
    const flickable = this.raw_html_element;
    [this._last_view_h, this._last_view_w] = [Math.max(1, flickable.clientHeight), Math.max(1, flickable.clientWidth)];

    const [scale_y, scale_x] = [flickable.clientHeight / r.h, flickable.clientWidth / r.w];
    const [translate_y, translate_x] = [-r.y, -r.x];
    e.style.transformOrigin = 0 + "px " + 0 + "px";
    e.style.transform = "scale(" + scale_x + ", " + scale_y + ") " + "translate(" + translate_x +  "px, " + translate_y + "px) ";
  }

  get content_size() {
    const content = this.flick_content;
    if(!content)
      return [1, 1];
    return [content.clientHeight, content.clientWidth];
  }

  map_screen_to_viewport_delta(y, x) {
    const e = this.flick_content;
    if(e === null)
      return [1, 1];
    const r = this.viewport;
    const flickable = this.raw_html_element;
    const [scale_y, scale_x] = [flickable.clientHeight / r.h, flickable.clientWidth / r.w];
    return [y / scale_y, x / scale_x];
  }

  reset_viewport_to_center() {
    if(g_transcation_flickable !== null)
      rollback_viewport();
    const e = this.flick_content;
    if(e === null)
      return;
    const flickable = this.raw_html_element;
    const flickable_aspect = flickable.clientHeight / flickable.clientWidth;
    const [eh, ew] = [e.clientHeight, e.clientWidth];
    let h, w, x, y;
    [y, x] = [e.clientTop, e.clientLeft];
    if(eh / ew <= flickable_aspect) {
      [h, w] = [ew * flickable_aspect, ew];
      y = (eh - h) * 0.5;
    }
    else {
      [h, w] = [eh, eh / flickable_aspect];
      x = (ew - w) * 0.5;
    }
    this.viewport = Rect.from_yxhw(y, x, h, w);
  }
  
  unbind() {
    const html_element = this.raw_html_element;
    if(g_transcation_flickable === this)
      rollback_viewport();
    this.flick_content = null;
    this.viewport = null;
    html_element.removeEventListener("mousedown", this._on_mousedown);
    html_element.removeEventListener("wheel", this._on_wheel);
    html_element.removeEventListener("touchstart", this._on_touchstart);
    html_element.removeEventListener("contextmenu", this._on_contextmenu);
    super.unbind();
  }
}