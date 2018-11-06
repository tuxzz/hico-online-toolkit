let on_device_pixel_ratio_changed = null;
let saved_device_pixel_ratio = null;
let device_pixel_ratio_match = window.matchMedia("screen and (min-resolution: 2dppx)");

export function hico_assert(cond, msg) {
  if(!cond)
    throw msg;
}

export function hico_assert_intern(cond, msg) {
  hico_assert(cond, "Internal error: " + msg);
}

export function find_one_from_list(l, cond) {
  const n = l.length;
  if(cond instanceof Function) {
    for(let i = 0; i < n; ++i) {
      if(cond(l[i]) === true)
        return i;
    }
  }
  else {
    for(let i = 0; i < n; ++i) {
      if(l[i] === cond)
        return i;
    }
  }
  return null;
}

export function remove_all_from_list(l, cond) {
  const n = l.length;
  let n_removed = 0;
  if(cond instanceof Function) {
    for(let i = 0; i < n; ++i) {
      if(cond(l[i])) {
        l.splice(i, 1);
        ++n_removed;
      }
    }
  }
  else {
    for(let i = 0; i < n; ++i) {
      if(l[i] === cond) {
        l.splice(i, 1);
        ++n_removed;
      }
    }
  }
  return n_removed;
}

export function remove_one_from_list(l, cond) {
  const n = l.length;
  if(cond instanceof Function) {
    for(let i = 0; i < n; ++i) {
      if(cond(l[i])) {
        l.splice(i, 1);
        return true;
      }
    }
  }
  else {
    for(let i = 0; i < n; ++i) {
      if(l[i] === cond) {
        l.splice(i, 1);
        return true;
      }
    }
  }
  return false;
}

export function lerp(a, b, ratio) {
  return a + (b - a) * ratio;
}

export function number_to_string_with_comma(x)
{ return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ","); }

export function choose_file(cb, mime, multi)
{
  let f = document.createElement("input");
  f.type = "file";
  f.style.display = "none";
  if(multi) {
    f.multiple = "multiple";
    f.onchange = ()=>{ cb(f.files) }; 
  }
  else
    f.onchange = ()=>{ cb(f.files[0]) }; 
  f.accept = mime;
  document.body.appendChild(f);
  f.click();
  f.remove();
}

export function copy_text(s) {
  hico_assert(s.length <= 262144, "String is too long for copy(max 262144 character, got " + s.length);

  const textarea = document.createElement("textarea");
  textarea.innerText = s;

  document.body.appendChild(textarea);
  
  textarea.select();
  document.execCommand("copy");

  textarea.remove();
}

export function download_url(name, url) {
  const link = document.createElement("a");
  link.href = url
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

export function save_bytearray(name, mime, data) {
  const blob = new Blob([data], {type: mime});
  download_url(name, URL.createObjectURL(blob));
};

function on_devicel_pixel_ratio_changed(ev) {
  const r = windows.devicePixelRatio;
  if(r === saved_device_pixel_ratio)
    return;
  on_device_pixel_ratio_changed.forEach((v)=>{
    v(r, saved_device_pixel_ratio);
  });
  saved_device_pixel_ratio = r;
}

export function bind_device_pixel_ratio_changed(f) {
  if(on_device_pixel_ratio_changed === null) {
    on_device_pixel_ratio_changed = [f];
    saved_device_pixel_ratio = window.devicePixelRatio;
    device_pixel_ratio_match = window.matchMedia('screen and (min-resolution: 2dppx)')
    device_pixel_ratio_match.addListener(on_devicel_pixel_ratio_changed);
  }
  else {
    if(find_one_from_list(on_device_pixel_ratio_changed, f) !== null)
      return;
    on_device_pixel_ratio_changed.push(f);
  }
}

export function unbind_device_pixel_ratio_changed(f) {
  if(on_device_pixel_ratio_changed === null)
    return;
  const i = find_one_from_list(on_device_pixel_ratio_changed, f);
  if(i === null)
    return;
  if(on_device_pixel_ratio_changed.length === 1) {
    device_pixel_ratio_match.removeListener(on_devicel_pixel_ratio_changed);
    device_pixel_ratio_match = null;
    on_device_pixel_ratio_changed = null;
    saved_device_pixel_ratio = null;
  }
  else
    delete on_device_pixel_ratio_changed[i];
}

export function load_data_from_img(img) {
  const canvas = document.createElement("canvas"); 
  canvas.height = img.height;
  canvas.width = img.width;
  const ctx = canvas.getContext("2d", {
    "alpha": true,
    "willReadFrequently": true,
  });
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  return {
    "h": canvas.height,
    "w": canvas.width,
    "data": data,
  };
}