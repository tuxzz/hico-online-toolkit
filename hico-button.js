import { HTMLBindable } from "./hico-htmlbindable.js";

function prevent_default(ev) {
  ev.preventDefault();
}

function rebind_handler(button, f) {
  const element = button.raw_html_element;
  if(button._real_handler !== null)
    element.removeEventListener("click", button._real_handler);
  button._real_handler = f;
  if(f !== null)
    element.addEventListener("click", f);
}

export class Button extends HTMLBindable {
  constructor(html_element) {
    if(!html_element instanceof EventTarget)
      throw new TypeError("Argument of Button must be a HTMLElement + EventTarget");
    super(html_element);

    this._real_handler = null;

    const tag = html_element.tagName.toLowerCase();
    if(tag === "a" && html_element.href) {
      this.click_handler = html_element.href;
      this._saved_href = html_element.href;
    }
    else {
      this._click_handler = null;
      this._saved_href = null;
    }
    html_element.addEventListener("dragstart", prevent_default);
  }

  set click_handler(v) {
    const element = this.raw_html_element;
    const is_a = element.tagName.toLowerCase() === "a";
    if(v instanceof Function) {
      if(is_a)
        element.href = "javascript:void(0);";
      rebind_handler(this, (ev)=>{
        v(ev);
        ev.preventDefault();
      });
    }
    else if(typeof v === "string") {
      if(is_a) {
        element.href = v;
        rebind_handler(this, null);
      }
      else {
        rebind_handler(this, (ev)=>{
          window.location.href = v;
          ev.preventDefault();
        });
      }
    }
    else if(v === null) {
      if(is_a)
        element.removeAttribute("href");
      rebind_handler(this, null);
    }
    else
      throw new TypeError("Invalid type for click_handler");
  }

  get click_handler() {
    return this._click_handler;
  }

  set allow_drag(v) {
    if(v)
      html_element.addEventListener("dragstart", prevent_default);
    else
      html_element.removeEventListener("dragstart", prevent_default);
  }

  unbind() {
    if(this.raw_html_element.tagName.toLowerCase() === "a" && this._saved_href)
      this.click_handler = this._saved_href;
    else
      this.click_handler = null;
    html_element.removeEventListener("dragstart", prevent_default);
    super.unbind();
  }
}