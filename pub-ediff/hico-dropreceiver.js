import { HTMLBindable } from "./hico-htmlbindable.js";

const c_text = ""

export class DropReceiver extends HTMLBindable {
  constructor(html_element) {
    if(!html_element instanceof EventTarget)
      throw new TypeError("Argument of DropReceiver must be a HTMLElement + EventTarget");
    super(html_element);

    this.on_dragenter = null;
    this.on_dragleave = null;
    this.on_received = null;

    this._entered_level = null;

    const me = this;
    this._on_dragenter = (ev)=>{
      ev.preventDefault();
      if(me._entered_level === null) {
        if(me.on_dragenter !== null)
          me.on_dragenter(ev, me);
        me._entered_level = [];
      }
      me._entered_level.push(ev.target);
    };

    this._on_dragleave = (ev)=>{
      ev.preventDefault();
      if(me._entered_level.length == 1) {
        me._entered_level = null;
        if(me.on_dragleave !== null)
          me.on_dragleave(ev, me);
      }
      else
        me._entered_level.pop();
    };

    this._on_dragover = (ev)=>{
      ev.preventDefault();
    };

    this._on_drop = (ev)=>{
      ev.preventDefault();
      me._entered_level = null;
      if(me.on_received !== null)
        me.on_received(ev, me);
    };

    this._on_dragend = (ev)=>{
      ev.preventDefault();
      if(me.on_dragenter !== null)
        me.on_dragleave(ev, me);
      me._entered_level = null;
    };

    html_element.addEventListener("dragenter", this._on_dragenter);
    html_element.addEventListener("dragleave", this._on_dragleave);
    html_element.addEventListener("dragover", this._on_dragover);
    html_element.addEventListener("drop", this._on_drop);
    html_element.addEventListener("dragend", this._on_dragend);
  }

  unbind() {
    const html_element = this.raw_html_element;
    
    this.on_dragenter = null;
    this.on_dragleave = null;
    this.on_received = null;
    html_element.removeEventListener("dragenter", this._on_dragenter);
    html_element.removeEventListener("dragleave", this._on_dragleave);
    html_element.removeEventListener("dragover", this._on_dragover);
    html_element.removeEventListener("drop", this._on_drop);
    html_element.removeEventListener("dragend", this._on_dragend);
    super.unbind();
  }
}