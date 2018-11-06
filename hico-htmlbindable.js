export class HTMLBindable {
  constructor(html_element) {
    if(new.target === HTMLBindable)
      throw "Cannot construct Bindable instances directly";
    if(!html_element instanceof HTMLElement)
      throw "Argument of HTMLBindable must be a HTMLElement";
    this._raw_html_element = html_element;
  }
  
  get raw_html_element() {
    return this._raw_html_element;
  }

  unbind() {
    if(!this.raw_html_element)
      throw new TypeError("Not binded");
    this._raw_html_element = null;
  }

  map_screen_to_element(y, x) {
    let rect = this.raw_html_element.getBoundingClientRect();
    return [y - rect.top, x - rect.left];
  }

  map_element_to_screen(ey, ex) {
    let rect = this.raw_html_element.getBoundingClientRect();
    return [ey + rect.top, ex + rect.left];
  }

  static bind_to(x) {
    if(Symbol.iterator in x) {
      const l = [];
      for(let v of x)
        l.push(bind_to(v));
      return l;
    }
    else
      return new this(x);
  }

  static bind_to_id(element_id) {
    const x = document.getElementById(element_id);
    if(!x)
      throw "No such element id:" + element_id;
    return this.bind_to(x);
  }

  static bind_to_class(class_name) {
    const x = document.getElementsByClassName(class_name);
    return this.bind_to(x);
  }
}