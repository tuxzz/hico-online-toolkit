export class Rect {
  constructor(y, x, h, w) {
    this.y = y;
    this.x = x;
    this.h = h;
    this.w = w;
  }

  normalize() {
    if(this.h < 0) {
      this.y += this.h;
      this.h = -this.h;
    }
    if(this.w < 0) {
      this.x += this.w;
      this.w = -this.w;
    }
  }

  normalized() {
    const o = this.clone();
    o.normalize();
    return o;
  }

  clone() {
    return new Rect(this.y, this.x, this.h, this.w);
  }

  get top() {
    return this.h >= 0 ? this.y : this.y + this.h;
  }

  get bottom() {
    return this.h >= 0 ? this.y + this.h : this.y;
  }

  get left() {
    return this.w >= 0 ? this.x : this.x + this.w;
  }

  get right() {
    return this.w >= 0 ? this.x + this.w : this.x;
  }

  set top(v) {
    this.normalize();
    const d = v - this.top;
    this.y += d;
    this.h -= d;
  }

  set bottom(v) {
    this.normalize();
    this.h = v - this.top;
  }

  set left(v) {
    this.normalize();
    const d = v - this.left;
    this.x += d;
    this.w -= d;
  }

  set right(v) {
    this.normalize();
    this.w = v - this.left;
  }

  static from_yxhw(y, x, h, w) {
    const v = new Rect(y, x, h, w);
    v.normalize();
    return v;
  }

  static from_tblr(t, b, l, r) {
    const v = new Rect(t, b - t, l, r - l);
    v.normalize();
    return v;
  }
}