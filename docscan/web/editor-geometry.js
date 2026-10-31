/* Image-edge coordinates throughout. The viewport never changes saved geometry. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.DocScanGeometry = factory();
})(globalThis, function () {
  'use strict';
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  function rotationLayout(width, height, degrees) {
    const radians = degrees * Math.PI / 180;
    const c = Math.cos(radians), s = Math.sin(radians);
    const w = Math.max(1, Math.ceil(Math.abs(width*c) + Math.abs(height*s) - 1e-9));
    const h = Math.max(1, Math.ceil(Math.abs(height*c) + Math.abs(width*s) - 1e-9));
    return {width:w, height:h, matrix:[c, s, -s, c, w/2-c*width/2+s*height/2, h/2-s*width/2-c*height/2]};
  }
  function transform(point, matrix) {
    const [a,b,c,d,e,f] = matrix;
    return {x:a*point.x+c*point.y+e, y:b*point.x+d*point.y+f};
  }
  function inverse(point, matrix) {
    const [a,b,c,d,e,f] = matrix, determinant = a*d-b*c;
    const x = point.x-e, y = point.y-f;
    return {x:(d*x-c*y)/determinant, y:(a*y-b*x)/determinant};
  }
  function toScreen(point, view) {
    return {x:point.x*view.scale+view.x, y:point.y*view.scale+view.y};
  }
  function fromScreen(point, view) {
    return {x:(point.x-view.x)/view.scale, y:(point.y-view.y)/view.scale};
  }
  function eventPoint(event, bounds) {
    return {x:event.clientX-bounds.left, y:event.clientY-bounds.top};
  }
  function corners(rect) {
    return [{x:rect.x,y:rect.y}, {x:rect.x+rect.w,y:rect.y},
      {x:rect.x+rect.w,y:rect.y+rect.h}, {x:rect.x,y:rect.y+rect.h}];
  }
  function boundingRect(points) {
    const xs=points.map(p=>p.x), ys=points.map(p=>p.y);
    const x=Math.min(...xs), y=Math.min(...ys);
    return {x,y,w:Math.max(...xs)-x,h:Math.max(...ys)-y};
  }
  function orderQuad(points) {
    const center = points.reduce((sum,p)=>({x:sum.x+p.x/4,y:sum.y+p.y/4}),{x:0,y:0});
    const sorted = points.slice().sort((a,b)=>Math.atan2(a.y-center.y,a.x-center.x)-Math.atan2(b.y-center.y,b.x-center.x));
    let first=0;
    sorted.forEach((p,i)=>{ const best=sorted[first]; if (p.x+p.y<best.x+best.y-1e-6 ||
      (Math.abs(p.x+p.y-best.x-best.y)<1e-6 && p.y<best.y)) first=i; });
    return sorted.slice(first).concat(sorted.slice(0,first));
  }
  function validQuad(points) {
    if (points.length!==4 || points.some(p=>!Number.isFinite(p.x)||!Number.isFinite(p.y))) return false;
    return points.every((p,i)=>{
      const next=points[(i+1)%4], last=points[(i+2)%4];
      return Math.hypot(next.x-p.x,next.y-p.y)>=2 &&
        (next.x-p.x)*(last.y-next.y)-(next.y-p.y)*(last.x-next.x)>1e-6;
    });
  }
  function fitAspect(rect, ratio) {
    if (!ratio) return {...rect};
    let w=rect.w,h=rect.h;
    if (w/h>ratio) w=h*ratio; else h=w/ratio;
    return {x:rect.x+(rect.w-w)/2,y:rect.y+(rect.h-h)/2,w,h};
  }
  function resizeRect(rect, handle, dx, dy, bounds, ratio=null) {
    if (handle==='move') return {...rect,x:clamp(rect.x+dx,0,bounds.width-rect.w),y:clamp(rect.y+dy,0,bounds.height-rect.h)};
    const sx=handle.includes('w')?-1:handle.includes('e')?1:0;
    const sy=handle.includes('n')?-1:handle.includes('s')?1:0;
    const minimum=Math.min(10,bounds.width,bounds.height);
    if (!ratio) {
      let left=rect.x,top=rect.y,right=rect.x+rect.w,bottom=rect.y+rect.h;
      if(sx<0) left=clamp(left+dx,0,right-minimum);
      if(sx>0) right=clamp(right+dx,left+minimum,bounds.width);
      if(sy<0) top=clamp(top+dy,0,bottom-minimum);
      if(sy>0) bottom=clamp(bottom+dy,top+minimum,bounds.height);
      return {x:left,y:top,w:right-left,h:bottom-top};
    }
    const anchorX=sx<0?rect.x+rect.w:sx>0?rect.x:rect.x+rect.w/2;
    const anchorY=sy<0?rect.y+rect.h:sy>0?rect.y:rect.y+rect.h/2;
    const maxW=sx<0?anchorX:sx>0?bounds.width-anchorX:2*Math.min(anchorX,bounds.width-anchorX);
    const maxH=sy<0?anchorY:sy>0?bounds.height-anchorY:2*Math.min(anchorY,bounds.height-anchorY);
    let w=(!sy || (sx && Math.abs(dx)>=Math.abs(dy*ratio))) ? rect.w+sx*dx : (rect.h+sy*dy)*ratio;
    const maximum=Math.min(maxW,maxH*ratio);
    w=clamp(w,Math.min(maximum,Math.max(minimum,minimum*ratio)),maximum);
    const h=w/ratio;
    return {x:sx<0?anchorX-w:sx>0?anchorX:anchorX-w/2,
      y:sy<0?anchorY-h:sy>0?anchorY:anchorY-h/2,w,h};
  }
  return {clamp,rotationLayout,transform,inverse,toScreen,fromScreen,eventPoint,corners,boundingRect,orderQuad,validQuad,fitAspect,resizeRect};
});
