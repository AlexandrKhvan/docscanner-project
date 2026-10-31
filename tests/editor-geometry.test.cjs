const {test}=require('node:test');
const assert=require('node:assert/strict');
const G=require('../docscan/web/editor-geometry.js');
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-8,`${a} != ${b}`);

test('quarter turn maps source edges into exact rotated bounds',()=>{
  const layout=G.rotationLayout(640,480,90);
  assert.equal(layout.width,480);assert.equal(layout.height,640);
  const point=G.transform({x:12,y:34},layout.matrix);near(point.x,446);near(point.y,12);
});

for(const degrees of [-180,-91,-45,0,1,31,90,180]){
  test(`pointer maps back to the same photo pixel after ${degrees} degree rotation, zoom and pan`,()=>{
    const matrix=G.rotationLayout(640,480,degrees).matrix;
    const source={x:123.25,y:217.5},view={scale:2.5,x:-120,y:73};
    const screen=G.toScreen(G.transform(source,matrix),view);
    const pointer=G.eventPoint({clientX:screen.x+200,clientY:screen.y+80,offsetX:1,offsetY:1},{left:200,top:80});
    const restored=G.inverse(G.fromScreen(pointer,view),matrix);
    near(restored.x,source.x);near(restored.y,source.y);
  });
}

test('crop cannot cross its fixed opposite corner or image bounds',()=>{
  const rect={x:100,y:80,w:200,h:150},bounds={width:640,height:480};
  assert.deepEqual(G.resizeRect(rect,'nw',1000,1000,bounds),{x:290,y:220,w:10,h:10});
  assert.deepEqual(G.resizeRect(rect,'nw',-1000,-1000,bounds),{x:0,y:0,w:300,h:230});
  assert.deepEqual(G.resizeRect(rect,'move',1000,-1000,bounds),{x:440,y:0,w:200,h:150});
});

test('aspect-constrained corner keeps opposite corner fixed',()=>{
  const result=G.resizeRect({x:100,y:80,w:200,h:100},'nw',-50,-25,{width:640,height:480},2);
  near(result.x+result.w,300);near(result.y+result.h,180);near(result.w/result.h,2);
});

test('edge resize maintains aspect and perpendicular centre',()=>{
  const result=G.resizeRect({x:100,y:100,w:200,h:100},'e',80,0,{width:640,height:480},2);
  near(result.x,100);near(result.y+result.h/2,150);near(result.w,280);near(result.h,140);
});

test('all resize handles respect bounds and aspect under extreme dragging',()=>{
  for(const handle of ['n','s','e','w','nw','ne','sw','se']){
    for(const ratio of [null,1,210/297,4/3]){
      for(const delta of [-10000,-90,0,90,10000]){
        const rect=G.fitAspect({x:100,y:100,w:300,h:200},ratio);
        const result=G.resizeRect(rect,handle,delta,delta,{width:640,height:480},ratio);
        assert.ok(result.x>=-1e-7 && result.y>=-1e-7);
        assert.ok(result.x+result.w<=640+1e-7 && result.y+result.h<=480+1e-7);
        assert.ok(result.w>0 && result.h>0);
        if(ratio)near(result.w/result.h,ratio);
      }
    }
  }
});

test('crossed, concave and collapsed quads cannot be applied',()=>{
  const points=[{x:0,y:0},{x:100,y:0},{x:100,y:100},{x:0,y:100}];
  assert.equal(G.validQuad(points),true);
  assert.equal(G.validQuad([points[0],points[2],points[1],points[3]]),false);
  assert.equal(G.validQuad([points[0],points[1],{x:10,y:10},points[3]]),false);
  assert.equal(G.validQuad([points[0],points[1],points[1],points[3]]),false);
});

test('diamond keeps four unique corners in clockwise order',()=>{
  const result=G.orderQuad([{x:0,y:50},{x:50,y:100},{x:100,y:50},{x:50,y:0}]);
  assert.deepEqual(result,[{x:50,y:0},{x:100,y:50},{x:50,y:100},{x:0,y:50}]);
});
