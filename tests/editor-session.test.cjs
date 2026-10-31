const {test}=require('node:test');
const assert=require('node:assert/strict');
const Editor=require('../docscan/web/editor.js');

// Small DOM/canvas adapter: these tests exercise the real controller's event
// handlers and requests. Raster output is checked separately against OpenCV.
function environment(){
  const stored=new Map(),errors=[],requests=[],listeners=new Map();
  const globals=['document','window','Image','ResizeObserver','fetch'];
  const previous=new Map(globals.map(key=>[key,globalThis[key]]));
  class Classes extends Set{
    contains(value){return this.has(value);}
    remove(...values){values.forEach(value=>this.delete(value));}
    toggle(value,force){if(force??!this.has(value))this.add(value);else this.delete(value);}
  }
  class Element{
    constructor(id,tag='BUTTON'){
      this.id=id;this.tagName=tag;this.classList=new Classes();this.dataset={};this.style={};this.events=new Map();
      this.clientWidth=800;this.clientHeight=600;this.captured=new Set();this.value='32';this.context={arcs:[],lines:[]};
      const context=this.context;
      this.context.arc=(...args)=>context.arcs.push(args);this.context.lineTo=(...args)=>context.lines.push(args);
      this.context=new Proxy(this.context,{get:(target,key)=>key in target?target[key]:()=>{}});
    }
    setAttribute(name,value){this[name]=value;}
    removeAttribute(name){delete this[name];}
    addEventListener(name,fn){this.events.set(name,fn);}
    getContext(){return this.context;}
    getBoundingClientRect(){return {left:100,top:50};}
    getClientRects(){return [1];}
    setPointerCapture(id){this.captured.add(id);}
    hasPointerCapture(id){return this.captured.has(id);}
    releasePointerCapture(id){this.captured.delete(id);}
    toBlob(callback){callback(new Blob(['mask'],{type:'image/png'}));}
    focus(){document.activeElement=this;}
    closest(selector){
      if(selector==='.handle')return this.classList.has('handle')?this:null;
      return ['INPUT','SELECT','TEXTAREA','BUTTON'].includes(this.tagName)?this:null;
    }
    querySelectorAll(selector){
      return selector==='[data-tool]'?tools:controls;
    }
    fire(name,event){this.events.get(name)?.({preventDefault(){},...event});}
  }
  const ids=['edModal','wrap','edCanvas','rectBox','q0','q1','q2','q3','applyBtn','editorStatus',
    'cancelBtn','prevBtn','nextBtn','fitBtn','pageInfo','rectPanel','perspPanel','brushPanel','rotatePanel',
    'brushHud','rectUi','quadUi','rotL','rotR','rotM1','rotP1','rotReset','aspectSel','resetSelection',
    'resetQuad','clearMask','resetFile','brushSize','brushSizeLbl','editPreview','editPreviewStatus','previewFull',
    'autoCrop','expandCrop','refreshPreview','editorTextStrength','editorTextStrengthVal','previewMode'];
  const elements=new Map(ids.map(id=>[id,new Element(id,id.endsWith('Btn')||id.startsWith('rot')?'BUTTON':'DIV')]));
  elements.get('aspectSel').tagName='SELECT';elements.get('brushSize').tagName='INPUT';
  elements.get('previewMode').tagName='SELECT';elements.get('editorTextStrength').tagName='INPUT';
  const tools=['pan','rect','persp','brush','rotate'].map(tool=>{const element=new Element(tool);element.dataset.tool=tool;return element;});
  const controls=[...elements.values()].filter(el=>['BUTTON','SELECT','INPUT'].includes(el.tagName)).concat(tools);
  const files=[new File(['one'],'one.png'),new File(['two'],'two.png')];
  globalThis.document={getElementById:id=>elements.get(id),createElement:tag=>new Element(tag,tag.toUpperCase()),body:new Element('body','BODY'),activeElement:new Element('thumb','DIV')};
  globalThis.window={devicePixelRatio:2,addEventListener:(name,fn)=>listeners.set(name,fn)};
  const decodes=[];
  globalThis.Image=class{
    naturalWidth=640;naturalHeight=480;
    decode(){return environment.deferImages?new Promise(resolve=>decodes.push(resolve)):Promise.resolve();}
  };
  globalThis.ResizeObserver=class{observe(){}};
  globalThis.fetch=async(url,options)=>{requests.push({url,options});return {ok:true,blob:async()=>new Blob(['edited'],{type:'image/png'})};};
  const options={mode:'bw',long_side:2400,text_strength:.5,align:true,trim:true,trim_margin:0,trim_pad:12};
  const editor=new Editor({getFiles:()=>files,getEdited:i=>stored.get(i),onSave:(i,file)=>stored.set(i,file),
    onReset:i=>stored.delete(i),onError:e=>errors.push(e),errorMessage:async()=> 'Ошибка сохранения',getOptions:()=>options,
    onTextStrength:value=>options.text_strength=value,onMode:value=>options.mode=value});
  return {editor,elements,stored,errors,requests,decodes,listeners,options,restore(){
    editor.close();environment.deferImages=false;previous.forEach((value,key)=>{if(value===undefined)delete globalThis[key];else globalThis[key]=value;});
  }};
}

test('opening and fitting an image never creates a hidden crop',async()=>{
  const env=environment();try{
    await env.editor.open();assert.equal(env.editor.state.mode,null);assert.equal(env.editor.state.dirty,false);
    await env.editor.apply();assert.equal(env.requests.length,0);
    env.editor.rotate(90);await env.editor.apply();
    const settings=JSON.parse(env.requests[0].options.body.get('settings'));
    assert.deepEqual(settings,{rotation:90});assert.ok(env.stored.has(0));assert.equal(env.editor.state.dirty,false);
  }finally{env.restore();}
});

test('drafts survive page navigation, close and reopen; original reset affects one file',async()=>{
  const env=environment();try{
    const ed=env.editor;await ed.open();ed.selectTool('rect');ed.state.rect={x:50,y:60,w:300,h:200};ed.rotate(90);
    const draft=ed.state;await ed.load(1);ed.rotate(-1);const other=ed.state;
    await ed.load(0);assert.equal(ed.state,draft);ed.close();await ed.open(0);assert.equal(ed.state,draft);
    env.elements.get('resetFile').onclick();await Promise.resolve();await Promise.resolve();
    assert.notEqual(ed.state,draft);assert.equal(ed.state.rotation,0);assert.equal(ed.drafts.get(1),other);
  }finally{env.restore();}
});

test('resize and fit preserve the selection in photo pixels',async()=>{
  const env=environment();try{
    const ed=env.editor;await ed.open();ed.selectTool('rect');ed.state.rect={x:50,y:60,w:300,h:200};
    const expected={...ed.state.rect};env.elements.get('wrap').clientWidth=320;ed.fit();
    assert.deepEqual(ed.state.rect,expected);assert.equal(ed.canvas.width,640);
    await ed.apply();assert.deepEqual(JSON.parse(env.requests[0].options.body.get('settings')).crop,[50,60,300,200]);
  }finally{env.restore();}
});

test('fine rotations and reset do not gradually enlarge the crop',async()=>{
  const env=environment();try{
    const ed=env.editor;await ed.open();ed.selectTool('rect');ed.state.rect={x:100,y:120,w:300,h:200};ed.anchorRect();
    const original={...ed.state.rect};
    for(let i=0;i<20;i++){ed.rotate(1);ed.rotate(-1);}
    for(const key of ['x','y','w','h'])assert.ok(Math.abs(ed.state.rect[key]-original[key])<1e-8);
    ed.rotate(31);ed.rotate(-31);
    for(const key of ['x','y','w','h'])assert.ok(Math.abs(ed.state.rect[key]-original[key])<1e-8);
  }finally{env.restore();}
});

test('dragging a crop handle uses wrapper coordinates and inverse zoom',async()=>{
  const env=environment();try{
    const ed=env.editor;await ed.open();ed.selectTool('rect');ed.state.rect={x:100,y:80,w:200,h:150};ed.view={scale:2,x:-30,y:10};
    const handle={classList:['handle','nw'],closest(){return this;}};
    ed.wrap.fire('pointerdown',{target:handle,button:0,pointerId:1,clientX:270,clientY:220,offsetX:1,offsetY:1});
    ed.wrap.fire('pointermove',{pointerId:1,clientX:290,clientY:260,offsetX:0,offsetY:0});
    assert.deepEqual(ed.state.rect,{x:110,y:100,w:190,h:130});
    ed.wrap.fire('pointercancel',{pointerId:1});assert.equal(ed.drag,null);assert.equal(ed.wrap.hasPointerCapture(1),false);
  }finally{env.restore();}
});

test('brush follows source pixels after rotation and zoom and connects strokes',async()=>{
  const env=environment();try{
    const ed=env.editor;await ed.open();ed.rotate(90);ed.selectTool('brush');ed.view={scale:.5,x:40,y:20};
    ed.wrap.fire('pointerdown',{target:ed.wrap,button:0,pointerId:1,clientX:363,clientY:76});
    const [x,y,r]=ed.state.mask.getContext('2d').arcs.at(-1);
    assert.ok(Math.abs(x-12)<1e-8);assert.ok(Math.abs(y-34)<1e-8);assert.equal(r,16);
    ed.wrap.fire('pointermove',{pointerId:1,clientX:350,clientY:80});
    assert.equal(ed.state.mask.getContext('2d').lines.length,2);assert.equal(ed.state.maskDirty,true);
    await ed.apply();assert.ok(env.requests[0].options.body.get('mask'));
  }finally{env.restore();}
});

test('failed save keeps the draft and can be retried without duplicate requests',async()=>{
  const env=environment();try{
    const ed=env.editor;await ed.open();ed.rotate(1);const draft=ed.state;
    let finish;globalThis.fetch=()=>new Promise(resolve=>finish=resolve);
    const pending=ed.apply();await ed.apply();assert.equal(ed.saving,true);
    finish({ok:false});await pending;
    assert.equal(ed.state,draft);assert.equal(ed.state.dirty,true);assert.equal(ed.saving,false);assert.equal(ed.applyButton.disabled,false);
    globalThis.fetch=async()=>({ok:true,blob:async()=>new Blob(['edited'],{type:'image/png'})});
    await ed.apply();assert.equal(ed.state.dirty,false);assert.ok(env.stored.has(0));
  }finally{env.restore();}
});

test('late image load cannot restore a closed editor',async()=>{
  const env=environment();try{
    environment.deferImages=true;const pending=env.editor.open();env.editor.close();env.decodes[0]();await pending;
    assert.equal(env.editor.isOpen(),false);assert.equal(env.editor.image,null);assert.equal(env.editor.state,null);
  }finally{env.restore();}
});

test('a slow previous page cannot replace the last requested page',async()=>{
  const env=environment();try{
    environment.deferImages=true;const first=env.editor.open(0),second=env.editor.load(1);
    env.decodes[1]();await second;const page=env.editor.state;
    env.decodes[0]();await first;
    assert.equal(env.editor.index,1);assert.equal(env.editor.state,page);assert.equal(page.source.name,'two.png');
  }finally{env.restore();}
});

test('closing during a save aborts the request and a late response cannot replace the file',async()=>{
  const env=environment();try{
    const ed=env.editor;await ed.open();ed.rotate(1);const draft=ed.state;
    let finish,signal;
    globalThis.fetch=(url,options)=>{signal=options.signal;return new Promise(resolve=>finish=resolve);};
    const pending=ed.apply();await Promise.resolve();ed.close();assert.equal(signal.aborted,true);
    finish({ok:true,blob:async()=>new Blob(['late'],{type:'image/png'})});await pending;
    assert.equal(env.stored.size,0);assert.equal(ed.isOpen(),false);
    await ed.open();assert.equal(ed.state,draft);assert.equal(ed.state.dirty,true);
  }finally{env.restore();}
});

test('select/input keyboard arrows and Enter never navigate or apply edits',async()=>{
  const env=environment();try{
    const ed=env.editor;await ed.open();ed.rotate(1);
    for(const key of ['ArrowRight','Enter'])ed.keydown({key,target:env.elements.get('aspectSel'),preventDefault(){}});
    assert.equal(ed.index,0);assert.equal(env.requests.length,0);
  }finally{env.restore();}
});

test('auto crop creates an editable suggestion on the full original photo',async()=>{
  const env=environment();try{
    const ed=env.editor;await ed.open();const source=ed.state.source;
    globalThis.fetch=async()=>({ok:true,json:async()=>({kind:'persp',found:true,trimmed:true,
      quad:[[100,80],[500,80],[500,400],[100,400]],crop:[100,80,400,320]})});
    await ed.autoDetect();
    assert.equal(ed.state.source,source);assert.equal(env.stored.size,0);
    assert.equal(ed.state.mode,'persp');assert.equal(ed.tool,'persp');assert.equal(ed.state.dirty,true);
    const before=ed.state.quad.map(p=>({...p}));ed.expandSelection();
    assert.ok(ed.state.quad[0].x<before[0].x && ed.state.quad[0].y<before[0].y);
    assert.ok(ed.state.quad[2].x>before[2].x && ed.state.quad[2].y>before[2].y);
  }finally{env.restore();}
});

test('more margins can recover pixels outside the suggested rectangle',async()=>{
  const env=environment();try{
    const ed=env.editor;await ed.open();ed.selectTool('rect');ed.state.rect={x:100,y:80,w:200,h:150};
    const source=ed.state.source;ed.expandSelection();
    assert.deepEqual(ed.state.rect,{x:84,y:68,w:232,h:174});assert.equal(ed.state.source,source);
  }finally{env.restore();}
});

test('preview uses current geometry and shared enhancer settings without committing a crop',async()=>{
  const env=environment();try{
    const ed=env.editor;await ed.open();ed.selectTool('rect');ed.state.rect={x:50,y:60,w:300,h:200};
    env.options.mode='gray';env.options.text_strength=.8;ed.refreshOptions();
    await ed.renderPreview(ed.previewToken);
    const request=env.requests[0],url=new URL(request.url,'http://local');
    assert.equal(url.pathname,'/preview');assert.equal(url.searchParams.get('text_strength'),'0.8');
    assert.equal(url.searchParams.get('mode'),'gray');
    assert.deepEqual(JSON.parse(request.options.body.get('settings')).crop,[50,60,300,200]);
    assert.equal(env.stored.size,0);assert.ok(ed.previewUrl);assert.equal(ed.previewLink.href,ed.previewUrl);
  }finally{env.restore();}
});

test('old preview responses cannot replace a more recent crop preview',async()=>{
  const env=environment();try{
    const ed=env.editor;await ed.open();let finish,signal;
    globalThis.fetch=(url,options)=>{signal=options.signal;return new Promise(resolve=>finish=resolve);};
    const old=ed.renderPreview(ed.previewToken);await Promise.resolve();
    ed.selectTool('rect');ed.state.rect={x:50,y:60,w:300,h:200};ed.changed();
    assert.equal(signal.aborted,true);
    globalThis.fetch=async()=>({ok:true,blob:async()=>new Blob(['new'],{type:'image/png'})});
    await ed.renderPreview(ed.previewToken);const latest=ed.previewUrl;
    finish({ok:true,blob:async()=>new Blob(['old'],{type:'image/png'})});await old;
    assert.equal(ed.previewUrl,latest);
  }finally{env.restore();}
});

test('failed preview keeps manual edits and reports the error inside the preview',async()=>{
  const env=environment();try{
    const ed=env.editor;await ed.open();ed.rotate(1);const state=ed.state;
    globalThis.fetch=async()=>({ok:false});await ed.renderPreview(ed.previewToken);
    assert.equal(ed.state,state);assert.equal(ed.state.dirty,true);
    assert.equal(ed.previewStatus.textContent,'Ошибка сохранения');assert.equal(ed.previewLink.classList.contains('hidden'),true);
  }finally{env.restore();}
});

test('auto detection response from another page cannot overwrite current edits',async()=>{
  const env=environment();try{
    const ed=env.editor;await ed.open();let finish;
    globalThis.fetch=()=>new Promise(resolve=>finish=resolve);
    const old=ed.autoDetect();await ed.load(1);ed.rotate(1);const current=ed.state;
    finish({ok:true,json:async()=>({kind:'rect',found:false,trimmed:true,
      quad:[[10,10],[600,10],[600,400],[10,400]],crop:[10,10,590,390]})});await old;
    assert.equal(ed.index,1);assert.equal(ed.state,current);assert.equal(current.rotation,1);assert.equal(current.mode,null);
  }finally{env.restore();}
});

test('auto review detects each new page and preserves drafts when returning',async()=>{
  const env=environment();try{
    let detections=0;
    globalThis.fetch=async()=>{detections++;return {ok:true,json:async()=>({kind:'rect',found:false,trimmed:true,
      quad:[[10,10],[600,10],[600,400],[10,400]],crop:[10,10,590,390]})};};
    const ed=env.editor;await ed.open(0,{auto:true});const draft=ed.state;
    assert.equal(detections,1);await ed.load(1);assert.equal(detections,2);
    await ed.load(0);assert.equal(ed.state,draft);assert.equal(detections,2);
  }finally{env.restore();}
});
