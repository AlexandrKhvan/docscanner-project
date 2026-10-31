(function (root, factory) {
  if (typeof module==='object' && module.exports) module.exports=factory(require('./editor-geometry.js'));
  else root.DocScanEditor=factory(root.DocScanGeometry);
})(globalThis, function (G) {
  'use strict';
  return class DocScanEditor {
    constructor({getFiles, getEdited, onSave, onReset, onError, errorMessage,
      getOptions=()=>({mode:'bw',long_side:2400,text_strength:.5,align:true,trim:true}), onTextStrength=()=>{}, onMode=()=>{}}) {
      Object.assign(this,{getFiles,getEdited,onSave,onReset,onError,errorMessage,getOptions,onTextStrength,onMode});
      this.modal=document.getElementById('edModal');
      this.wrap=document.getElementById('wrap');
      this.canvas=document.getElementById('edCanvas');
      this.ctx=this.canvas.getContext('2d');
      this.rectBox=document.getElementById('rectBox');
      this.points=[0,1,2,3].map(i=>document.getElementById('q'+i));
      this.applyButton=document.getElementById('applyBtn');
      this.status=document.getElementById('editorStatus');
      this.previewImage=document.getElementById('editPreview');
      this.previewStatus=document.getElementById('editPreviewStatus');
      this.previewLink=document.getElementById('previewFull');
      this.previewToken=0;this.detectToken=0;this.detecting=false;
      this.drafts=new Map(); this.index=0; this.token=0; this.saveToken=0; this.saving=false;
      this.view={scale:1,x:0,y:0}; this.tool='pan'; this.drag=null; this.space=false;
      this.bind(); this.updateTools();
      this.observer=new ResizeObserver(()=>{if(this.isOpen() && this.image) this.fit();});
      this.observer.observe(this.wrap);
    }
    isOpen(){return this.modal.classList.contains('show');}
    source(index=this.index){return this.getEdited(index)||this.getFiles()[index];}
    async open(index=0,{auto=false}={}){
      if(this.saving || !this.getFiles().length) return;
      this.returnFocus=document.activeElement;
      this.autoReview=auto;
      this.modal.classList.add('show'); this.modal.setAttribute('aria-hidden','false');
      document.body.classList.add('editor-open');
      await this.load(index);
      if(this.isOpen()) document.getElementById('cancelBtn').focus();
    }
    close(){
      this.saveToken++; this.abortSave?.abort(); this.saving=false;
      this.cancelPreview();this.cancelDetection();
      this.endDrag(); this.token++; this.space=false;
      this.modal.classList.remove('show'); this.modal.setAttribute('aria-hidden','true');
      document.body.classList.remove('editor-open');
      this.image=null; this.state=null;
      const focus=this.returnFocus?.isConnected?this.returnFocus:document.getElementById('btnEdit');
      focus?.focus();
    }
    clear(){this.close(); this.drafts.clear(); this.image=null; this.state=null;}
    async load(index){
      if(this.saving || !this.isOpen()) return;
      this.endDrag(); this.index=index;
      this.cancelPreview();this.cancelDetection();
      const token=++this.token, file=this.source(index);
      this.image=null; this.state=null; this.tool='pan'; this.setStatus('Открываю изображение…'); this.controls(); this.draw();
      const url=URL.createObjectURL(file), image=new Image();
      try {
        image.src=url; await image.decode();
        if(token!==this.token || !this.isOpen()) return;
        this.image=image;
        const saved=this.drafts.get(index);
        if(saved?.source===file) this.state=saved;
        else {
          const mask=document.createElement('canvas'); mask.width=0; mask.height=0;
          this.state={source:file,width:image.naturalWidth,height:image.naturalHeight,rotation:0,
            mode:null,aspect:'free',mask,maskDirty:false,dirty:false};
          this.resetSelection(); this.drafts.set(index,this.state);
        }
        document.getElementById('aspectSel').value=this.state.aspect;
        this.updateTools(); this.fit(); this.controls(); this.describe(); this.refreshOptions();
        if(this.autoReview && !this.getEdited(index) && !this.state.dirty) await this.autoDetect();
      } catch(error) {
        if(token!==this.token || !this.isOpen()) return;
        this.setStatus('Не удалось открыть изображение. Выберите другую страницу.');
        this.onError('Не удалось открыть изображение'); this.controls();
      } finally {URL.revokeObjectURL(url);}
    }
    layout(){return G.rotationLayout(this.state.width,this.state.height,this.state.rotation);}
    defaultQuad(){
      return G.orderQuad(G.corners({x:0,y:0,w:this.state.width,h:this.state.height}).map(p=>G.transform(p,this.layout().matrix)));
    }
    resetSelection(){
      const layout=this.layout();
      this.state.rect={x:0,y:0,w:layout.width,h:layout.height}; this.state.quad=this.defaultQuad(); this.state.rectAnchor=null;
    }
    anchorRect(){this.state.rectAnchor=G.corners(this.state.rect).map(p=>G.inverse(p,this.layout().matrix));}
    setStatus(message){this.status.textContent=message;}
    describe(){
      if(!this.state) return;
      const s=this.state, selection=s.mode==='rect'?' · обрезка':s.mode==='persp'?' · четыре угла':'';
      this.setStatus(`${s.width} × ${s.height} px · ${s.rotation}°${selection} · ${s.dirty?'есть неприменённые правки':'нет неприменённых правок'}`);
    }
    changed(){this.state.dirty=true; this.describe(); this.draw(); this.controls();this.schedulePreview();}
    controls(){
      const unavailable=!this.image || this.saving || this.detecting;
      this.modal.querySelectorAll('button,input,select').forEach(el=>el.disabled=unavailable);
      document.getElementById('cancelBtn').disabled=false;
      document.getElementById('cancelBtn').textContent=this.saving||this.detecting?'Отменить обработку (Esc)':'Закрыть (Esc)';
      document.getElementById('prevBtn').disabled=this.saving||this.index===0;
      document.getElementById('nextBtn').disabled=this.saving||this.index>=this.getFiles().length-1;
      this.applyButton.disabled=unavailable||!this.state?.dirty;
      this.applyButton.textContent=this.saving?'Применяю…':'Применить (Enter)';
      document.getElementById('pageInfo').textContent=`${this.index+1} / ${this.getFiles().length}`;
      this.wrap.classList.toggle('locked',unavailable);
    }
    updateTools(){
      this.modal.querySelectorAll('[data-tool]').forEach(el=>{
        el.classList.toggle('active',el.dataset.tool===this.tool); el.setAttribute('aria-pressed',String(el.dataset.tool===this.tool));
      });
      ['rect','persp','brush','rotate'].forEach(t=>document.getElementById(t+'Panel').classList.toggle('hidden',t!==this.tool));
      document.getElementById('brushHud').classList.toggle('hidden',this.tool!=='brush');
      document.getElementById('rectUi').classList.toggle('hidden',!this.state||this.state.mode!=='rect'||this.tool!=='rect');
      document.getElementById('quadUi').classList.toggle('hidden',!this.state||this.state.mode!=='persp'||this.tool!=='persp');
      this.wrap.dataset.tool=this.tool;
    }
    selectTool(tool){
      if(!this.image || this.saving) return;
      this.endDrag(); this.tool=tool;
      if(tool==='rect' && this.state.mode!=='rect'){
        if(this.state.mode==='persp') this.state.rect=G.boundingRect(this.state.quad);
        this.state.mode='rect'; this.state.rect=G.fitAspect(this.state.rect,this.ratio()); this.anchorRect(); this.changed();
      } else if(tool==='persp' && this.state.mode!=='persp'){
        if(this.state.mode==='rect') this.state.quad=G.corners(this.state.rect);
        this.state.mode='persp'; this.changed();
      }
      this.updateTools(); this.draw();
    }
    ratio(){
      const value=this.state.aspect;
      if(value==='free') return null;
      if(value==='A4') return 210/297;
      const [w,h]=value.split(':').map(Number); return w/h;
    }
    fit(){
      if(!this.image) return;
      const {width,height}=this.layout(), cw=this.wrap.clientWidth,ch=this.wrap.clientHeight;
      if(!cw||!ch) return;
      const scale=Math.min(cw/width,ch/height)*.9;
      this.minimumScale=scale*.2; this.maximumScale=Math.max(8,scale*8);
      this.view={scale,x:(cw-width*scale)/2,y:(ch-height*scale)/2}; this.draw();
    }
    pointer(event){return G.eventPoint(event,this.wrap.getBoundingClientRect());}
    scenePoint(event){return G.fromScreen(this.pointer(event),this.view);}
    sourcePoint(event){return G.inverse(this.scenePoint(event),this.layout().matrix);}
    rotate(delta){
      if(!this.image||this.saving||this.detecting) return;
      const old=this.layout(), state=this.state;
      const project=p=>G.transform(G.inverse(p,old.matrix),this.layout().matrix);
      if(state.mode==='rect' && !state.rectAnchor) this.anchorRect();
      state.rotation=((state.rotation+delta)%360+360)%360;
      if(state.rotation>180) state.rotation-=360;
      const layout=this.layout();
      if(state.mode==='rect'){
        const box=G.boundingRect(state.rectAnchor.map(p=>G.transform(p,layout.matrix)));
        const x=G.clamp(box.x,0,layout.width),y=G.clamp(box.y,0,layout.height);
        state.rect=G.fitAspect({x,y,w:Math.min(box.w,layout.width-x),h:Math.min(box.h,layout.height-y)},this.ratio());
      }
      if(state.mode==='persp') state.quad=G.orderQuad(state.quad.map(project));
      if(!state.mode) this.resetSelection();
      this.changed(); this.fit();
    }
    brush(event, start=false){
      const point=this.sourcePoint(event), state=this.state, previous=start?point:this.drag.last;
      if(!state.mask.width){state.mask.width=state.width;state.mask.height=state.height;}
      const ctx=state.mask.getContext('2d'), radius=Number(document.getElementById('brushSize').value)/2;
      ctx.save(); ctx.globalCompositeOperation=this.drag.erase?'destination-out':'source-over';
      ctx.strokeStyle=ctx.fillStyle='#fff'; ctx.lineWidth=radius*2; ctx.lineCap='round'; ctx.lineJoin='round';
      ctx.beginPath(); ctx.moveTo(previous.x,previous.y); ctx.lineTo(point.x,point.y); ctx.stroke();
      ctx.beginPath(); ctx.arc(point.x,point.y,radius,0,Math.PI*2); ctx.fill(); ctx.restore();
      this.drag.last=point; state.maskDirty=true; this.changed();
    }
    endDrag(){
      const id=this.drag?.pointerId; this.drag=null;
      if(id!==undefined && this.wrap.hasPointerCapture(id)) this.wrap.releasePointerCapture(id);
      this.wrap.classList.remove('drag');
    }
    bind(){
      this.modal.querySelectorAll('[data-tool]').forEach(el=>el.onclick=()=>this.selectTool(el.dataset.tool));
      document.getElementById('cancelBtn').onclick=()=>this.close();
      document.getElementById('prevBtn').onclick=()=>{if(this.index>0)this.load(this.index-1);};
      document.getElementById('nextBtn').onclick=()=>{if(this.index+1<this.getFiles().length)this.load(this.index+1);};
      document.getElementById('fitBtn').onclick=()=>this.fit();
      this.applyButton.onclick=()=>this.apply();
      document.getElementById('autoCrop').onclick=()=>this.autoDetect();
      document.getElementById('expandCrop').onclick=()=>this.expandSelection();
      document.getElementById('refreshPreview').onclick=()=>this.schedulePreview(0);
      document.getElementById('editorTextStrength').oninput=event=>{
        const value=Number(event.target.value)/100;
        document.getElementById('editorTextStrengthVal').textContent=Math.round(value*100)+'%';
        this.onTextStrength(value);this.schedulePreview();
      };
      document.getElementById('previewMode').onchange=event=>{this.onMode(event.target.value);this.schedulePreview();};
      document.getElementById('rotL').onclick=()=>this.rotate(-90);
      document.getElementById('rotR').onclick=()=>this.rotate(90);
      document.getElementById('rotM1').onclick=()=>this.rotate(-1);
      document.getElementById('rotP1').onclick=()=>this.rotate(1);
      document.getElementById('rotReset').onclick=()=>this.rotate(-this.state.rotation);
      document.getElementById('aspectSel').onchange=event=>{
        this.state.aspect=event.target.value; this.state.rect=G.fitAspect(this.state.rect,this.ratio()); this.anchorRect(); this.changed();
      };
      document.getElementById('resetSelection').onclick=()=>{
        this.state.mode=null; this.resetSelection(); this.changed(); this.updateTools();
      };
      document.getElementById('resetQuad').onclick=()=>{
        this.state.quad=this.defaultQuad(); this.changed();
      };
      document.getElementById('clearMask').onclick=()=>{
        this.state.mask.width=0; this.state.mask.height=0;
        this.state.maskDirty=false; this.changed();
      };
      document.getElementById('resetFile').onclick=()=>{
        if(this.saving)return; this.onReset(this.index); this.drafts.delete(this.index); this.load(this.index);
      };
      document.getElementById('brushSize').oninput=event=>document.getElementById('brushSizeLbl').textContent=event.target.value;
      this.wrap.addEventListener('contextmenu',event=>{if(this.tool==='brush')event.preventDefault();});
      this.wrap.addEventListener('wheel',event=>{
        event.preventDefault(); if(!this.image||this.saving||this.detecting)return;
        const screen=this.pointer(event),scene=G.fromScreen(screen,this.view);
        this.view.scale=G.clamp(this.view.scale*Math.exp(-event.deltaY*.001),this.minimumScale,this.maximumScale);
        this.view.x=screen.x-scene.x*this.view.scale; this.view.y=screen.y-scene.y*this.view.scale; this.draw();
      },{passive:false});
      this.wrap.addEventListener('pointerdown',event=>{
        if(!this.image||this.saving||this.detecting||this.drag||![0,1,2].includes(event.button))return;
        const pan=this.space||event.button===1||this.tool==='pan';
        if(event.button===2 && this.tool!=='brush')return;
        event.preventDefault(); this.wrap.setPointerCapture(event.pointerId);
        const base={pointerId:event.pointerId,start:this.pointer(event),view:{...this.view}};
        const handle=event.target.closest('.handle');
        const handleName=handle?[...handle.classList].find(name=>name!=='handle'):null;
        const pointIndex=this.points.indexOf(event.target);
        if(pan){this.drag={...base,type:'pan'};this.wrap.classList.add('drag');}
        else if(this.tool==='brush'){
          this.drag={...base,type:'brush',erase:event.button===2||event.altKey};this.brush(event,true);
        } else if(this.tool==='rect' && (handle||event.target===this.rectBox)){
          this.drag={...base,type:'rect',handle:handleName||'move',rect:{...this.state.rect},scene:this.scenePoint(event)};
        } else if(this.tool==='persp' && pointIndex>=0) this.drag={...base,type:'quad',index:pointIndex};
        else {this.drag={...base,type:'pan'};this.wrap.classList.add('drag');}
      });
      this.wrap.addEventListener('pointermove',event=>{
        const drag=this.drag; if(!drag||drag.pointerId!==event.pointerId)return;
        if(drag.type==='pan'){
          const point=this.pointer(event);this.view.x=drag.view.x+point.x-drag.start.x;this.view.y=drag.view.y+point.y-drag.start.y;this.draw();
        } else if(drag.type==='rect'){
          const point=this.scenePoint(event);
          this.state.rect=G.resizeRect(drag.rect,drag.handle,point.x-drag.scene.x,point.y-drag.scene.y,this.layout(),this.ratio());this.anchorRect();this.changed();
        } else if(drag.type==='quad'){
          const point=this.scenePoint(event),layout=this.layout();
          const quad=this.state.quad.slice();quad[drag.index]={x:G.clamp(point.x,0,layout.width),y:G.clamp(point.y,0,layout.height)};
          if(G.validQuad(quad)){this.state.quad=quad;this.changed();}
        } else if(drag.type==='brush') this.brush(event);
      });
      ['pointerup','pointercancel','lostpointercapture'].forEach(name=>this.wrap.addEventListener(name,event=>{
        if(event.pointerId===this.drag?.pointerId)this.endDrag();
      }));
      window.addEventListener('blur',()=>{this.space=false;this.endDrag();});
      window.addEventListener('keyup',event=>{if(event.code==='Space')this.space=false;});
      window.addEventListener('keydown',event=>this.keydown(event));
    }
    keydown(event){
      if(!this.isOpen())return;
      if(event.key==='Tab'){
        const controls=[...this.modal.querySelectorAll('button,input,select')].filter(el=>!el.disabled&&el.getClientRects().length);
        const first=controls[0],last=controls.at(-1);
        if(event.shiftKey && document.activeElement===first){event.preventDefault();last?.focus();}
        else if(!event.shiftKey && document.activeElement===last){event.preventDefault();first?.focus();}
        return;
      }
      if(event.key==='Escape'){event.preventDefault();this.close();return;}
      if(this.saving||this.detecting)return;
      if(event.target.closest('input,select,textarea,button,[contenteditable="true"]'))return;
      if(event.code==='Space'){event.preventDefault();this.space=true;}
      if(event.key==='Enter'){event.preventDefault();this.apply();}
      if(event.key==='ArrowLeft' && !event.shiftKey && this.index>0){event.preventDefault();this.load(this.index-1);}
      if(event.key==='ArrowRight' && !event.shiftKey && this.index+1<this.getFiles().length){event.preventDefault();this.load(this.index+1);}
    }
    draw(){
      const cw=this.wrap.clientWidth,ch=this.wrap.clientHeight,dpr=window.devicePixelRatio||1;
      if(!cw||!ch)return;
      const width=Math.round(cw*dpr),height=Math.round(ch*dpr);
      if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
      const ctx=this.ctx; ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,cw,ch);
      this.updateTools();
      if(!this.image||!this.state)return;
      const layout=this.layout(),view=this.view;
      ctx.save();ctx.translate(view.x,view.y);ctx.scale(view.scale,view.scale);
      ctx.fillStyle='#fff';ctx.fillRect(0,0,layout.width,layout.height);
      ctx.transform(...layout.matrix);ctx.drawImage(this.image,0,0);
      if(this.state.mask.width)ctx.drawImage(this.state.mask,0,0);
      ctx.restore();
      const points=this.state.mode==='rect'?G.corners(this.state.rect):this.state.mode==='persp'?this.state.quad:null;
      if(points){
        const screen=points.map(p=>G.toScreen(p,view));
        ctx.beginPath();ctx.rect(0,0,cw,ch);ctx.moveTo(screen[0].x,screen[0].y);
        screen.slice(1).forEach(p=>ctx.lineTo(p.x,p.y));ctx.closePath();ctx.fillStyle='rgba(0,0,0,.45)';ctx.fill('evenodd');
        ctx.beginPath();ctx.moveTo(screen[0].x,screen[0].y);screen.slice(1).forEach(p=>ctx.lineTo(p.x,p.y));ctx.closePath();
        ctx.strokeStyle='#22d3ee';ctx.lineWidth=2;ctx.stroke();
      }
      const rect=this.state.rect,position=G.toScreen(rect,view);
      Object.assign(this.rectBox.style,{left:position.x+'px',top:position.y+'px',width:rect.w*view.scale+'px',height:rect.h*view.scale+'px'});
      this.points.forEach((el,i)=>{
        const point=G.toScreen(this.state.quad[i],view);el.style.left=point.x+'px';el.style.top=point.y+'px';
      });
    }
    editSettings(){
      const state=this.state,settings={rotation:state.rotation};
      if(state.mode==='rect')settings.crop=[state.rect.x,state.rect.y,state.rect.w,state.rect.h];
      if(state.mode==='persp')settings.quad=state.quad.map(p=>[p.x,p.y]);
      return settings;
    }
    async editForm(state,settings){
      const data=new FormData();data.append('file',state.source,state.source.name);data.append('settings',JSON.stringify(settings));
      if(state.maskDirty){
        const mask=await new Promise(resolve=>state.mask.toBlob(resolve,'image/png'));
        if(!mask)throw new Error('Не удалось сохранить кисть');data.append('mask',mask,'mask.png');
      }
      return data;
    }
    cancelPreview(clear=true){
      this.previewToken++;clearTimeout(this.previewTimer);this.abortPreview?.abort();
      if(clear){
        this.previewImage.removeAttribute('src');this.previewImage.classList.add('hidden');
        this.previewLink.removeAttribute('href');this.previewLink.classList.add('hidden');
        if(this.previewUrl){URL.revokeObjectURL(this.previewUrl);this.previewUrl=null;}
        this.previewStatus.textContent='';
      }
    }
    refreshOptions(){
      if(!this.state)return;
      const options=this.getOptions();
      document.getElementById('editorTextStrength').value=Math.round(options.text_strength*100);
      document.getElementById('editorTextStrengthVal').textContent=Math.round(options.text_strength*100)+'%';
      document.getElementById('previewMode').value=options.mode;
      this.schedulePreview();
    }
    schedulePreview(delay=450){
      this.cancelPreview(false);
      if(!this.state||!this.image||!this.isOpen()||this.saving||this.detecting)return;
      this.previewImage.classList.add('stale');this.previewLink.classList.add('hidden');
      this.previewStatus.textContent='Обновляю предпросмотр…';
      const token=this.previewToken;
      this.previewTimer=setTimeout(()=>this.renderPreview(token),delay);
    }
    async renderPreview(token){
      if(token!==this.previewToken||!this.state||!this.isOpen()||this.saving||this.detecting)return;
      const state=this.state,settings=this.editSettings(),options=this.getOptions();
      const query=new URLSearchParams({mode:options.mode,long_side:options.long_side,text_strength:options.text_strength});
      const controller=new AbortController();this.abortPreview=controller;
      try{
        const data=await this.editForm(state,settings);
        if(token!==this.previewToken)return;
        const response=await fetch('/preview?'+query,{method:'POST',body:data,signal:controller.signal});
        if(!response.ok)throw new Error(await this.errorMessage(response));
        const blob=await response.blob();if(token!==this.previewToken)return;
        const url=URL.createObjectURL(blob),previous=this.previewUrl;
        this.previewUrl=url;this.previewImage.src=url;this.previewImage.classList.remove('hidden','stale');
        this.previewLink.href=url;this.previewLink.classList.remove('hidden');
        this.previewStatus.textContent='Результат с текущими границами и чёткостью текста';
        if(previous)URL.revokeObjectURL(previous);
      }catch(error){
        if(token===this.previewToken && error.name!=='AbortError'){
          this.previewStatus.textContent=error.message||'Не удалось обновить предпросмотр';
          this.previewLink.classList.add('hidden');this.previewImage.classList.add('stale');
        }
      }
    }
    cancelDetection(){this.detectToken++;this.abortDetection?.abort();this.detecting=false;}
    async autoDetect(){
      if(!this.state||this.saving||this.detecting)return;
      this.endDrag();this.cancelPreview();this.detecting=true;this.controls();this.setStatus('Ищу границы документа…');
      const state=this.state,token=++this.detectToken,options=this.getOptions();
      const controller=new AbortController();this.abortDetection=controller;
      const query=new URLSearchParams({rotation:state.rotation});
      ['align','trim','trim_margin','trim_thr','trim_sustain','trim_pad'].forEach(key=>{
        if(options[key]!==undefined)query.set(key,options[key]);
      });
      try{
        const data=new FormData();data.append('file',state.source,state.source.name);
        const response=await fetch('/detect?'+query,{method:'POST',body:data,signal:controller.signal});
        if(!response.ok)throw new Error(await this.errorMessage(response));
        const result=await response.json();if(token!==this.detectToken||state!==this.state)return;
        const points=result.quad.map(([x,y])=>({x,y}));
        if(!G.validQuad(points))throw new Error('Не удалось подобрать корректные границы. Выделите область вручную.');
        state.aspect='free';document.getElementById('aspectSel').value='free';
        if(result.kind==='rect'){
          const [x,y,w,h]=result.crop;state.mode='rect';state.rect={x,y,w,h};this.anchorRect();this.tool='rect';
        }else{state.mode='persp';state.quad=points;this.tool='persp';}
        this.changed();this.updateTools();
        this.setStatus(result.found||result.trimmed?'Автограницы найдены. Передвиньте их или нажмите «Больше полей».':'Надёжная автообрезка не найдена. Сохранён весь кадр — задайте область вручную.');
      }catch(error){
        if(token===this.detectToken && error.name!=='AbortError'){
          this.onError(error.message);this.setStatus(error.message||'Ошибка поиска границ');
        }
      }finally{
        if(token===this.detectToken){this.detecting=false;this.controls();this.schedulePreview(0);}
      }
    }
    expandSelection(){
      if(!this.state||this.saving||this.detecting)return;
      const state=this.state,layout=this.layout();
      if(state.mode==='rect'){
        const rect=state.rect,dx=layout.width*.025,dy=layout.height*.025;
        const x=Math.max(0,rect.x-dx),y=Math.max(0,rect.y-dy);
        state.rect=G.fitAspect({x,y,w:Math.min(layout.width,rect.x+rect.w+dx)-x,h:Math.min(layout.height,rect.y+rect.h+dy)-y},this.ratio());
        this.anchorRect();
      }else if(state.mode==='persp'){
        const center=state.quad.reduce((sum,p)=>({x:sum.x+p.x/4,y:sum.y+p.y/4}),{x:0,y:0});
        const points=state.quad.map(p=>({x:G.clamp(center.x+(p.x-center.x)*1.08,0,layout.width),
          y:G.clamp(center.y+(p.y-center.y)*1.08,0,layout.height)}));
        if(G.validQuad(points))state.quad=points;
      }else return;
      this.changed();
    }
    async apply(){
      if(!this.image||this.saving||this.detecting||!this.state?.dirty)return;
      this.cancelPreview();
      this.endDrag();this.saving=true;this.controls();this.setStatus('Сохраняю правки…');
      const operation=++this.saveToken;
      this.abortSave=new AbortController();
      const index=this.index,state=this.state,settings=this.editSettings();
      try {
        const data=await this.editForm(state,settings);
        if(operation!==this.saveToken)return;
        const response=await fetch('/edit',{method:'POST',body:data,signal:this.abortSave.signal});
        if(!response.ok)throw new Error(await this.errorMessage(response));
        const blob=await response.blob(),extension=blob.type==='image/jpeg'?'jpg':'png';
        if(operation!==this.saveToken)return;
        const name=state.source.name.replace(/\.[^.]+$/,'')+'.'+extension;
        const file=new File([blob],name,{type:blob.type});
        this.onSave(index,file);this.drafts.delete(index);this.saving=false;await this.load(index);
        this.setStatus('Правки применены. Можно продолжить редактирование или закрыть окно.');
      } catch(error){
        if(operation===this.saveToken && error.name!=='AbortError'){
          this.onError(error.message||'Не удалось применить правки');this.setStatus(error.message||'Ошибка сохранения');
        }
      } finally {if(operation===this.saveToken){this.saving=false;this.controls();}}
    }
  };
});
