const $=s=>document.querySelector(s);let items=[],tab='baseline',selected;
const sections={baseline:['01 / OBSERVE THE EXECUTION','Actions leave traces.','Inspect recorded behavior, follow the skill path, and evaluate the next revision.'],piano:['02 / STUDY A SKILL','From contact to music.','Inspect piano performances and the recorded evidence behind skill selection.'],dexjoco:['03 / DEVELOP THE NEXT REVISION','Practice becomes evidence.','Inspect generated layouts, demonstrations and configured training rounds.']};
function log(text,kind='response'){const p=document.createElement('p');p.className=kind;p.textContent=text;$('#transcript').append(p);$('#transcript').scrollTop=$('#transcript').scrollHeight;}
function select(item){selected=item;$('#empty-media').hidden=true;$('#mode').textContent=item.mode||'RECORDED ROLLOUT';$('#evolution').hidden=!item.evolution;$('#evaluate').hidden=item.group!=='baseline';$('#player').src=item.media;$('#video-title').textContent=item.title;$('#note').textContent=item.note;$('#metric').textContent=item.metrics?.f1?'Mechanical key F1 · '+(item.metrics.f1*100).toFixed(1)+'%':'';$('#graph').replaceChildren();item.stages.forEach((s,i)=>{if(i){const a=document.createElement('span');a.className='arrow';a.textContent='→';$('#graph').append(a)}const node=document.createElement('div');node.className='node';node.textContent=s;$('#graph').append(node)});renderCards();log('Opened '+item.title+'\n'+item.mode+' · '+item.group);}
function renderCards(){const rows=items.filter(x=>x.group===tab);$('#count').textContent=rows.length+' recordings';$('#cards').replaceChildren();rows.forEach(x=>{const b=document.createElement('button');b.className='card'+(selected?.id===x.id?' selected':'');const icon=document.createElement('span');icon.className='play-icon';icon.textContent='▶';const label=document.createElement('div');const title=document.createElement('strong');title.textContent=x.title;const small=document.createElement('small');small.textContent=x.mode;label.append(title,small);b.append(icon,label);b.onclick=()=>select(x);$('#cards').append(b)});}
function switchTab(next){tab=next;$('#cycle-panel').hidden=tab!=='dexjoco'||!cycle;document.querySelectorAll('nav button').forEach(b=>{b.classList.toggle('active',b.dataset.tab===tab);b.setAttribute('aria-pressed',String(b.dataset.tab===tab))});$('#section').textContent=tab.toUpperCase();const [eye,title,intro]=sections[tab];$('#eyebrow').textContent=eye;$('#title').textContent=title;$('#intro').textContent=intro;const first=items.find(x=>x.group===tab);if(first)select(first);else{$('#empty-media').hidden=false;$('#metric').textContent='';$('#mode').textContent='AWAITING RECORDING';$('#player').removeAttribute('src');$('#player').load();$('#video-title').textContent='No recording selected';$('#note').textContent='Initialize the pinned RoboDojo baseline and configure its runtime to run evaluations.';$('#graph').replaceChildren();$('#evolution').hidden=true;$('#evaluate').hidden=true;selected=null;renderCards();log('No recordings for '+tab+'. The pinned baseline is available through /robodojo in the terminal.')}}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{log(b.dataset.tab==='baseline'?'/baselines':'/demo '+b.dataset.tab,'typed');switchTab(b.dataset.tab)});
$('#command-form').onsubmit=e=>{
 e.preventDefault();const input=$('#command');const cmd=input.value.trim();if(!cmd)return;
 log('❯ '+cmd,'typed');input.value='';
 if(/^\/layouts(?: \d+)?$/.test(cmd))startJob('/api/layouts',{count:Number(cmd.split(' ')[1]||2)});
 else if(/^\/collect(?: \d+)?$/.test(cmd))startJob('/api/collect',{episodes:Number(cmd.split(' ')[1]||2)});
 else if(/^\/cycle(?: \d+)?$/.test(cmd))startJob('/api/cycle',{rounds:Number(cmd.split(' ')[1]??2)});
 else if(/^\/train(?: \d+)?$/.test(cmd))startJob('/api/train',{steps:Number(cmd.split(' ')[1]||100)});
 else if(cmd==='/demo piano')switchTab('piano');
 else if(cmd==='/baselines')switchTab('baseline');
 else if(cmd==='/demo dexjoco'){switchTab('dexjoco');log('Generate layouts with /layouts 2, synthesize data with /collect 2, then /train 100. Use /cycle 2 for the configured continual pi05 loop.');}
 else if(cmd==='/status')showStatus();
 else if(cmd==='/play')$('#player').play();
 else if(cmd.startsWith('/'))log('/baselines · /demo piano · /demo dexjoco · /layouts 2 · /collect 2 · /train 100 · /cycle 2 · /status · /play');
 else startJob('/api/chat',{message:cmd});
};
$('#expand').onclick=()=>$('#player').requestFullscreen();fetch('/api/catalog').then(r=>r.json()).then(d=>{items=d.items;switchTab(location.hash.slice(1) in sections?location.hash.slice(1):'baseline')}).catch(e=>log(e.message));

async function startJob(endpoint,payload){
 const button=$('#evaluate');button.disabled=true;
 try{
  const response=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  let job=await response.json();if(!response.ok)throw Error(job.error);
  if(endpoint==='/api/chat'){if(job.answer)log(job.answer);if(!job.job)return;job=job.job;}
  log('Run started.');await watchJob(job);
 }catch(error){log(error.message)}finally{button.disabled=false}
};

$('#evolution').onclick=()=>{
 const e=selected.evolution;if(!e)return;
 log(e.scope);log('Practice round '+e.round);
 $('#graph').replaceChildren();
 for(const c of e.candidates){
  const node=document.createElement('div');node.className='node';
  node.textContent=c.name+' · '+(c.contact_f1*100).toFixed(1)+'% contact F1'+(c.promoted?' · retained':'');
  $('#graph').append(node);
 }
 log('Selected '+e.selected+'\nMemory '+e.memory_revision.slice(0,12));
};

$('#evaluate').onclick=()=>startJob('/api/evaluate',{id:selected.id});

async function watchJob(job){
let previous=[];
  while(true){
   if(JSON.stringify(previous)!==JSON.stringify(job.lines)){
    let overlap=Math.min(previous.length,job.lines.length);
    while(overlap && previous.slice(-overlap).join('\n')!==job.lines.slice(0,overlap).join('\n'))overlap--;
    job.lines.slice(overlap).forEach(line=>log(line));previous=job.lines;
   }
   if(job.status!=='running'){log('Run '+job.status+' · exit '+job.returncode);const library=await fetch('/api/catalog').then(r=>r.json());items=library.items;renderCards();break;}
   await new Promise(resolve=>setTimeout(resolve,1000));
   const result=await fetch('/api/jobs/'+job.id);if(!result.ok)throw Error('Log connection interrupted');job=await result.json();
  }
}
async function showStatus(){
 try{const job=await fetch('/api/status').then(r=>r.json());
 if(job)await watchJob(job);else log('No runs yet.');
 }catch(error){log(error.message)}
}

let cycle=null,cycleLines=[],followCycle=false;
async function refreshCycle(){
 try{
  const response=await fetch('/api/cycle');if(!response.ok)return;
  cycle=await response.json();$('#cycle-panel').hidden=tab!=='dexjoco'||!cycle;
  if(!cycle)return;
  $('#cycle-round').textContent='Round '+cycle.round;
  const training=cycle.state==='training'&&cycle.metrics.step?' · step '+cycle.metrics.step+'/'+cycle.steps+' · loss '+cycle.metrics.loss.toFixed(3):'';
  $('#cycle-progress').textContent=cycle.successful+'/'+cycle.layouts+' demonstrations · '+cycle.devices+' GPUs · '+cycle.state.replaceAll('_',' ')+training;
  if(cycle.validation){
   const v=cycle.validation;
   $('#cycle-progress').textContent+=' · validation '+v.completed+'/'+v.layouts+' · '+v.successful+' successful';
  }
  if(followCycle){
   let overlap=Math.min(cycleLines.length,cycle.lines.length);
   while(overlap&&cycleLines.slice(-overlap).join('\n')!==cycle.lines.slice(0,overlap).join('\n'))overlap--;
   cycle.lines.slice(overlap).forEach(line=>log(line));cycleLines=cycle.lines;
  }
 }catch(error){if(followCycle)log('Cycle status unavailable: '+error.message)}
}
$('#cycle-logs').onclick=()=>{followCycle=false;showStatus()};
setInterval(refreshCycle,5000);refreshCycle();
