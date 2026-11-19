/* Industrial Mission Control: framework-free UI, API-driven state, no mock data. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let state = null, busy = false, timer = null, selectedSector = 'construction_materials', mode = 'holdout';
  const sectorNames = {agro_processing:'Agro-processing',textiles:'Textile manufacturing',construction_materials:'Construction materials',pharmaceuticals:'Pharmaceuticals'};
  const stageNames = {unstudied:'UNEXPLORED',studied:'RESEARCHED',pilot:'PILOT ACTIVE',operating:'OPERATIONAL',exited:'DIVESTED'};
  const fmt = (x,d=2) => Number(x).toFixed(d);
  const signed = (x,d=4) => (Number(x)>=0?'+':'')+fmt(x,d);
  const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  async function api(route, body) {
    const opts = body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};
    const response = await fetch(route, opts);
    let data;
    try { data=await response.json(); } catch { throw new Error('The Python server sent an unreadable response.'); }
    if (!response.ok) throw new Error(data.error || 'Request failed');
    return data;
  }
  function toast(message) { const n=$('toast'); n.textContent=message; n.hidden=false; clearTimeout(toast.tid); toast.tid=setTimeout(()=>n.hidden=true,7000); }
  function setBusy(flag) { busy=flag; $('stepButton').disabled=flag || !!state?.meta.done; $('playButton').disabled=flag || !!state?.meta.done; $('resetButton').disabled=flag; }
  function showPolicy(prefix, policy) {
    const p=policy.portfolio;
    $(prefix+'Reward').textContent=signed(policy.total_reward);
    $(prefix+'Wealth').textContent=fmt(p.net_worth);
    $(prefix+'Cash').textContent=fmt(p.cash);
    $(prefix+'Jobs').textContent=fmt(p.employment_index,3);
    $(prefix+'Imports').textContent=fmt(p.import_substitution_index,3);
    $(prefix+'Risk').textContent=fmt(p.risk_exposure_index,3);
  }
  function renderSectors() {
    const sectors=state.policies.ppo.portfolio.sectors;
    $('sectorList').innerHTML=Object.entries(sectors).map(([key,value],i)=>{
      const active=selectedSector===key?' selected':'';
      const stage=String(value.stage||'unstudied').toLowerCase();
      return '<button type="button" class="sector-card'+active+'" data-sector="'+esc(key)+'" aria-pressed="'+!!active+'"><span class="sector-num">0'+(i+1)+'</span><span class="sector-main"><b>'+esc(sectorNames[key]||key)+'</b><small>Demand · '+fmt(value.demand_score,3)+' &nbsp;|&nbsp; '+fmt(value.invested_capital)+' capital</small></span><span class="sector-state">'+esc(stageNames[stage]||stage)+'</span></button>';
    }).join('');
    $('sectorList').querySelectorAll('[data-sector]').forEach(el=>el.addEventListener('click',()=>{
      selectedSector=el.dataset.sector; renderSectors(); window.IndustrialWorld?.focus?.(selectedSector);
    }));
    const selected=sectors[selectedSector];
    if(selected){$('sectorNote').innerHTML='<strong>'+esc(sectorNames[selectedSector])+':</strong> '+esc(stageNames[String(selected.stage).toLowerCase()]||selected.stage)+' · Project asset value '+fmt(selected.asset_value)+' · Feasibility estimate '+fmt(selected.feasibility_estimate,3)+' · Uncertainty '+fmt(selected.uncertainty,3)+'.';}
  }
  function renderEvents() {
    const events=state.events;
    for (const [which,id] of [['ppo','ppoEvents'],['rule_based','ruleEvents']]) {
      const filtered=events.filter(e=>e.policy===which);
      const root=$(id);
      if (!filtered.length){root.innerHTML='<div class="terminal-empty">$ awaiting agent actions<span class="cursor">_</span></div>';continue;}
      root.innerHTML=filtered.map(e=>'<div class="event-item"><span class="event-month">M'+String(e.month).padStart(2,'0')+'</span><span class="event-action">'+esc(e.description)+'</span><span class="event-reward'+(e.reward<0?' neg':'')+'">'+signed(e.reward,3)+'</span></div>').join('');
      root.scrollTop=root.scrollHeight;
    }
    $('eventCount').textContent=String(events.length).padStart(2,'0')+' EVENTS';
  }
  function renderChart() {
    const a=state.policies.ppo.history,b=state.policies.rule_based.history;
    const vals=[...a,...b].map(p=>p.cumulative_reward);
    const low=Math.min(-.03,...vals), high=Math.max(.03,...vals);
    const yr=high-low||1;
    const x=m=>35+(m/24)*640, y=v=>106-((v-low)/yr)*88;
    const path=hist=>hist.map((p,i)=>(i?'L':'M')+x(p.month).toFixed(1)+' '+y(p.cumulative_reward).toFixed(1)).join(' ');
    const lines=[0,.5,1].map(i=>{const v=low+yr*i;return '<line x1="35" x2="680" y1="'+y(v)+'" y2="'+y(v)+'" stroke="#486079" stroke-dasharray="3 5" opacity=".5"/><text x="30" y="'+(y(v)+4)+'" fill="#8196af" font-size="10" text-anchor="end">'+v.toFixed(2)+'</text>';}).join('');
    const xlabels=[0,6,12,18,24].map(m=>'<text x="'+x(m)+'" y="118" fill="#7e94ad" text-anchor="middle" font-size="9">'+m+'</text>').join('');
    $('performanceChart').innerHTML=lines+xlabels+'<path d="'+path(b)+'" fill="none" stroke="#e9bc76" stroke-width="2.5" stroke-linejoin="round"/><path d="'+path(a)+'" fill="none" stroke="#61e2c4" stroke-width="3" stroke-linejoin="round"/><circle cx="'+x(state.meta.month)+'" cy="'+y(a.at(-1).cumulative_reward)+'" r="4" fill="#61e2c4" stroke="#0b1828" stroke-width="2"/>';
  }
  function render(next) {
    state=next;
    const m=state.meta, p=state.policies.ppo, r=state.policies.rule_based;
    mode=m.split;
    $('monthLabel').textContent=String(m.month).padStart(2,'0');
    $('progressFill').style.width=(m.month/m.horizon*100)+'%';
    $('timelineStatus').textContent=m.done?'EPISODE COMPLETE':m.month>0?'SIMULATION IN PROGRESS':'READY TO BEGIN';
    $('sceneYear').textContent='INPUT YEAR · '+(m.year||'—');
    $('sceneSignal').textContent=(m.year||'—')+' / RWANDA IMPORT PROXIES';
    $('seedHint').textContent=(m.split==='holdout'?'2024 held-out signal':'2016–2023 historical signal')+' · synthetic transitions';
    $('holdoutMode').classList.toggle('active',mode==='holdout');$('historicalMode').classList.toggle('active',mode==='historical');
    $('seed').value=m.seed;
    showPolicy('ppo',p);showPolicy('rule',r);
    const difference=p.total_reward-r.total_reward;
    $('ppoDelta').textContent=signed(difference);
    $('deltaText').textContent=m.done?(difference>=0?'PPO ahead at completion.':'Heuristic ahead at completion.'):'Current cumulative PPO − Rule-Based';
    $('lastDecision').textContent=p.last_action?.description||'Awaiting first investment decision';
    $('lastDecisionReward').textContent=p.last_action?'Monthly reward '+signed(p.last_reward,5):'Checkpoint loaded · no decisions yet';
    renderSectors();renderEvents();renderChart();
    window.IndustrialWorld?.update?.(state);
    setBusy(false);
    if(m.done) stop();
  }
  async function step() {
    if(busy || state?.meta.done)return;
    setBusy(true);
    try{render(await api('/api/step',{}));}catch(err){stop();toast('Step failed: '+err.message);setBusy(false);}
  }
  function stop(){if(timer!==null){clearInterval(timer);timer=null;}$('playLabel').textContent='PLAY SIMULATION';$('playIcon').textContent='▶';}
  function play(){if(timer!==null){stop();return;}if(!state || state.meta.done)return;$('playLabel').textContent='PAUSE SIMULATION';$('playIcon').textContent='Ⅱ';timer=setInterval(step,Number($('speed').value));step();}
  async function reset() {
    stop();setBusy(true);
    const seed=Number($('seed').value);
    if(!Number.isSafeInteger(seed)||seed<0||seed>2_000_000_000){toast('Enter a valid non-negative integer seed (maximum 2 billion).');setBusy(false);return;}
    try{render(await api('/api/reset',{seed,split:mode}));}catch(err){toast('Reset failed: '+err.message);setBusy(false);}
  }
  $('stepButton').addEventListener('click',step);
  $('playButton').addEventListener('click',play);
  $('resetButton').addEventListener('click',reset);
  $('speed').addEventListener('change',()=>{if(timer!==null){stop();play();}});
  document.querySelectorAll('[data-mode]').forEach(btn=>btn.addEventListener('click',()=>{mode=btn.dataset.mode;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b===btn));$('seed').value=mode==='holdout'?60000:60007;reset();}));
  window.addEventListener('keydown',event=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement?.tagName))return;if(event.code==='Space'){event.preventDefault();play();}if(event.code==='ArrowRight')step();});
  window.addEventListener('industrial-world-ready',()=>{if(state)window.IndustrialWorld?.update?.(state);});
  api('/api/state').then(render).catch(err=>{$('terminalStatus').textContent='OFFLINE';toast('Cannot contact the Python simulation server: '+err.message);});
})();
