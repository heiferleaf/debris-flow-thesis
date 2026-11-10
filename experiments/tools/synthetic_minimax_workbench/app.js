const $ = s => document.querySelector(s);
let overview, selectedScene, selectedJob, drawingTool = 'polygon', picture = new Image(), draft, challenge;
let dirty = false, refreshing = false, newFlow = false;
const canvas = $('#locator'), context = canvas.getContext('2d');
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || '操作未完成');
  return value;
}
function error(e) {$('#error').textContent=e.message; $('#error').hidden=false;}
function clearError() {$('#error').hidden=true;}
function modified() {dirty=true; challenge=null; $('#draft-message').textContent='本次修改尚未保存';}
function collect() {
  return {...draft, scene_id:selectedScene, prompt:$('#prompt').value, duration:Number($('#duration').value), profile:$('#quality-profile').value, input_mode:$('#input-mode').value, reference_ok:$('#reference-ok').checked};
}
function referenceView(){
  const wet=$('#input-mode').value==='first_frame';
  const path=`/api/scenes/${selectedScene}/${wet?($('#quality-profile').value==='standard-water-v2'?'wet_user':'wet'):'dry'}`;
  $('#dry').src=path;picture.onload=draw;picture.src=path;
  $('#reference-title').textContent=wet?'本次实际上传：有水首帧':'干床场景参考图';
  $('#reference-note').textContent=wet?'片头即有连续水面；只上传这一张图':'与定位图一起作为软参考';
  $('#locator-note').textContent=wet?'下方箭头只用于本地核查，不上传。动画接口只接收上方有水首帧与提示词。':'定位图和干床图一起交给模型作软参考，当前接口没有区域遮罩参数。';
}
function cost() {const p=overview?.profiles?.[$('#quality-profile').value]||overview?.profile;if(!p)return;$('#estimate').textContent=`预计 ¥${p.estimated_cny.toFixed(2)} / 条`;$('#model-name').textContent=p.model;$('#output-setting').textContent=`${p.resolution} · 16:9`;$('#duration').options[0].textContent=`4 秒 · 预计 ¥${p.estimated_cny.toFixed(2)}`;}
function draw() {
  if (!picture.complete || !picture.naturalWidth) return;
  canvas.width=picture.naturalWidth; canvas.height=picture.naturalHeight;
  context.drawImage(picture,0,0);
  const point=p=>[p[0]*canvas.width,p[1]*canvas.height];
  const polygon=draft.polygon.map(point);
  context.strokeStyle='#00bce8'; context.fillStyle='#00bce820'; context.lineWidth=Math.max(3,canvas.width/250);
  if(polygon.length){context.beginPath();context.moveTo(...polygon[0]);polygon.slice(1).forEach(p=>context.lineTo(...p));if(polygon.length>2){context.closePath();context.fill();}context.stroke();}
  context.fillStyle='#00bce8';polygon.forEach(p=>{context.beginPath();context.arc(...p,5,0,Math.PI*2);context.fill();});
  const flow=draft.flow.map(point);context.strokeStyle='#ff791f';context.fillStyle='#ff791f';
  if(flow.length){context.beginPath();context.arc(...flow[0],6,0,Math.PI*2);context.fill();}
  if(flow.length>=2){const [a,b]=flow.slice(-2),angle=Math.atan2(b[1]-a[1],b[0]-a[0]),size=Math.max(18,canvas.width/45);context.beginPath();context.moveTo(...flow[0]);flow.slice(1).forEach(p=>context.lineTo(...p));context.stroke();context.beginPath();context.moveTo(...b);context.lineTo(b[0]-size*Math.cos(angle-.45),b[1]-size*Math.sin(angle-.45));context.lineTo(b[0]-size*Math.cos(angle+.45),b[1]-size*Math.sin(angle+.45));context.closePath();context.fill();}
}
canvas.addEventListener('pointerdown',event=>{
  if(!draft)return;const rect=canvas.getBoundingClientRect(),point=[(event.clientX-rect.left)/rect.width,(event.clientY-rect.top)/rect.height];
  if(drawingTool==='polygon'){if(draft.polygon.length<80)draft.polygon.push(point);}else{if(newFlow){draft.flow=[];newFlow=false;}if(draft.flow.length<16)draft.flow.push(point);}
  $('#reference-ok').checked=false;modified();draw();
});
function tool(name){drawingTool=name;newFlow=name==='flow';$('#polygon-tool').classList.toggle('selected',name==='polygon');$('#flow-tool').classList.toggle('selected',name==='flow');$('#drawing-help').textContent=name==='polygon'?'沿希望生成水流的河道边界依次点击至少 3 个点。定位图不会提前画出水流。':'沿河道从上游向下游依次点击，至少 2 点，最后一点为箭头。首次点击会重画流向。这表示生成意图，并非已核实的原视频流向。';}
$('#polygon-tool').onclick=()=>tool('polygon');$('#flow-tool').onclick=()=>tool('flow');
$('#undo').onclick=()=>{draft[drawingTool==='polygon'?'polygon':'flow'].pop();$('#reference-ok').checked=false;modified();draw();};
$('#clear').onclick=()=>{draft.polygon=[];draft.flow=[];$('#reference-ok').checked=false;modified();draw();};
async function save(){draft=await api('/api/draft',collect());dirty=false;$('#draft-message').textContent='草稿已保存到本机';return draft;}
$('#save-draft').onclick=()=>save().catch(error);
$('#restore-prompt').onclick=()=>{$('#prompt').value=$('#input-mode').value==='first_frame'?overview.wet_prompt:overview.default_prompt;modified();};
$('#input-mode').onchange=()=>{$('#prompt').value=$('#input-mode').value==='first_frame'?overview.wet_prompt:overview.default_prompt;$('#reference-ok').checked=false;referenceView();modified();};
$('#quality-profile').onchange=()=>{if($('#quality-profile').value==='standard-water-v2')$('#input-mode').value='first_frame';$('#reference-ok').checked=false;referenceView();cost();modified();renderJobs();};
['prompt','duration','reference-ok'].forEach(id=>$('#'+id).addEventListener('input',()=>{modified();cost();}));
$('#save-key').onclick=async()=>{try{const input=$('#official-key');const value=await api('/api/credentials',{key:input.value,replace:$('#replace-key').checked});input.value='';$('#key-message').textContent=value.message;await refresh();}catch(e){$('#official-key').value='';$('#key-message').textContent=e.message;}};
async function chooseScene(identity){
  if(selectedScene===identity)return;
  if(dirty)await save();
  const value=await api(`/api/scenes/${identity}/draft`);
  selectedScene=identity;draft=value;dirty=false;challenge=null;
  const row=overview.scenes.find(x=>x.scene_id===identity);
  $('#scene-name').textContent=row.camera;
  $('#source').src=`/api/scenes/${identity}/source`;
  $('#quality-profile').querySelector('[value="standard-water-v2"]').disabled=!row.wet_user_available;
  $('#quality-profile').value=draft.profile==='standard-water-v2'&&row.wet_user_available?'standard-water-v2':'lite';
  $('#input-mode').querySelector('[value="first_frame"]').disabled=!row.wet_available&&!row.wet_user_available;
  $('#input-mode').value=draft.input_mode==='first_frame'&&(row.wet_available||row.wet_user_available)?'first_frame':'image_url+image_url';
  referenceView();
  $('#prompt').value=draft.prompt;$('#duration').value=4;$('#reference-ok').checked=draft.reference_ok;
  $('#draft-message').textContent=draft.updated_at?'已读取本机草稿':'尚未保存本场景草稿';
  document.querySelectorAll('#scenes button').forEach(x=>x.classList.toggle('active',x.dataset.scene===identity));
  selectedJob=null;cost();renderJobs();
}
function reviewLabel(job){return job.review?.outcome==='rejected'?(job.review.reviewer==='assistant'?'初检未通过':'质量未通过'):job.review?.outcome==='accepted'?(job.review.scope==='motion_reference_only'?'运动参考初检通过':'质量已通过'):job.has_video?'等待播放验收':'尚无视频';}
function renderComparison(jobs){
  const batch=jobs.find(j=>j.batch_id)?.batch_id;
  const variants=jobs.filter(j=>j.batch_id===batch&&batch).sort((a,b)=>a.variant_code.localeCompare(b.variant_code));
  $('#prompt-comparison').hidden=!variants.length;
  $('#comparison-description').textContent=batch==='user-water-region-20261109-v2'?'H：使用用户重新圈定河道后生成的完整有水首帧，标准版MiniMax-H3、768P、4秒、预计2元。本条同时更换首帧与模型规格，不作为单变量比较。':batch==='water-prompts-20261109-v1'?'A/B/C使用同一场景与同一双图，只比较三种提示词。H3-Lite · 480P · 4秒。':batch==='wet-first-frame-20261109-v1'?'F：完整有水首帧直接动画。G：放大已有水面的近处河段，生成局部动画，再仅将水面放回原场景。已按原片核定流向：左下方进入，向中部和右侧流出。每条480P、4秒、约0.64元。':'D为通用流水近景对照，没有使用目标场景图。D通过运动参考初检后，才会生成E：目标干床场景＋D中的水流运动参考。';
  const container=$('#comparison-cards');
  const comparisonKey=[batch,...variants.map(j=>j.id)].join('|');
  if(container.dataset.batch!==comparisonKey){
    container.dataset.batch=comparisonKey;container.replaceChildren();
    variants.forEach(job=>{
      const card=document.createElement('article');card.className='comparison-card';card.dataset.job=job.id;
      const title=document.createElement('h4');title.textContent=`${job.variant_code} · ${job.variant_name}`;
      const status=document.createElement('p');status.className='muted comparison-status';
      const video=document.createElement('video');video.controls=true;video.muted=true;video.preload='metadata';video.playsInline=true;
      const details=document.createElement('details'),summary=document.createElement('summary'),prompt=document.createElement('p');
      summary.textContent='查看这个版本的提示词';prompt.textContent=job.prompt;details.append(summary,prompt);
      const select=document.createElement('button');select.type='button';select.textContent='选中此版本进行验收';select.onclick=()=>{selectedJob=job.id;$('#job-select').value=job.id;renderResult();$('#result-panel').scrollIntoView({behavior:'smooth',block:'start'});};
      card.append(title,status,video,details,select);container.append(card);
    });
  }
  variants.forEach(job=>{const card=Array.from(container.children).find(c=>c.dataset.job===job.id);if(!card)return;card.querySelector('.comparison-status').textContent=job.has_video?reviewLabel(job)+(job.motion_check?.flag==='low_motion'?' · 低运动警告':''):(job.message||'等待生成');const video=card.querySelector('video');video.hidden=!job.has_video;if(job.has_video&&!video.getAttribute('src'))video.src=`/api/jobs/${job.id}/video`;});
  $('#play-comparison').disabled=!variants.length||variants.some(j=>!j.has_video);
  $('#play-comparison').textContent=variants.length===1?'从头播放这一条':`从头同步播放 ${variants.length} 条`;
}
$('#play-comparison').onclick=async()=>{const videos=Array.from(document.querySelectorAll('#comparison-cards video'));videos.forEach(v=>{v.pause();v.currentTime=0;v.muted=true;});await Promise.allSettled(videos.map(v=>v.play()));};
function renderJobs(){
  renderLocalBatch().catch(error);
  const jobs=overview.jobs.filter(j=>j.scene_id===selectedScene), select=$('#job-select');
  if(!jobs.some(j=>j.id===selectedJob))selectedJob=jobs[0]?.id;
  select.replaceChildren();
  jobs.forEach((j,i)=>{const option=document.createElement('option');option.value=j.id;option.textContent=`${j.variant_code?j.variant_code+' '+j.variant_name:i===0?'最近一条':'历史试片'} · ${j.duration||10} 秒 · ${reviewLabel(j)}`;select.append(option);});
  select.value=selectedJob||'';select.disabled=!jobs.length;
  const active=overview.jobs.find(j=>['uploading','submission_unknown','submitted','running','poll_timeout','download_failed'].includes(j.status));
  const standard=$('#quality-profile').value==='standard-water-v2';
  const phaseUsed=overview.jobs.filter(j=>j.batch_id===(standard?'user-water-region-20261109-v2':'wet-first-frame-20261109-v1')).length;
  const phaseExhausted=$('#input-mode').value==='first_frame'&&phaseUsed>=(standard?1:2);
  $('#prepare').disabled=!!active||phaseExhausted;
  if(phaseExhausted)$('#submission-note').textContent=standard?'本次标准版已分配一条作业，不会重复提交。':'低价阶段两条试片已用完，共 ¥1.28。';
  else if(!active)$('#submission-note').textContent='未提交新任务，请先核对本条输入与费用。';
  $('#active-job').hidden=!active;
  const oldResume=$('#resume');if(oldResume)oldResume.remove();
  if(active){$('#job-status').textContent=active.message||active.status;$('#job-detail').textContent=`该任务预计 ¥${Number(active.estimated_cny||0).toFixed(2)}；只跟踪此任务，不会自动重新生成。`;if(active.can_resume){const button=document.createElement('button');button.id='resume';button.textContent='查询并获取已有任务';$('#active-job').append(button);button.onclick=async()=>{try{await api('/api/poll',{job_id:active.id});button.textContent='正在查询已有任务';}catch(e){error(e);}};}}
  const failed=jobs.find(j=>j.review?.outcome==='rejected'),accepted=jobs.find(j=>j.review?.outcome==='accepted'&&j.input_mode!=='text_only'&&j.review.scope!=='motion_reference_only');
  $('#workflow-status').textContent=accepted?'已有一条试片通过媒体质量验收。仍需核对训练用途，尚不能由一条推广到所有场景。':failed?'工作流尚未验证通过。最近未通过的原因：'+(failed.review.note||'水流内容不符合要求。'):'工作流尚待验证：先准备河道定位，生成一条试片并播放验收，再决定是否推广。';
  $('#stage').textContent=active?'任务进行中':accepted?'已有质量通过试片':'准备试片';
  renderComparison(jobs);renderResult();
}
let localPanelReady=false,showCloudHistory=false;
async function renderLocalBatch(){
  const identity=selectedScene;
  const local=await api('/api/local/state');
  if(identity!==selectedScene)return;
  const all=local.jobs.filter(j=>j.batch?.startsWith('controlled-flow'));
  const job=all.find(j=>j.scene_id===identity);
  if(!job)return;
  if(!localPanelReady){
    const panel=document.createElement('section');panel.id='local-batch';
    panel.innerHTML='<div class="section-heading"><h2>本轮首帧与12秒视频</h2><span id="local-batch-total" class="pill"></span></div><p id="local-batch-status" role="status"></p><div class="media-grid"><figure><figcaption>实际使用的有水首帧</figcaption><img id="local-batch-image" alt="本轮河道约束有水首帧"></figure><figure><figcaption>本地生成视频</figcaption><video id="local-batch-video" controls playsinline preload="metadata" hidden></video><p id="local-batch-wait"></p></figure></div><progress id="local-batch-progress" max="100"></progress><div class="actions"><a id="local-batch-download" download="water-12s.mp4" hidden>下载12秒视频</a><a id="local-batch-record">查看配置</a><button id="cloud-history-toggle" type="button">查看早期云端试片</button></div><p class="muted">当前视频由有水首帧与已保存的河道、流向在本地生成。新增场景等待你验收，不自动进入训练。</p>';
    $('#scene').before(panel);
    const promptDetails=document.createElement('details');
    promptDetails.innerHTML='<summary>查看本次首帧制作提示词</summary><pre id="local-frame-prompt"></pre>';
    panel.append(promptDetails);
    const clarity=document.createElement('section');clarity.id='local-clarity';clarity.hidden=true;
    clarity.innerHTML='<h3 id="local-clarity-title">水纹清晰度改进对照</h3><p id="local-clarity-description"></p><video id="local-clarity-video" controls playsinline preload="metadata" style="width:100%;max-height:600px"></video><p><a id="local-clarity-download" download="water-clarity-12s.mp4">下载改进版12秒视频</a> · <a id="local-clarity-previous" target="_blank" hidden>播放上一次清晰度改进版</a> · 改进版等待验收，不自动进入训练。</p>';
    panel.append(clarity);
    const originalHistory=document.createElement('details');originalHistory.id='local-original-history';originalHistory.hidden=true;
    originalHistory.innerHTML='<summary>查看最初首帧与初版视频（历史版本）</summary>';
    panel.append(originalHistory);
    const beforeAfter=document.createElement('details');beforeAfter.id='local-before-after';beforeAfter.hidden=true;beforeAfter.open=true;
    beforeAfter.innerHTML='<summary>同步查看修正前后（上：修正前，下：修正后）</summary><p id="local-before-after-description"></p><video id="local-before-after-video" controls playsinline preload="metadata" style="width:100%"></video><p><a id="local-before-after-download" download="shoreline-before-after-12s.mp4">下载同步对照</a></p>';
    clarity.append(beforeAfter);
    const boundaryDetails=document.createElement('details');boundaryDetails.id='local-boundary-details';boundaryDetails.hidden=true;
    boundaryDetails.innerHTML='<summary>查看岸线细化对照：橙色为粗范围，绿色为细化边界</summary><p>细化根据水面原图、岸石保护信息生成，原始标记不变。绿色是待验收的图像边界，不代表自动分割必然准确。</p><img id="local-boundary-image" alt="原始粗范围与细化岸线对照" style="width:100%">';
    clarity.append(boundaryDetails);
    const saveActions=$('#save-draft').parentElement;$('#scene').append(saveActions);
    $('#cloud-history-toggle').onclick=()=>{showCloudHistory=!showCloudHistory;['configuration','progress','acceptance'].forEach(id=>$('#'+id).hidden=!showCloudHistory);$('#cloud-history-toggle').textContent=showCloudHistory?'收起早期云端试片':'查看早期云端试片';};
    $('.brand span').textContent='场景首帧 · 本地12秒水流';
    $('.steps').hidden=true;
    localPanelReady=true;
  }
  ['configuration','progress','acceptance'].forEach(id=>$('#'+id).hidden=!showCloudHistory);
  const currentFor=base=>[base.physics_comparison_id,base.boundary_comparison_id,base.bank_comparison_id,base.clarity_comparison_id].map(id=>local.jobs.find(j=>id&&j.id===id&&j.status==='completed')).find(Boolean)||base;
  const accepted=all.filter(j=>currentFor(j).user_accepted===true).length;
  $('#local-batch-total').textContent=`已验收 ${accepted} / ${all.length} · 已有试片 ${all.filter(j=>j.status==='completed').length} / ${all.length}`;
  const labels={queued:'首帧已准备，等待渲染',paused:'已暂停，等待清晰度改进',rendering:'正在本地渲染',completed:job.user_accepted?'试片已获你认可':'视频已生成，待你验收',failed:'渲染失败'};
  $('#local-batch-status').textContent=job.quality_issue||(labels[job.status]||job.status)+(job.status==='rendering'?` · ${job.progress||0}%`:'');
  $('#workflow-status').textContent=job.user_accepted?'本场景试片已获你认可；其余场景使用各自新标记的范围与流向制作。':'本轮已收到你的范围和流向。首帧制作 → 本地生成12秒视频 → 播放验收。';
  $('#stage').textContent=job.quality_issue?'修订中':labels[job.status]||job.status;
  if($('#local-batch-image').dataset.job!==job.id){$('#local-batch-image').src=job.first_frame_url;$('#local-batch-image').dataset.job=job.id;}
  const promptView=$('#local-frame-prompt');
  if(promptView.dataset.upload!==job.upload_id){
    promptView.dataset.upload=job.upload_id;promptView.textContent='读取中…';
    api(`/api/local/uploads/${job.upload_id}/draft`).then(v=>{if(promptView.dataset.upload===job.upload_id)promptView.textContent=v.source.first_frame_prompt||'本场景沿用此前已认可的有水首帧。';}).catch(error);
  }
  const video=$('#local-batch-video'),done=job.status==='completed';
  const version=job.id+'|'+(job.active_output||'original');
  if(video.dataset.job!==version){video.pause();video.removeAttribute('src');delete video.dataset.job;}
  if(done&&video.dataset.job!==version){video.src=`/api/local/jobs/${job.id}/candidate.mp4?v=${encodeURIComponent(job.active_output||'original')}`;video.dataset.job=version;}
  video.hidden=!done;$('#local-batch-wait').hidden=done;$('#local-batch-wait').textContent=job.message;
  $('#local-batch-progress').hidden=job.status!=='rendering';$('#local-batch-progress').value=job.progress||0;
  $('#local-batch-download').hidden=!done;$('#local-batch-download').href=`/api/local/jobs/${job.id}/candidate.mp4`;
  $('#local-batch-record').href=`/api/local/jobs/${job.id}/record.json`;
  const bankJob=local.jobs.find(j=>j.id===job.bank_comparison_id&&j.status==='completed');
  const boundaryJob=local.jobs.find(j=>j.id===job.boundary_comparison_id&&j.status==='completed');
  const physicsJob=local.jobs.find(j=>j.id===job.physics_comparison_id&&j.status==='completed');
  const clarityJob=physicsJob||boundaryJob||bankJob||local.jobs.find(j=>j.id===job.clarity_comparison_id&&j.status==='completed');
  $('#local-clarity-title').textContent=physicsJob?'物理流场驱动试片 · 待验收':boundaryJob?'岸线细化与运动修正试片':bankJob?'岸边运动修正试片':'水纹清晰度改进对照';
  $('#local-clarity-description').textContent=physicsJob?(physicsJob.preview_description||'12秒，1280×720。浅水方程计算流场，近岸与中部共同运动；水深、河床和相机尺度为假设，纹理采用短周期更新，属于外观近似。请观察岸边是否自然、纹理是否清楚，以及是否出现规律性闪动。尚未用于训练。'):boundaryJob?'本次把粗多边形与实际水面边界分开：只调整岸线附近的合成与局部运动，保留内部水纹和远处背景。请重点看岸线凹凸、露石和原来直线截断的位置。':bankJob?'首帧不变，中部复用清晰度改进版；近岸形态保持附着，只作小幅波动。请观察是否还有沿岸平移、消失的感觉。这是局部视觉处理，不是水动力模拟。':'上方为旧版，下方提高了河道内部渲染精度，首帧、河道范围和流向保持不变。请同时观察水纹清晰度、石块和岸边的稳定性。';
  $('#local-clarity-previous').hidden=!(physicsJob||boundaryJob||bankJob);
  if(physicsJob||boundaryJob||bankJob){$('#local-clarity-previous').href=`/api/local/jobs/${physicsJob?(physicsJob.previous_comparison_id||job.boundary_comparison_id):boundaryJob?job.bank_comparison_id:job.clarity_comparison_id}/candidate.mp4`;$('#local-clarity-previous').textContent=physicsJob?(physicsJob.previous_label||'播放上一版（岸边近乎静止）'):boundaryJob?'播放上一版岸边试片':'播放上一次清晰度改进版';}
  $('#local-boundary-details').hidden=!boundaryJob;
  if(boundaryJob&&$('#local-boundary-image').dataset.job!==boundaryJob.id){$('#local-boundary-image').src=`/api/local/jobs/${boundaryJob.id}/motion-controls.jpg`;$('#local-boundary-image').dataset.job=boundaryJob.id;}
  const clarityVideo=$('#local-clarity-video');
  const pairedJob=local.jobs.find(j=>j.id===(physicsJob?job.physics_before_after_id:job.boundary_before_after_id)&&j.status==='completed');
  $('#local-before-after-description').textContent=physicsJob?(pairedJob?.preview_description||'同一时刻、同一水面裁剪范围，未缩放。上方是此前岸边近乎静止的版本，下方为当前物理流场驱动试片。'):'同一时刻、同一水面裁剪范围，未缩放。对比的是岸线细化前后的变化。';
  $('#local-before-after').hidden=!pairedJob;
  const pairedVideo=$('#local-before-after-video');
  if(pairedVideo.dataset.job!==(pairedJob?.id||'')){
    pairedVideo.pause();pairedVideo.removeAttribute('src');delete pairedVideo.dataset.job;
    if(pairedJob){const url=`/api/local/jobs/${pairedJob.id}/candidate.mp4`;pairedVideo.src=url;pairedVideo.dataset.job=pairedJob.id;$('#local-before-after-download').href=url;}
  }
  $('#local-clarity').hidden=!clarityJob;
  const mediaGrid=$('#local-batch').querySelector('.media-grid'),history=$('#local-original-history');
  if(clarityJob){
    if(mediaGrid.parentElement!==history){history.append(mediaGrid);$('#local-batch-status').after($('#local-clarity'));}
    history.hidden=false;
    mediaGrid.querySelectorAll('figcaption')[0].textContent='初版有水首帧';
    mediaGrid.querySelectorAll('figcaption')[1].textContent='初版视频（不是当前候选）';
    $('#local-batch-download').hidden=false;$('#local-batch-download').href=`/api/local/jobs/${clarityJob.id}/candidate.mp4`;
    $('#local-batch-download').textContent='下载当前试片';$('#local-batch-record').href=`/api/local/jobs/${clarityJob.id}/record.json`;
    $('#stage').textContent=clarityJob.user_accepted?'已验收':clarityJob.review?.outcome==='rejected'?'待修订':'待你验收';
  }else{
    if(mediaGrid.parentElement===history)$('#local-batch-status').after(mediaGrid);
    history.hidden=true;
    mediaGrid.querySelectorAll('figcaption')[0].textContent='实际使用的有水首帧';
    mediaGrid.querySelectorAll('figcaption')[1].textContent='本地生成视频';
    $('#local-batch-download').textContent='下载12秒视频';
  }
  if(clarityVideo.dataset.job!==(clarityJob?.id||'')){
    clarityVideo.pause();clarityVideo.removeAttribute('src');delete clarityVideo.dataset.job;
    if(clarityJob){const url=`/api/local/jobs/${clarityJob.id}/candidate.mp4`;clarityVideo.src=url;clarityVideo.dataset.job=clarityJob.id;$('#local-clarity-download').href=url;}
  }
}
function renderResult(){
  const job=overview.jobs.find(j=>j.id===selectedJob),hasVideo=!!job?.has_video;
  $('#no-result').hidden=hasVideo;$('#result-panel').hidden=!hasVideo;
  if(!hasVideo){$('#no-result').textContent=job?(job.message||'这条任务尚未得到视频。'):'本场景还没有试片。保存草稿不会生成视频。';$('#result').pause();$('#result').removeAttribute('src');delete $('#result').dataset.job;return;}
  $('#result-title').textContent=`${job.variant_code?job.variant_code+' · '+job.variant_name+' · ':''}${job.model||'MiniMax-H3'} · ${job.duration||10} 秒`;
  const video=$('#result');const url=`/api/jobs/${job.id}/video`;if(video.dataset.job!==job.id){video.dataset.job=job.id;video.src=url;}
  $('#quality-status').textContent=reviewLabel(job);$('#quality-status').className='pill '+(job.review?.outcome==='rejected'?'bad':job.review?.outcome==='accepted'?'good':'');
  const inputName=job.reference_kind==='wet_crop'?'局部有水首帧动画后回贴，主视频为合成结果':({'first_frame':'有水首帧约束，未上传定位图或运动视频','image_url+image_url':'场景与定位双图参考','text_only':'通用流水能力对照，没有使用目标场景图','image+motion_video':'目标干床场景＋独立水流视频运动参考'}[job.input_mode]||'历史首帧生成');
  $('#result-info').textContent=`${inputName} · 该条估算 ¥${Number(job.estimated_cny||0).toFixed(2)} · 平台状态：${job.provider_status||job.status} · 内容验收单独记录`;
  $('#previous-review').hidden=!job.review?.note;$('#previous-review').textContent=job.review?.note||'';
  $('#motion-check').hidden=!job.motion_check;$('#motion-check').textContent=job.motion_check?.message||'';
  $('#actual-input-panel').hidden=job.input_mode!=='first_frame';
  if(job.input_mode==='first_frame'&&$('#actual-input-image').dataset.job!==job.id){$('#actual-input-image').src=`/api/jobs/${job.id}/input.png`;$('#actual-input-image').dataset.job=job.id;}
  $('#actual-input-note').textContent=job.reference_kind==='wet_crop'?'这一张为原场景近处河道的几何放大图，保留了已有水面；模型只接收这张首帧和提示词，没有接收河道标线。':'模型接收这一张已有连续水面的首帧和提示词，没有接收河道标线。';
  $('#raw-video-panel').hidden=job.presentation_video!=='candidate-composited.mp4';
  if(job.presentation_video==='candidate-composited.mp4'&&$('#raw-video').dataset.job!==job.id){$('#raw-video').src=`/api/jobs/${job.id}/raw-video`;$('#raw-video').dataset.job=job.id;}
  $('#inspection16-panel').hidden=!job.has_inspection16;
  if(job.has_inspection16&&$('#inspection16-image').dataset.job!==job.id){$('#inspection16-image').src=`/api/jobs/${job.id}/inspection16.jpg`;$('#inspection16-image').dataset.job=job.id;}
  if($('#result-panel').dataset.job!==job.id){$('#result-panel').dataset.job=job.id;['motion','shape','background'].forEach(k=>$('#check-'+k).checked=job.review?.checks?.[k]===true);$('#review-note').value=job.review?.note||'';}
  $('#job-record').textContent=JSON.stringify(job,null,2);
}
$('#job-select').onchange=()=>{selectedJob=$('#job-select').value;renderResult();};
async function review(outcome){try{clearError();const checks=Object.fromEntries(['motion','shape','background'].map(k=>[k,$('#check-'+k).checked]));await api('/api/review',{job_id:selectedJob,outcome,checks,note:$('#review-note').value});await refresh();}catch(e){error(e);}}
$('#reject').onclick=()=>review('rejected');$('#accept').onclick=()=>review('accepted');
$('#prepare').onclick=async()=>{try{clearError();await save();challenge=await api('/api/prepare',collect());const v=challenge.intent;$('#confirm-dry').src=`/api/prepared/${challenge.fingerprint}/reference.png`;$('#confirm-locator').src=challenge.locator_url;$('#confirm-locator').closest('figure').hidden=v.input_mode==='first_frame';$('#confirm-reference-title').textContent=v.input_mode==='first_frame'?'实际上传：这一张有水首帧':'图一 · 场景参考';$('#confirm-prompt').textContent=v.prompt;$('#confirm-cost').textContent=`${v.model} · ${v.resolution} · ${v.duration} 秒 · ${v.input_mode==='first_frame'?'仅一张有水首帧':'两张参考图'} · 预计 ¥${v.estimated_cny.toFixed(2)}。实际费用以 MiniMax 官方账单为准。`;$('#consent-label').textContent=`同意上传本次已展示的素材与提示词，并支付本条预计 ${v.estimated_cny.toFixed(2)} 元费用`;$('#submit').textContent=`确认上传并生成 · ¥${v.estimated_cny.toFixed(2)}`;$('#consent').checked=false;$('#submit').disabled=true;$('#confirm-error').hidden=true;$('#confirmation').showModal();}catch(e){$('#submission-note').textContent=e.message;error(e);}};
$('#consent').onchange=()=>{$('#submit').disabled=!$('#consent').checked;};
function cancel(){challenge=null;$('#confirmation').close();}
$('#close-confirm').onclick=cancel;$('#cancel-confirm').onclick=cancel;
$('#confirm-form').onsubmit=async event=>{event.preventDefault();if(!challenge||!$('#consent').checked)return;$('#submit').disabled=true;try{const value=await api('/api/submit',{token:challenge.token,approved:true});challenge=null;selectedJob=value.id;$('#confirmation').close();$('#submission-note').textContent='本条已提交；正在跟踪任务';await refresh();document.querySelector('#progress').scrollIntoView({behavior:'smooth',block:'start'});}catch(e){$('#confirm-error').textContent=e.message;$('#confirm-error').hidden=false;challenge=null;}};
async function refresh(){if(refreshing)return;refreshing=true;try{overview=await api('/api/workbench');$('#connection').textContent='本机已连接';$('#credential-status').textContent=overview.credential_configured?'已配置官方密钥':'尚未配置';if(!selectedScene){$('#scenes').replaceChildren();overview.scenes.forEach(s=>{const b=document.createElement('button');b.type='button';b.dataset.scene=s.scene_id;const n=document.createElement('span');n.textContent=String(s.index).padStart(2,'0');b.append(n,document.createTextNode(s.camera));b.onclick=()=>chooseScene(s.scene_id).catch(error);$('#scenes').append(b);});if(overview.scenes.length)await chooseScene(overview.scenes[0].scene_id);}else renderJobs();}catch(e){$('#connection').textContent='连接中断，请检查本地服务';error(e);}finally{refreshing=false;}}
refresh();setInterval(refresh,5000);
