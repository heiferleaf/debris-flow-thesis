"use strict";
const select = selector => document.querySelector(selector);
let current, challenge, requestBusy = false, promptDirty = false, resultUrl = null;
let autoDownloadAttempted = false;
const statusText = {draft:"尚未提交 · 等待本条费用确认",submitted:"已提交 · 等待生成",running:"正在生成",succeeded:"生成完成",downloaded:"视频已保存，可播放",failed:"生成失败",canceled:"任务已取消",submission_unknown:"提交回执待核实，请勿重复生成",provider_unknown:"服务端状态待核实"};
function errorMessage(error) { select("#generation-error").textContent = error.message; select("#generation-error").hidden = false; }
async function api(path, method="GET", body) {
  const response = await fetch(path,{method,cache:"no-store",headers:body===undefined?{}:{"Content-Type":"application/json"},body:body===undefined?undefined:JSON.stringify(body)});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error?.message || "本地操作未完成");
  return value;
}
const jobPath = suffix => `/api/jobs/${current.jobs[0].id}/${suffix}`;
function render(snapshot) {
  current = snapshot;
  const job = snapshot.jobs[0];
  const currentJob = document.querySelector("main").dataset.currentJobId;
  if (currentJob && job.id !== currentJob) {
    select("#generation-title").textContent="历史作业回执 · 当前入口已更新";
    select("#generation-lead").innerHTML='<a href="http://127.0.0.1:8773/generate">进入普通水流新试片生成页 →</a>';
    select("#generation-status").textContent="旧作业保留，当前试片请使用 8773";
    select("#generation-status-detail").textContent="这个地址的服务尚未加载新参考图与新作业。";
    select("#generation-reference").removeAttribute("src");
    select("#generation-start").disabled=true;
    select("#generation-prompt-zh").disabled=true;
    select("#generation-prompt-en").disabled=true;
    select("#generation-save").disabled=true;
    select("#generation-id").textContent=`历史作业：${job.id}`;
    return;
  }
  const heldForTaskMatch = document.querySelector("main").dataset.heldJobId === job.id || job.reference_acceptance?.status === "paused_for_task_match_review";
  const rejected = job.reference_acceptance?.status === "rejected_by_user", blocked = rejected || heldForTaskMatch, editable = job.status === "draft" && !blocked;
  const holdReason = "助手暂停单一参考方案，待任务匹配确认。黄色本身不等于不合格；含沙细流保留为备选，不计为用户拒绝，不改变原视频的 N 标签。";
  select("#generation-status").textContent = heldForTaskMatch ? "助手暂停单一参考方案，待任务匹配确认" : rejected ? "当前参考图验收不通过，暂停本条付费生成" : statusText[job.status] || job.provider_status || job.status;
  select("#generation-status-detail").textContent = heldForTaskMatch ? holdReason : rejected ? job.reference_acceptance?.reason_zh || "当前参考图验收不通过，暂停新提交。" : job.error?.message || (job.remote_task_id ? `真实服务端状态：${job.provider_status || "已提交"}。每 15 秒查询已有任务，不会重新生成。` : "参考图及提示词已准备。打开核对窗口不会上传或扣费，只有最终确认才会提交。");
  if(blocked){challenge=null;select("#generation-confirmation").close();select("#generation-confirm-submit").disabled=true;}
  const reference=snapshot.scenarios.find(scene=>scene.scenario_id===job.scenario_id);
  const realReference=reference.reference_status?.startsWith("real_");
  select("#generation-title").textContent=heldForTaskMatch?"单一参考方案待任务匹配确认":rejected?"当前参考图暂停生成":job.title;
  select("#generation-lead").textContent=heldForTaskMatch?"这是助手对单一参考方案的暂缓判断，并非用户正式否决。11 张干床图的认可和原始 N 标签保持。":rejected?"此参考图已被用户拒绝，保留核对问题，不继续付费生成。":realReference?"使用已认可的真实N细流画面作为首帧，避免再生成背景和水体。它不是干床图；新结果仍须检查，尚未纳入训练。":"核对本次首帧与提示词后，可单独确认这一条生成。";
  select("#generation-reference-title").textContent=heldForTaskMatch?"含沙细流备选参考 · 待任务匹配":rejected?"被拒绝的历史首帧":realReference?"本次首帧：真实N细流原图":"本次上传首帧";
  select("#generation-reference-note").textContent=heldForTaskMatch?"黄色本身不等于不合格。这张图保留为含沙细流备选，但不足以单独覆盖普通水流、其他机位和洪水的不同外观；原 N 标签含义不变。":rejected?"用户验收不通过，不再用于付费生成。":realReference?"原始JPEG字节，包含已认可的浅黄细流；山体、植被、河岸和水池没有经过生图重绘。仅登记为本次媒体试片参考，上传及1.50元费用须最后确认。":"直接复用你认可为普通水流的demo灰水图，原文件字节保持。认可范围是本条媒体试片首帧内容；上传及1.50元费用须最终确认，生成结果与训练标签另行验收。";
  select("#generation-reference").src=reference.reference_url;
  select("#generation-reference-open").href=reference.reference_url;
  if (!promptDirty) {select("#generation-prompt-zh").value=job.prompt_zh;select("#generation-prompt-en").value=job.prompt_en;}
  ["#generation-prompt-zh","#generation-prompt-en"].forEach(id=>{select(id).disabled=!editable||requestBusy});
  select("#generation-save").disabled=!editable||requestBusy||!promptDirty;
  select("#generation-start").disabled=blocked||!job.user_confirmation_available||requestBusy||promptDirty;
  select("#generation-start").textContent=heldForTaskMatch?"助手暂停方案 · 待任务匹配确认":rejected?"当前参考图验收不通过 · 暂停生成":"核对并生成这一条 · ¥1.50";
  select("#generation-credential").textContent=snapshot.credential_configured?"已配置（仅检查存在）":"尚未配置";
  select("#generation-reserved").textContent=`¥${job.cost_reserved_cny.toFixed(2)}`;
  select("#generation-task").textContent=job.remote_task_id||"尚未创建";
  select("#generation-poll").disabled=!job.poll_allowed||requestBusy;
  select("#generation-download").disabled=!job.download_allowed||requestBusy;
  select("#generation-qc").disabled=!job.qc_allowed||requestBusy;
  select("#generation-blockers").textContent=heldForTaskMatch?holdReason:promptDirty?"提示词已修改，请先保存。":job.user_confirmation_available?"新请求、新首帧、新费用确认；此前授权不用于这一条。":(job.user_confirmation_blockers||[]).join("；");
  select("#generation-id").textContent=`作业：${job.id} · 合成媒体预览，尚未纳入训练`;
  if (job.local_video_url && resultUrl!==job.local_video_url) {
    resultUrl=job.local_video_url;
    const video=document.createElement("video");video.controls=true;video.preload="metadata";video.src=resultUrl;video.setAttribute("aria-label","新生成的普通水流动态试片");
    select("#generation-result").replaceChildren(video);
    select("#generation-result-download").hidden=false;select("#generation-result-download").href=resultUrl;
    select("#generation-result-note").textContent="已获取本条真实生成结果。请检查水流连贯性和固定背景；尚未作为训练样本验收。";
  }
  const activity=select("#generation-activity"); activity.replaceChildren();
  snapshot.activity.slice(-15).reverse().forEach(event=>{const li=document.createElement("li");li.textContent=`${new Date(event.at).toLocaleTimeString("zh-CN",{hour12:false})} · ${event.message}`;activity.append(li)});
  if(job.qc)select("#generation-qc-note").textContent=`技术检查：${job.qc.technical_status || "待完成"}。技术结果不代表水流和背景已经人工验收。`;
  const videoReview=job.qc?.content_acceptance;
  if(videoReview?.status==="rejected_by_user") {
    select("#generation-status").textContent="服务端生成完成 · 视频内容验收不通过";
    select("#generation-status-detail").textContent=videoReview.reason_zh;
    select("#generation-lead").textContent="第一条动态视频已被你判为不通过；首帧内容认可保留，本条不重试、不进入训练。";
    select("#generation-result-note").textContent=`视频内容验收不通过：${videoReview.reason_zh} 原视频、任务回执及费用记录保留。`;
    select("#generation-start").disabled=true;
    select("#generation-start").textContent="本条视频验收不通过 · 不重试";
    select("#generation-qc").disabled=true;
    select("#generation-qc-note").textContent="用户动态内容验收不通过；技术检查状态与服务端SUCCEEDED记录分别保留。";
  }
}
async function refresh() {
  if(requestBusy)return;
  try {render(await api("/api/generation"));select("#generation-error").hidden=true;}catch(error){errorMessage(error)}
}
async function action(name) {
  if(requestBusy||!current)return;
  requestBusy=true;render(current);
  try {const value=await api(jobPath(name),"POST",{});render(value.workbench);select("#generation-error").hidden=true;}
  catch(error){errorMessage(error)}
  finally{requestBusy=false;render(current)}
}
select("#generation-refresh").addEventListener("click",refresh);
select("#generation-poll").addEventListener("click",()=>action("poll"));
select("#generation-download").addEventListener("click",()=>action("download"));
select("#generation-qc").addEventListener("click",()=>action("qc"));
["#generation-prompt-zh","#generation-prompt-en"].forEach(id=>select(id).addEventListener("input",()=>{promptDirty=true;select("#generation-save-note").textContent="提示词已修改，请保存后再确认生成。";select("#generation-save").disabled=false;select("#generation-start").disabled=true}));
select("#generation-prompt-form").addEventListener("submit",async event=>{
  event.preventDefault();if(requestBusy)return;
  const payload={prompt_zh:select("#generation-prompt-zh").value,prompt_en:select("#generation-prompt-en").value};
  requestBusy=true;
  try{const snapshot=await api(`/api/jobs/${current.jobs[0].id}`,"PATCH",payload);promptDirty=false;challenge=null;select("#generation-save-note").textContent="提示词已保存。";render(snapshot)}catch(error){errorMessage(error)}finally{requestBusy=false;render(current)}
});
select("#generation-start").addEventListener("click",async()=>{
  if(requestBusy||promptDirty)return;
  requestBusy=true;render(current);
  try{
    const value=await api(jobPath("prepare-confirmation"),"POST",{});challenge=value.confirmation;render(value.workbench);
    select("#generation-confirm-reference").src=challenge.reference_url;
    select("#generation-confirm-prompt").textContent=challenge.prompt_en;
    select("#generation-confirm-fingerprint").textContent=`本次请求指纹 ${challenge.request_sha256.slice(0,16)}…`;
    select("#generation-consent").checked=false;select("#generation-confirm-submit").disabled=true;select("#generation-confirm-error").hidden=true;
    select("#generation-confirmation").showModal();
  }catch(error){errorMessage(error)}finally{requestBusy=false;render(current)}
});
function closeConfirmation(){if(requestBusy)return;challenge=null;select("#generation-confirmation").close()}
select("#generation-confirm-close").addEventListener("click",closeConfirmation);
select("#generation-confirm-cancel").addEventListener("click",closeConfirmation);
select("#generation-confirmation").addEventListener("cancel",()=>{challenge=null});
select("#generation-consent").addEventListener("change",()=>{select("#generation-confirm-submit").disabled=!select("#generation-consent").checked||requestBusy});
select("#generation-confirm-form").addEventListener("submit",async event=>{
  event.preventDefault();if(requestBusy||!challenge||!select("#generation-consent").checked)return;
  requestBusy=true;select("#generation-confirm-submit").disabled=true;select("#generation-confirm-submit").textContent="正在提交这一条…";
  try{
    const value=await api(jobPath("confirm-submit"),"POST",{confirmation_token:challenge.token,request_sha256:challenge.request_sha256,accepted_upload:true,max_cost_cny:1.5});
    challenge=null;select("#generation-confirmation").close();render(value.workbench);
  }catch(error){select("#generation-confirm-error").textContent=error.message;select("#generation-confirm-error").hidden=false;await refresh()}
  finally{requestBusy=false;select("#generation-confirm-submit").textContent="确认上传并生成 · ¥1.50";select("#generation-confirm-submit").disabled=!select("#generation-consent").checked;render(current)}
});
refresh();
setInterval(async()=>{
  if(requestBusy)return;
  await refresh();
  const job=current?.jobs[0];
  if(job?.poll_allowed)await action("poll");
  if(current?.jobs[0].download_allowed&&!autoDownloadAttempted){autoDownloadAttempted=true;await action("download")}
},15000);
