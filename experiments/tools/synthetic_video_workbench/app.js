"use strict";
const $ = (selector) => document.querySelector(selector);
const signatures = new Map();
const waterSignatures = new Map();
let busy = false;
const escapeText = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const asText = (value) => Array.isArray(value) ? value.map(asText).join("；") : typeof value === "object" && value ? JSON.stringify(value, null, 2) : String(value || "");
function getText(object, keys, fallback = "") {
  for (const key of keys) if (object[key]) return asText(object[key]);
  return fallback;
}
function prompt(title, text, kind = "") {
  return `<div class="prompt ${kind}"><h4>${title}</h4><p>${escapeText(text)}</p></div>`;
}
function media(title, url, type, alt, empty, overlay = "") {
  return `<figure><figcaption><span>${title}</span>${url ? `<a href="${escapeText(url)}" target="_blank" rel="noopener">${type === "video" ? "保存视频 ↗" : "查看大图 ↗"}</a>` : ""}</figcaption><div class="media">${url ? type === "video" ? `<video controls preload="none" src="${escapeText(url)}" aria-label="${escapeText(alt)}"></video>` : `<img src="${escapeText(url)}" alt="${escapeText(alt)}" loading="lazy">${overlay}` : `<div class="placeholder"><span aria-hidden="true">${type === "video" ? "▷" : "▧"}</span><strong>${escapeText(empty)}</strong><small>${type === "video" ? "图片就绪后在本地编码" : "完成后自动出现"}</small></div>`}</div></figure>`;
}
function regionOverlay(spatial) {
  const points = (value) => Array.isArray(value) ? value.map(pair => pair.map(Number).map(v => v * 1000).join(",")).join(" ") : "";
  const polygon = points(spatial.allowed_polygon_normalized), path = points(spatial.flow_path_normalized);
  if (!polygon) return "";
  return `<svg class="region-overlay" viewBox="0 0 1000 1000" preserveAspectRatio="none" role="img" aria-label="绿色为允许水域审查范围，黄色虚线为槽线，不代表已核实流向"><polygon points="${escapeText(polygon)}"/><polyline points="${escapeText(path)}"/></svg>`;
}
function renderScene(scene) {
  const spatial = scene.spatial || {}, metadata = scene.metadata || {}, record = scene.video_record || {}, technical = record.technical || {};
  const rejected = scene.image_status === "rejected_by_user";
  const accepted = scene.user_acceptance?.status === "accepted_by_user";
  const firstFrameRejected = scene.first_frame_acceptance?.status === "rejected_by_user";
  const needsRevision = rejected || scene.image_status === "needs_revision";
  const badge = rejected ? "用户验收不通过" : needsRevision ? "需修正" : accepted ? "用户已认可参考图内容" : scene.image_available ? "生成完成 · 待用户验收" : "干床图待生成";
  const staticPrompt = getText(spatial, ["static_prompt_zh", "dry_prompt_zh"], getText(metadata, ["prompt_zh", "dry_edit_prompt_zh"], "以此机位原图为参考，移除沟道内全部黄色、褐色及透明水流、泡沫和湿流表面，重建自然、干燥的砂石沟床。保持原有河岸、固定物体、视角和画幅；独立蓄水池保持原样。将这一张干床图连续重复为 250 帧，编码成 10 秒静态视频，不添加运动。"));
  const channel = getText(spatial, ["channel_description_zh", "region_description_zh", "channel_region_zh", "allowed_region_zh", "region_zh"], "此机位的沟道边界、入流口、出流口及固定区域正在整理，尚未形成可执行的空间规则。");
  const ordinary = getText(spatial, ["ordinary_water_prompt_zh", "ordinary_prompt_zh", "water_prompt_zh"], "待空间区域和水流首帧准备。当前不提交动态生成。");
  const flood = getText(spatial, ["flood_prompt_zh", "dynamic_prompt_zh"], "待空间区域和水流首帧准备。当前不提交动态生成。");
  const immutable = getText(spatial, ["immutable_regions_zh", "fixed_regions_zh", "preserve_zh", "protected_regions_zh"]);
  const observations = getText(metadata, ["observations_zh", "review_zh", "observations"]);
  const limitations = getText(metadata, ["limitations_zh", "limitations"]);
  const actualPrompt = getText(metadata, ["prompt", "dry_edit_prompt", "prompt_en"]);
  const videoFact = scene.video_url ? `${technical.duration_seconds ?? 10} 秒 · ${technical.decoded_frames ?? 250} 帧 · ${(technical.size_bytes / 1024 / 1024).toFixed(2)} MB · 1 张源图` : "视频尚未编码，图片完成后自动更新。";
  const node = document.createElement("article");
  node.className = "scene"; node.id = scene.scene_id; node.dataset.sceneId = scene.scene_id;
  node.innerHTML = `<div class="scene-heading"><div class="scene-title"><span class="scene-index">${String(scene.index).padStart(2,"0")}</span><div><h3>${escapeText(scene.camera)}</h3><p>${escapeText(scene.scene_id)}</p></div></div><div class="badges"><span class="badge ${needsRevision ? "warning" : scene.image_available ? "ready" : "pending"}">${badge}</span><span class="badge ${rejected ? "warning" : scene.video_url ? "ready" : "pending"}">${scene.video_url ? rejected ? "被拒图的静态预览" : "10 秒视频已编码" : "静态视频待完成"}</span></div></div>
    <div class="media-grid">${media("01 / 参考原图",scene.source_url,"image",`${scene.camera}参考原图`,"原图未找到")}${media(rejected ? "02 / 被拒生成图" : "02 / 干床生成图",scene.image_url,"image",`${scene.camera}干床生成图`,"等待干床图",regionOverlay(spatial))}${media(rejected ? "03 / 被拒图的静态预览" : "03 / 10 秒静态视频",scene.video_url,"video",`${scene.camera}10秒重复帧静态视频`,"等待静态视频")}</div>
    <div class="scene-footnote"><span>${escapeText(videoFact)}</span><span>${rejected ? "用户验收不通过：背景几何和纹理被不自然地重绘。原图与被拒图保留对照；不用于训练。" : needsRevision ? "当前图需修正；视频即使已编码，也仅为该图的静态预览。" : "无源运动；H.264 压缩可能带来微小像素差异。"}</span></div>
    ${prompt(rejected ? "历史静态制作提示词 / 中文" : "静态制作提示词 / 中文", staticPrompt, "static-prompt")}
    <div class="spatial-heading"><h4>动态水流的空间规则${scene.index === 1 ? "" : " · 提示词草案"}</h4><label class="region-label"><input class="region-toggle" type="checkbox" checked>显示审查区域</label></div>
    <div class="spatial-description"><p class="region-legend"><span></span>绿色：允许水域审查范围 · 黄色虚线：槽线（不代表流向）</p><p>${escapeText(channel)}</p>${immutable ? `<p><strong>保持固定：</strong>${escapeText(immutable)}</p>` : ""}${spatial.flow_direction_status ? `<p><strong>流向依据：</strong>${escapeText(asText(spatial.flow_direction_status))}</p>` : ""}<p><strong>先干床 → 单独制作有水首帧 → I2V 只动首帧已有水。</strong>Wan2.6 当前首帧接口未提供区域 mask；彩色区域仅供审查，不会修改或送出原图，也不保证模型严格遵守。</p></div>
    ${scene.first_frame_url ? `<div class="first-frame-block">${media(firstFrameRejected ? "普通水流首帧 / 用户验收不通过" : "普通水流首帧 / 待验收",scene.first_frame_url,"image",`${scene.camera}普通水流首帧图片示例`,"有水首帧待准备")}<p>${firstFrameRejected ? "此首帧已被用户拒绝；保留核对问题，不继续提交。" : "有水首帧须另行验收及确认具体请求后，才可提交动态生成。"}</p></div>` : ""}
    <div class="prompt-grid">${prompt("普通水流提示词 / 中文",ordinary)}${prompt("洪水提示词 / 中文",flood)}</div>
    <details class="evidence"><summary>查看实际编辑提示词、检查记录与空间坐标</summary>${actualPrompt ? `<h4>图片实际编辑提示词</h4><p lang="en">${escapeText(actualPrompt)}</p>` : "<p>图片编辑记录尚未写入。</p>"}${observations ? `<h4>已有检查记录</h4><p>${escapeText(observations)}</p>` : ""}${limitations ? `<h4>已知限制</h4><p>${escapeText(limitations)}</p>` : ""}${Object.keys(spatial).length ? `<h4>本机位空间规则记录</h4><pre>${escapeText(JSON.stringify(spatial,null,2))}</pre>` : ""}<p class="muted">上述空间坐标用于审查首帧和约束提示词，不作为服务端空间蒙版发送。单条制作页实际提交的是有水首帧与文字提示词。</p></details>`;
  return node;
}
function filterScenes() {
  const selected = $("#scene-filter").value;
  document.querySelectorAll("#scenes article.scene").forEach((node) => { node.hidden = selected !== "all" && node.dataset.sceneId !== selected; });
}
function render(data) {
  renderWaterGallery(data.ordinary_water);
  const summary = data.summary;
  const allRejected = summary.expected > 0 && summary.rejected_by_user === summary.expected;
  const allAccepted = summary.expected > 0 && summary.accepted_by_user === summary.expected;
  $("#batch-lead").textContent = `当前版本：${data.batch || "本地制作"}。原图、生成图和静态预览逐场景对照。`;
  $("#batch-status").textContent = allRejected ? "本版用户验收不通过 · 暂停付费生成" : allAccepted ? "11 张参考图内容已获用户认可" : "局部编辑修复版 · 逐张待内容验收";
  $("#batch-detail").textContent = allRejected ? "背景几何和纹理被不自然地重绘；保留被拒素材作历史对照。" : allAccepted ? "这 11 张干床参考图的内容已获认可。普通水流图和动态试片分别验收。" : "只替换沟道水域，背景保持原图。合成河床是假设，不能据像素检查或视频播放自动认定合格；未验收素材不进入训练。";
  $("#count-total").textContent = summary.generated;
  $("#count-generated").textContent = summary.accepted_by_user ?? 0;
  $("#count-video").textContent = summary.rejected_by_user ?? 0;
  $("#count-pending").textContent = summary.video_complete;
  const filter = $("#scene-filter"), previous = filter.value;
  if (filter.options.length !== data.scenes.length + 1) {
    filter.innerHTML = `<option value="all">全部 ${summary.expected} 个场景</option>` + data.scenes.map(scene => `<option value="${escapeText(scene.scene_id)}">${String(scene.index).padStart(2,"0")} · ${escapeText(scene.camera)}</option>`).join("");
    filter.value = previous;
  }
  if (!signatures.size) $("#scenes").replaceChildren();
  data.scenes.forEach((scene) => {
    const signature = JSON.stringify(scene);
    if (signatures.get(scene.scene_id) === signature) return;
    const node = renderScene(scene), existing = document.getElementById(scene.scene_id);
    if (existing) { node.querySelector("details").open = existing.querySelector("details").open; existing.replaceWith(node); }
    else $("#scenes").append(node);
    signatures.set(scene.scene_id, signature);
  });
  filterScenes();
  if (!$("#legacy-videos").children.length && data.legacy_previews.length) {
    $("#legacy").hidden = false;
    $("#legacy-videos").innerHTML = data.legacy_previews.map(item => media(item.title,item.url,"video",item.title,"")).join("");
  }
  $("#last-updated").textContent = `最近同步 ${new Date(data.updated_at).toLocaleTimeString("zh-CN",{hour12:false})} · 每 8 秒自动刷新`;
}
async function refresh() {
  if (busy) return;
  busy = true; $("#refresh-button").disabled = true;
  try {
    const response = await fetch("/api/workbench", {cache:"no-store"});
    if (!response.ok) throw new Error("读取状态失败");
    render(await response.json());
    await refreshWaterReference();
    $("#connection-status").textContent = "本地服务已连接";
    $("#connection-dot").classList.add("connected"); $("#error-banner").hidden = true;
  } catch (error) {
    $("#connection-status").textContent = "连接中断 · 自动重试";
    $("#connection-dot").classList.remove("connected"); $("#error-banner").hidden = false;
  } finally {
    busy = false; $("#refresh-button").disabled = false;
  }
}
function filterWaterScenes() {
  const selected = $("#water-scene-filter").value;
  document.querySelectorAll("#water-scenes article").forEach(node => {node.hidden = selected !== "all" && node.dataset.sceneId !== selected;});
}
function renderWaterGallery(data) {
  if (!data) {
    $("#water-batch-status").textContent = "此地址保留旧作业回执，尚未加载普通水流新批次";
    $("#water-batch-detail").textContent = "当前工作入口已改为 http://127.0.0.1:8773/，请在新入口核对并生成第一条。";
    document.querySelectorAll(".current-generation-link").forEach(link => {link.href="http://127.0.0.1:8773/generate";});
    $("#water-scenes").innerHTML = '<p><a href="http://127.0.0.1:8773/">打开普通水流新批次工作台 →</a></p>';
    return;
  }
  const summary = data.summary;
  $("#water-batch-lead").textContent = `当前版本：${data.batch}。先检查第一条动态效果，其余水流参考图陆续更新。`;
  $("#water-batch-status").textContent = `${summary.generated} / ${summary.expected} 张普通水流参考图已生成`;
  $("#water-batch-detail").textContent = `普通水流图内容已认可 ${summary.accepted_by_user} 张，待生成 ${summary.pending} 张；已认可干床图另列在下方。`;
  $("#water-count-total").textContent=summary.expected;
  $("#water-count-generated").textContent=summary.generated;
  $("#water-count-accepted").textContent=summary.accepted_by_user;
  $("#water-count-pending").textContent=summary.pending;
  const filter=$("#water-scene-filter"), previous=filter.value;
  if(filter.options.length!==data.scenes.length+1){filter.innerHTML='<option value="all">全部 11 个场景</option>'+data.scenes.map(scene=>`<option value="${escapeText(scene.scene_id)}">${String(scene.index).padStart(2,"0")} · ${escapeText(scene.camera)}</option>`).join("");filter.value=previous;}
  if(!waterSignatures.size)$("#water-scenes").replaceChildren();
  data.scenes.forEach(scene=>{
    const signature=JSON.stringify(scene);if(waterSignatures.get(scene.scene_id)===signature)return;
    const accepted=scene.user_acceptance?.status==="accepted_by_user";
    const rejected=scene.image_status==="rejected_by_user";
    const node=document.createElement("article");node.className="scene";node.id="water-"+scene.scene_id;node.dataset.sceneId=scene.scene_id;
    node.innerHTML=`<div class="scene-heading"><div class="scene-title"><span class="scene-index">${String(scene.index).padStart(2,"0")}</span><div><h3>${escapeText(scene.camera)}</h3><p>${escapeText(scene.scene_id)}</p></div></div><span class="badge ${rejected?"warning":scene.image_available?"ready":"pending"}">${rejected?"用户验收不通过":accepted?"水图内容已认可 · 本条媒体试片":scene.image_available?"水图已生成 · 待用户验收":"普通水流图生成中"}</span></div><div class="media-grid">${media("01 / 参考原图",scene.source_url,"image",scene.camera+"参考原图","原图未找到")}${media("02 / 已认可干床图",scene.dry_url,"image",scene.camera+"已认可干床图","干床图未找到")}${media("03 / 普通水流参考图",scene.image_url,"image",scene.camera+"普通水流参考图","等待本场景普通水流图")}</div><div class="scene-footnote"><span>${accepted?"你认可的demo水图直接复用；仅登记为第一条媒体试片参考。":scene.image_available?"新生成水图；内容待你逐张验收。":"此场景尚无水图，不以干床图替代完成状态。"}</span>${scene.index===1?'<a class="primary-link" href="/generate">核对并生成第一条 →</a>':'<span>动态视频尚未创建</span>'}</div><details class="evidence"><summary>查看来源与编辑记录</summary><p>输出：${escapeText(scene.output_path)}</p>${scene.current_sha256?`<p>SHA256：${escapeText(scene.current_sha256)}</p>`:""}<p>${escapeText(getText(scene.metadata,["observations_zh","limitations_zh","status"],"水图准备中；本页只按实际文件报告状态。"))}</p><p class="muted">普通水流参考图与视频均未登记训练真值，流向仅按已有证据逐场景核定。</p></details>`;
    const existing=document.getElementById(node.id);if(existing){node.querySelector("details").open=existing.querySelector("details").open;existing.replaceWith(node);}else $("#water-scenes").append(node);
    waterSignatures.set(scene.scene_id,signature);
  });
  filterWaterScenes();
}
async function refreshWaterReference() {
  try {
    const response = await fetch("/api/generation", {cache:"no-store"});
    if (!response.ok) throw new Error("reference unavailable");
    const data = await response.json(), job = data.jobs[0];
    const rejected = job.reference_acceptance?.status === "rejected_by_user";
    const heldForTaskMatch = job.reference_acceptance?.status === "paused_for_task_match_review" || $("#water-reference-card").dataset.heldJobId === job.id;
    if(job.id!=="ordinary-v116-water-20261108-05") {
      $("#water-reference-status").textContent="这是旧服务保留的历史作业。请使用 8773 新入口生成普通水流试片。";
      $("#water-reference-open").href="http://127.0.0.1:8773/";
      $("#water-reference-image").removeAttribute("src");
      return;
    }
    const reference = data.scenarios.find(scene => scene.scenario_id === job.scenario_id);
    if (reference?.reference_url) {
      const image = $("#water-reference-image");
      if (image.getAttribute("src") !== reference.reference_url) image.src = reference.reference_url;
      $("#water-reference-open").href = reference.reference_url;
    }
    const labels = {draft:"尚未提交视频",submitted:"视频已提交",running:"正在生成视频",succeeded:"视频生成完成",downloaded:"视频已保存",failed:"视频生成失败"};
    $("#water-reference-status").textContent = heldForTaskMatch ? "当前参考方案暂停，待核对。" : rejected ? job.reference_acceptance.reason_zh || "当前参考验收不通过。" : `当前作业：${labels[job.status] || job.status}。费用预留 ¥${job.cost_reserved_cny.toFixed(2)}；${job.remote_task_id?"已有真实远端任务。":"尚无远端任务，等待你的最终上传及费用确认。"}`;
    if(job.qc?.content_acceptance?.status==="rejected_by_user") {
      $("#water-reference-heading").textContent="第一条动态视频内容验收不通过";
      $("#water-reference-status").textContent=`${job.qc.content_acceptance.reason_zh} 首帧认可保留；服务端SUCCEEDED，费用记录¥${job.cost_reserved_cny.toFixed(2)}，本条不重试、不进入训练。`;
    }
  } catch (error) {
    $("#water-reference-status").textContent = "水流参考入口见右侧原图；当前作业状态请进入单条试片页查看。";
  }
}
$("#refresh-button").addEventListener("click", refresh);
$("#scene-filter").addEventListener("change",filterScenes);
$("#water-scene-filter").addEventListener("change",filterWaterScenes);
refresh(); setInterval(refresh,8000);
