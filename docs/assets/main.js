(function(){
  "use strict";

  /* ---------- RMSLE / RMSE calculator ---------- */
  var log1p = Math.log1p || function(x){ return Math.log(1+x); };
  var rows = document.querySelectorAll(".calc .r");
  var defaults = [[5,15],[1000,3000],[0,-2],[200,195]];

  function compute(){
    var seGap=0, seLog=0, n=0;
    rows.forEach(function(r){
      var a = parseFloat(r.querySelector(".ac").value);
      var p = parseFloat(r.querySelector(".pr").value);
      var clipEl = r.querySelector(".clip");
      if (isNaN(a) || isNaN(p)) { clipEl.textContent=""; return; }
      var pc = Math.max(0, p);
      clipEl.textContent = "→ " + pc;
      clipEl.classList.toggle("flag", p < 0);
      var ac = Math.max(0, a);
      seGap += Math.pow(pc - a, 2);
      seLog += Math.pow(log1p(pc) - log1p(ac), 2);
      n++;
    });
    if (n === 0){ return; }
    var rmse = Math.sqrt(seGap / n);
    var rmsle = Math.sqrt(seLog / n);
    var rmseEl = document.getElementById("rmseVal");
    var rmsleEl = document.getElementById("rmsleVal");
    if (rmseEl) rmseEl.textContent = rmse >= 100 ? rmse.toFixed(1) : rmse.toFixed(3);
    if (rmsleEl) rmsleEl.textContent = rmsle.toFixed(3);
  }
  var recalcBtn = document.getElementById("recalc");
  if (recalcBtn) recalcBtn.addEventListener("click", compute);
  document.querySelectorAll(".calc input").forEach(function(i){
    i.addEventListener("input", compute);
  });
  var resetBtn = document.getElementById("reset");
  if (resetBtn) resetBtn.addEventListener("click", function(){
    rows.forEach(function(r,idx){
      r.querySelector(".ac").value = defaults[idx][0];
      r.querySelector(".pr").value = defaults[idx][1];
    });
    compute();
  });
  if (rows.length) compute();

  /* ---------- weekly seasonality bar chart ---------- */
  var week = [
    {d:"Mon", v:2383}, {d:"Tue", v:2409}, {d:"Wed", v:2770}, {d:"Thu", v:2229},
    {d:"Fri", v:2414}, {d:"Sat", v:2323}, {d:"Sun", v:1031}
  ];
  var chart = document.getElementById("weekChart");
  var axis = document.getElementById("weekAxis");
  if (chart && axis) {
    var maxV = Math.max.apply(null, week.map(function(w){return w.v;}));
    var minV = Math.min.apply(null, week.map(function(w){return w.v;}));
    week.forEach(function(w){
      var bar = document.createElement("div");
      bar.className = "bar";
      bar.style.height = "0%";
      bar.title = w.d + ": " + w.v;
      var v = document.createElement("span");
      v.className = "v"; v.textContent = w.v;
      bar.appendChild(v);
      chart.appendChild(bar);
      requestAnimationFrame(function(){
        bar.style.height = Math.round((w.v / maxV) * 100) + "%";
      });
      var lab = document.createElement("div");
      lab.textContent = w.d;
      if (w.v === minV) lab.className = "lowest";
      axis.appendChild(lab);
    });
  }

  /* ---------- scroll spy + progress bar ---------- */
  var links = Array.prototype.slice.call(document.querySelectorAll(".navlink"));
  var sections = links.map(function(a){
    var href = a.getAttribute("href");
    if (!href || href.charAt(0) !== "#") return null;
    return document.querySelector(href);
  });
  var prog = document.getElementById("progbar");

  function onScroll(){
    var y = window.scrollY + 120;
    var current = 0;
    sections.forEach(function(sec, i){ if (sec && sec.offsetTop <= y) current = i; });
    links.forEach(function(a,i){ a.classList.toggle("active", i === current); });
    if (prog){
      var h = document.documentElement.scrollHeight - window.innerHeight;
      prog.style.width = (h > 0 ? (window.scrollY / h) * 100 : 0) + "%";
    }
  }
  window.addEventListener("scroll", onScroll, {passive:true});
  onScroll();
})();
