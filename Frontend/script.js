const input=document.getElementById("textInput"),count=document.getElementById("charCount"),button=document.getElementById("analyzeBtn"),error=document.getElementById("errorMessage");
const placeholder=document.getElementById("placeholder"),content=document.getElementById("resultContent"),nameEl=document.getElementById("sentimentName"),icon=document.getElementById("sentimentIcon"),confidence=document.getElementById("confidenceValue"),fill=document.getElementById("confidenceFill");
const vals={negative:document.getElementById("negativeValue"),neutral:document.getElementById("neutralValue"),positive:document.getElementById("positiveValue")};
const bars={negative:document.getElementById("negativeBar"),neutral:document.getElementById("neutralBar"),positive:document.getElementById("positiveBar")};

input.addEventListener("input",()=>count.textContent=`${input.value.length} / 2000`);
document.querySelectorAll(".example-btn").forEach(b=>b.onclick=()=>{input.value=b.dataset.text;input.dispatchEvent(new Event("input"));input.focus()});

function theme(s){
 const t={positive:["✓","#52d7a5","rgba(82,215,165,.11)"],neutral:["—","#f0b85b","rgba(240,184,91,.11)"],negative:["×","#ff6678","rgba(255,102,120,.11)"]}[s]||["—","#f0b85b","rgba(240,184,91,.11)"];
 icon.textContent=t[0];icon.style.color=t[1];icon.style.background=t[2];nameEl.style.color=t[1];
}
function updateBars(p){
 for(const s of ["negative","neutral","positive"]){const v=Number(p[s]||0);vals[s].textContent=`${v.toFixed(2)}%`;bars[s].style.width=`${v}%`;bars[s].style.background={negative:"#ff6678",neutral:"#f0b85b",positive:"#52d7a5"}[s]}
}
async function analyze(){
 const text=input.value.trim();error.textContent="";if(!text){error.textContent="Please enter a sentence first.";return}
 button.disabled=true;button.querySelector("span").textContent="Analyzing...";
 try{
  const r=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text})});
  const d=await r.json();if(!r.ok)throw new Error(d.error||"Prediction failed.");
  const s=String(d.sentiment).toLowerCase(),p=d.probabilities||{},c=Number(p[s]||0);
  nameEl.textContent=s;theme(s);updateBars(p);confidence.textContent=`${c.toFixed(2)}%`;fill.style.width="0%";requestAnimationFrame(()=>fill.style.width=`${c}%`);
  placeholder.classList.add("hidden");content.classList.remove("hidden");
 }catch(e){error.textContent=e.message}finally{button.disabled=false;button.querySelector("span").textContent="Analyze Sentiment"}
}
button.onclick=analyze;
input.addEventListener("keydown",e=>{if((e.ctrlKey||e.metaKey)&&e.key==="Enter")analyze()});
