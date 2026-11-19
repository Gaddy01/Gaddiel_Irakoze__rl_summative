/* Interactive Three.js industrial landscape. Actual project stages/metrics
   come from Python's TradeSignalEnv public_state() only. */
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const root=document.getElementById('threeWorld');
const fallback=document.getElementById('sceneFallback');
const scene=new THREE.Scene(); scene.background=new THREE.Color('#122338');scene.fog=new THREE.FogExp2('#152b3c',0.014);
const camera=new THREE.PerspectiveCamera(35,1,.1,200);
camera.position.set(24,26,29);
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'high-performance'});
renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));
renderer.setSize(Math.max(1,root.clientWidth),Math.max(1,root.clientHeight));
renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.7;
root.appendChild(renderer.domElement);
fallback?.classList.add('loaded');
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.06;controls.minDistance=15;controls.maxDistance=59;controls.maxPolarAngle=Math.PI/2.1;controls.target.set(0,0,0);
scene.add(new THREE.HemisphereLight(0xb3daff,0x22333c,2.2));
const sun=new THREE.DirectionalLight(0xffecce,3.2);sun.position.set(-11,22,16);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-24;sun.shadow.camera.right=24;sun.shadow.camera.top=24;sun.shadow.camera.bottom=-24;sun.shadow.normalBias=.045;scene.add(sun);
const accentLight=new THREE.PointLight(0x33e3c7,35,33);accentLight.position.set(0,4,0);scene.add(accentLight);
const mat=(color,roughness=.8,metalness=.06)=>new THREE.MeshStandardMaterial({color,roughness,metalness});
const ground=mat('#132b37');const concrete=mat('#354c5b');const road=mat('#182b38');const roadLine=mat('#c4c7aa');const steel=mat('#89a7b5',.45,.47);const glass=mat('#91d7ed',.22,.45);const roof=mat('#243d4f');const greenery=mat('#356b63');
function box(w,h,d,m,x,y,z,group=scene){const mesh=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),m);mesh.position.set(x,y,z);mesh.castShadow=true;mesh.receiveShadow=true;group.add(mesh);return mesh;}
function cylinder(r,h,m,x,y,z,group=scene,r2=r){const mesh=new THREE.Mesh(new THREE.CylinderGeometry(r2,r,h,10),m);mesh.position.set(x,y,z);mesh.castShadow=true;mesh.receiveShadow=true;group.add(mesh);return mesh;}
box(43,.8,35,ground,0,-.55,0);box(40,.15,32,mat('#1c3645'),0,-.06,0);
box(4,.04,32,road,0,.05,0);box(40,.04,4,road,0,.07,0);
for(let i=-17;i<=17;i+=4){box(.12,.008,1.5,roadLine,0,.087,i);box(1.5,.008,.12,roadLine,i,.092,0);}
const core=new THREE.Group();scene.add(core);cylinder(1.8,.22,mat('#1e5761'),0,.24,0,core);cylinder(1.15,1.5,mat('#274d61'),0,1.1,0,core);cylinder(.85,.2,glass,0,1.95,0,core);cylinder(.18,2,steel,0,2.9,0,core);const signal=new THREE.Mesh(new THREE.SphereGeometry(.31,16,12),new THREE.MeshStandardMaterial({color:'#7cffe0',emissive:'#3dffda',emissiveIntensity:2}));signal.position.set(0,3.97,0);core.add(signal);
const pads={
 agro_processing:{x:-10.6,z:-8.1,color:'#e8ad69'},
 textiles:{x:10.5,z:-8.1,color:'#9ca7fa'},
 construction_materials:{x:-10.6,z:8.1,color:'#78b9d4'},
 pharmaceuticals:{x:10.5,z:8.1,color:'#73dec2'},
};
const districts={};let activeSector='construction_materials';
function addFactory(group,kind,color){
 const wall=mat('#879aa7'),light=mat(color,.6,.2),dark=mat('#3f6070');
 box(6.5,2.5,4.6,wall,0,1.25,0,group);
 box(6.9,.30,4.9,roof,0,2.65,0,group);
 for(let i=-2;i<=2;i++){box(.62,.9,.045,glass,i*1.14,1.62,2.33,group);box(.62,.9,.045,glass,i*1.14,1.62,-2.33,group);}
 if(kind==='agro_processing'){
   cylinder(.94,1.8,steel,-1.8,3.65,-.75,group);cylinder(.94,1.8,steel,.4,3.65,-.75,group);
   cylinder(.45,1.3,light,2.15,3.43,0,group);cylinder(.16,1.2,dark,2.15,4.58,0,group);
   box(2.1,.9,.9,mat('#a66d51'),0,.5,3.32,group);
 } else if(kind==='textiles'){
   for(let i=-2;i<=2;i++)box(.45,.25,4.1,light,i*1.12,2.97,0,group);
   box(6.8,1.0,1.4,mat('#536d89'),0,.5,3.12,group);
   for(let i=-2;i<=2;i++)box(.62,.6,.08,glass,i*1.1,.56,3.86,group);
 } else if(kind==='construction_materials'){
   cylinder(1.1,3.7,steel,-2.0,2.0,-.9,group);cylinder(.88,3,mat('#6b838c'),.2,1.7,-.8,group);
   cylinder(.24,4.0,light,2.45,3.0,-.8,group);box(3.1,.14,.8,light,1.0,3.4,.6,group);
   box(2,.7,1.5,mat('#b9a68f'),-1,.4,3.0,group);
 } else {
   box(2.6,2.1,2.0,light,1.6,3.55,-.7,group);
   box(.32,1.5,3.9,glass,0,3.56,0,group);
   box(1.3,.18,.3,mat('#e8f7ee'),1.6,4.85,-.7,group);
   box(.25,.3,1.3,mat('#e8f7ee'),1.6,4.87,-.7,group);
 }
}
function addTrees(group){const trunk=mat('#705744');for(const [x,z] of [[-4,3.2],[4,3.2],[-4,-3.2],[4,-3.2]]){cylinder(.10,.7,trunk,x,.5,z,group);const crown=new THREE.Mesh(new THREE.IcosahedronGeometry(.57,0),greenery);crown.position.set(x,1.2,z);group.add(crown);}}
for(const [key,s] of Object.entries(pads)){
 const group=new THREE.Group();group.position.set(s.x,0,s.z);scene.add(group);
 box(14,.30,11,mat('#243b4b'),0,.15,0,group);
 const rim=box(13.55,.07,10.6,mat(s.color,.7,.15),0,.35,0,group);rim.material.transparent=true;rim.material.opacity=.30;
 box(13.25,.13,10.25,concrete,0,.46,0,group);
 const factory=new THREE.Group();factory.position.set(0,.54,0);group.add(factory);
 addFactory(factory,key,s.color);addTrees(group);
 const activeRing=box(13.9,.07,10.9,mat('#55ffdd'),0,.49,0,group);activeRing.material.transparent=true;activeRing.material.opacity=.03;
 districts[key]={group,factory,ring:activeRing,bar:rim,position:new THREE.Vector3(s.x,4.5,s.z),stage:'',target:0.2,color:s.color};
}
const edge=mat('#4d697a');for(const x of [-19,19])box(.2,.15,32,edge,x,.15,0);for(const z of [-15.7,15.7])box(38,.15,.2,edge,0,.15,z);
const drones=[];for(let i=0;i<3;i++){const drone=new THREE.Mesh(new THREE.IcosahedronGeometry(.17,0),new THREE.MeshBasicMaterial({color:i===1?'#ffca89':'#72ffe3'}));scene.add(drone);drones.push(drone);}
function mapStage(s){const k=String(s||'unstudied').toLowerCase();if(k==='operating')return 1;if(k==='pilot')return .70;if(k==='studied')return .43;if(k==='exited')return .13;return .19;}
function update(state){const sectors=state?.policies?.ppo?.portfolio?.sectors||{};for(const [id,dd] of Object.entries(districts)){const data=sectors[id]||{};dd.stage=data.stage||'unstudied';dd.target=mapStage(dd.stage);dd.bar.material.opacity=dd.stage==='operating'?.92:.23;}}
function focus(id){if(!districts[id])return;activeSector=id;const d=districts[id];controls.target.set(d.group.position.x*.26,.5,d.group.position.z*.26);}
let tick=0;function frame(){requestAnimationFrame(frame);tick+=.012;controls.update();signal.material.emissiveIntensity=1.3+Math.sin(tick*3)*.6;
 for(const [id,d] of Object.entries(districts)){
   const op=d.factory.scale.y;d.factory.scale.y=THREE.MathUtils.lerp(op,d.target,.056);
   d.ring.material.opacity=THREE.MathUtils.lerp(d.ring.material.opacity,id===activeSector?.34+.13*Math.sin(tick*2):.024,.065);
   const label=document.getElementById('world-label-'+id);if(label){const projected=d.position.clone().project(camera);const visible=projected.z<1&&projected.z>-1;label.style.opacity=visible?'.94':'0';label.style.left=((projected.x+1)/2*root.clientWidth)+'px';label.style.top=((-projected.y+1)/2*root.clientHeight)+'px';label.style.transform='translate(-50%,-50%)';}
 }
 drones.forEach((d,i)=>{const a=tick*.27+i*2.1;d.position.set(Math.cos(a)*7,2.5+Math.sin(a*3)*.3,Math.sin(a)*5.4);});
 renderer.render(scene,camera);
}
function resize(){const w=Math.max(1,root.clientWidth),h=Math.max(1,root.clientHeight);camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);}
if(typeof ResizeObserver!=='undefined')new ResizeObserver(resize).observe(root);else window.addEventListener('resize',resize);
resize();frame();
window.IndustrialWorld={update,focus};window.dispatchEvent(new Event('industrial-world-ready'));
