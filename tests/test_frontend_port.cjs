// Run with node tests/test_frontend_port.cjs. Pure stubs: no browser or robot calls.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(__dirname + '/../frontend/index.html', 'utf8');
const source = html.split('<script>')[1].split('</script>')[0];
new vm.Script(source);
const data = {poses: {
  arm: {keyframes: [{name:'parking', values:[1,2,3,4,5,6]}]},
  arm2: {keyframes: [{name:'home', values:[6,5,4,3,2,1]}]},
  play_order: [{arm:'arm2',name:'home'},{arm:'arm',name:'parking'}]
}};
const calls = [];
const ctx = vm.createContext({
  console, window: {}, busy(){}, setBusyOverlay(){}, setStatus(){}, log(message) {
    if (message.includes('failed')) throw new Error(message);
  }, loadPoses: async()=>{},
  api: async (path,opts) => { calls.push([path,opts]); return data; }
});
const section=(start,end)=>source.slice(source.indexOf(start),source.indexOf(end,source.indexOf(start)));
vm.runInContext(section('function flatPoses(', 'async function loadPoses('),ctx);
vm.runInContext(section('async function runSequence(', 'document.getElementById("btnRunArm1")'),ctx);
vm.runInContext(section('window.moveKfUp =', 'document.getElementById("btnLoadPoses")'),ctx);
(async()=>{
  assert.deepEqual(JSON.parse(JSON.stringify(ctx.flatPoses(data).map(k=>k.arm))),['arm2','arm']);
  for (const kind of ['arm1','arm2','both']) {
    calls.length=0;
    await ctx.runSequence(kind);
    const request=calls.find(c=>c[0]==='/api/poses/run');
    const frames=JSON.parse(request[1].body).keyframes;
    assert.deepEqual(frames.map(k=>k.arm),kind==='both'?['arm','arm2']:[kind==='arm1'?'arm':'arm2']);
  }
  calls.length=0;
  await ctx.window.moveKfUp(1);
  const posts=calls.filter(c=>c[1]?.method==='POST');
  assert.equal(posts.length,1);
  assert.deepEqual(JSON.parse(posts[0][1].body).poses.map(k=>k.arm),['arm','arm2']);
  console.log('PASS: JS syntax, saved table order, per-arm sequences, and reorder ownership.');
})().catch(e=>{console.error(e);process.exitCode=1;});
