"""A real Self-Harness round backed exclusively by native Isaac Sim episodes."""
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness, now, selection
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, bind_harness_modes, verify_harness
from .task import TASK, SCOPE, layout
from .policy import propose_repair

PROTOCOL='isaac-lab-sample-transfer-v2'


def implementation():
    root=Path(__file__).parent
    return {p.name:file_digest(p) for p in sorted(root.glob('*.py'))}


def manifest(root,name,**extra):
    components={k:{k+'.json':file_digest(root/(k+'.json'))} for k in CLOSURE}
    components['dependencies'].update({str(p.relative_to(root)):file_digest(p) for p in sorted((root/'source').glob('*.py'))})
    return bind_harness_modes(dict(id=name,root=str(root),components=components,**extra))


def verify_episode(path):
    path=Path(path);result=read_json(path/'result.json')
    if result['state']!='completed' or result['backend']!='isaacsim-5.1':
        raise ValueError('Native completion required')
    for name,sha in result['evidence'].items():
        p=(path/name).resolve()
        if not p.is_relative_to(path.resolve()) or file_digest(p)!=sha:
            raise ValueError('Episode evidence changed')
    if result['layout_sha256']!=digest(read_json(path/'layout.json')):
        raise ValueError('Layout identity differs')
    return result


class Evaluation:
    def __init__(self,python,workspace,gui=False):
        self.python=str(Path(python).expanduser().absolute());self.workspace=Path(workspace);self.gui=gui
    def identity(self):
        from .lab_style import asset_identity
        return dict(protocol=PROTOCOL,implementation=implementation(),assets=asset_identity(),python=self.python,gui=self.gui)
    def admit(self,candidate,output):
        memory=read_json(Path(candidate['root'])/'memory_rules.json')
        accepted=(read_json(Path(candidate['root'])/'dependencies.json')==implementation()
                  and .10<=memory['transit_height_m']<=.55 and memory['sample_height_m']==.12)
        return dict(accepted=accepted,freeze_sha256=verify_harness(candidate),
                    evidence={'implementation':implementation()},reason='Reviewed bounded code policy and fixed sensor/evaluator boundary')
    def execute(self,candidate,seed,output):
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():raise ValueError('Episode output already exists; inspect evidence before retry')
        python=Path(self.python)
        if not python.is_file():raise ValueError('Isaac Python executable not found')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMNI_KIT_ACCEPT_EULA='YES',
                 PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        # Own scratch paths only. Leave inherited simulator and fleet settings untouched.
        scratch=self.workspace/'runtime';scratch.mkdir(parents=True,exist_ok=True)
        for key,name in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('XDG_CONFIG_HOME','config'),('XDG_DATA_HOME','data')]:
            p=scratch/name;p.mkdir(exist_ok=True);env[key]=str(p)
        icd=scratch/'nvidia-vulkan.json'
        atomic_json(icd,{'file_format_version':'1.0.0','ICD':{'library_path':('libGLX_nvidia.so.0' if self.gui else 'libEGL_nvidia.so.0'),'api_version':'1.3.194'}})
        env['VK_ICD_FILENAMES']=str(icd)
        command=[self.python,'-m','PhysicalRSI_demos.auto_engineering.isaac_scene',
                 '--seed',str(seed),'--memory',str(Path(candidate['root'])/'memory_rules.json'),'--output',str(output),
                 '--runtime-root',str(scratch/'kit')]
        if self.gui:command.append('--gui')
        log=output.parent/(output.name+'.log')
        with log.open('x') as stream:
            process=subprocess.run(command,cwd=Path(__file__).resolve().parents[2],env=env,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=1000)
        if process.returncode:
            raise RuntimeError('Isaac episode failed; inspect '+str(log))
        result=verify_episode(output)
        if result['seed']!=seed or result['memory']!=read_json(Path(candidate['root'])/'memory_rules.json'):
            raise ValueError('Native episode binding differs')
        return result
    def validation(self,comparison,output):
        count=comparison['profile']['tasks'][TASK]['episodes']
        seeds=[]
        while len(seeds)<count:
            seed=secrets.randbelow(2**31)
            if seed!=17 and seed not in seeds:seeds.append(seed)
        atomic_json(output/'cases.json',seeds)
        return dict(split='validation',comparison_sha256=digest(comparison),generated_at=now(),
            layouts={TASK:[digest(layout(seed)) for seed in seeds]},
            admission_evidence={'cases.json':file_digest(output/'cases.json')})
    def evaluate(self,candidate,comparison,cohort,output):
        rows=[]
        for seed in read_json(output/'cases.json'):
            relative=Path('episodes')/candidate['id']/str(seed)
            result=self.execute(candidate,seed,output/relative)
            rows.append(dict(task=TASK,layout_sha256=result['layout_sha256'],state='completed',
                score=int(result['success']),success=result['success'],
                evidence_sha256={str(relative/'result.json'):file_digest(output/relative/'result.json'),
                    **{str(relative/name):sha for name,sha in result['evidence'].items()}}))
        return dict(candidate_id=candidate['id'],kind='native_isaac_development',native_exit_code=0,
            comparison_sha256=digest(comparison),cohort_sha256=digest(cohort),
            freeze_sha256=verify_harness(candidate),evaluator_revision=PROTOCOL,episodes=rows)


class Proposal:
    def __init__(self,evaluator):self.evaluator=evaluator
    def identity(self):return dict(kind='bounded_public_clearance_repair',implementation=implementation())
    def develop(self,parent,output):
        result=self.evaluator.execute(parent,17,output/'development')
        # Explicit public boundary: no object state, private checks or layout coordinates.
        return dict(split='evolve',success=result['success'],public_observation=result['public_observation'],
            evidence={'development/result.json':file_digest(output/'development/result.json')})
    def propose(self,parent,feedback,output):
        memory=read_json(Path(parent['root'])/'memory_rules.json');repair=propose_repair(feedback,memory)
        atomic_json(output/'diagnosis.json',dict(kind='bounded engineering rule',
            observation=feedback['public_observation'],success=feedback['success'],
            hypothesis='Increase carrying height above visually measured obstruction plus sample clearance.',
            proposed_memory=repair,private_evaluator_state_read=False,
            limitations='One declared repair family; no claim of general autonomous engineering.'))
        if repair is None:return []
        root=output/'candidate';shutil.copytree(parent['root'],root);atomic_json(root/'memory_rules.json',repair)
        return [manifest(root,'clearance-repair',parent_sha256=verify_harness(parent),
            method='public-RGBD-clearance-repair',changes={'memory_rules':repair},
            environment={'protocol':PROTOCOL},evidence=feedback['evidence'],costs={'proposals':1})]


class Selector:
    def identity(self):return {'source':file_digest(Path(selection.__file__))}
    def select(self,*args,**kwargs):return selection.select_survivor(*args,**kwargs)


def create_seed(seed):
    seed=Path(seed);seed.mkdir(parents=True,exist_ok=False)
    for kind in CLOSURE:
        value={'kind':kind,'protocol':PROTOCOL,'scope':SCOPE}
        if kind=='foundation':value={'agent':'Codex-authored reviewed code policy','weights':None}
        if kind=='dependencies':value=implementation()
        if kind=='memory_rules':value={'transit_height_m':.13,'sample_height_m':.12}
        if kind=='skills':value={'source':(Path(__file__).parent/'policy.py').read_text()}
        if kind=='tools':value={'inputs':['RGB','metric depth','calibration','proprioception'],
                               'privileged_scene_state_exposed':False,'policy_process_isolation_attested':False}
        atomic_json(seed/(kind+'.json'),value)
    (seed/'source').mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(source,seed/'source'/source.name)
    return manifest(seed,'initial')


def run(workspace,python,episodes=3,gui=False):
    if not 1<=episodes<=50:raise ValueError('Use 1 to 50 paired validation layouts')
    root=Path(workspace).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'state/current.json').exists():
        raise ValueError('Use a fresh workspace; existing native evidence is retained')
    initial=create_seed(root/'seed')
    evaluator=Evaluation(python,root,gui);state=HarnessState(root/'state')
    loop=SelfHarness(root/'round-001',state=state,proposer=Proposal(evaluator),evaluator=evaluator,
        selector=Selector(),scope=SCOPE,max_candidates=1,
        profile={'tasks':{TASK:{'weight':1,'episodes':episodes,'score_range':[0,1],'maximum_regression':0}},
                 'minimum_gain':0,'tie_tolerance':1e-10},
        protocol={'identity':PROTOCOL,'evaluation_kind':'native_isaac_development','qualification':None})
    state.initialize(initial,policy=loop.config,scope=SCOPE)
    current=loop.run()
    result={'state':'completed','selected_revision':current['revision'],'selected_policy':current['harness']['id'],
            'scope':SCOPE,'qualification':None,'benchmark_wide_score':None}
    atomic_json(root/'result.json',result)
    from .report import render
    render(root)
    return result
