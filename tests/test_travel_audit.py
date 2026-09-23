"""The result auditor must catch corruption independently of the solver."""
import importlib.util
import json
from pathlib import Path
import pytest
from test_solver import fixture

spec=importlib.util.spec_from_file_location('travel_auditor',Path(__file__).parents[1]/'scripts/audit_travel_v5.py')
auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)


@pytest.mark.parametrize('tamper',['objective','route_identity'])
def test_saved_objective_tampering_is_rejected(tmp_path,tamper):
    instance,matrix=fixture()
    ip,mp=tmp_path/'instance.json',tmp_path/'matrix.json'
    auditor.write(ip,instance);auditor.write(mp,matrix)
    config={'name':'unit_fixture','engine':'native','backend':'0.13.4','seconds':1}
    row={'route_id':'fixture','station_code':'unit','date_YYYY_MM_DD':'2018-07-27','instance_path':str(ip),'matrix_path':str(mp)}
    hashes={str(ip):auditor.sha(ip),str(mp):auditor.sha(mp)}
    auditor.write(tmp_path/'status.json',{'status':'complete'})
    auditor.write(tmp_path/'stage_protocol.json',{'rows':[row],'configs':[config],'seeds':[42],'input_source_hashes':hashes})
    (tmp_path/'routes').mkdir()
    record={'route_id':'fixture','station_code':'unit','date':'2018-07-27','config':config,'seed':42,'instance_sha256':hashes[str(ip)],'matrix_sha256':hashes[str(mp)],
        'end_to_end_seconds':1,'search_seconds':.9,'solution':{'pyvrp_version':'0.13.4','feasible':True,
        'route':['D','A','B','C','D'],'objective_scaled':4500,'audit':{'metrics':{
            'travel_seconds':4.5,'waiting_seconds':0,'service_seconds':.375,'elapsed_seconds':4.875,
            'lateness_seconds':0,'time_window_violations':0,'capacity_excess_cm3':0,'demand_cm3':3.375}}}}
    auditor.write(tmp_path/'routes/fixture.json',[record])
    auditor.audit(tmp_path)
    if tamper=='objective':record['solution']['objective_scaled']=4501
    else:record['route_id']='different_route'
    auditor.write(tmp_path/'routes/fixture.json',[record])
    with pytest.raises(AssertionError):auditor.audit(tmp_path)
    assert not json.loads((tmp_path/'independent_audit.json').read_text())['passed']
