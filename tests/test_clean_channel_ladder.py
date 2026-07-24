from src.evaluation.clean_end_to_end import clean_ladder_conditions

def test_clean_ladder_has_c0_to_c4_and_paired_snr_slices():
    rows=clean_ladder_conditions([30,20,15,10,5],seed=23)
    assert rows[0]['stage']=='C0' and rows[1]['stage']=='C1'
    assert {r['stage'] for r in rows}=={'C0','C1','C2','C3','C4'}
    for snr in (30,20,15,10,5):
        subset=[r for r in rows if r.get('snr_db')==snr]
        assert len(subset)==3 and len({r['seed'] for r in subset})==1
