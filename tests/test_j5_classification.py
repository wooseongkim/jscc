from speech_jscc.diagnostics.j5_pilot import classify_j5, j5_gate

def m(imp=.2,corr=.4,power=.2): return {'relative_improvement_over_zero':imp,'pearson_correlation':corr,'power_ratio':power,'finite':True}
def result(): return {'aggregate':m(),'layers1_to_7':m(.12,.3,.12),'layers6_to_7':m(.08,.3,.1),'layer7':m(.05,.2,.06)}
def test_j5_pass_and_tail_classification():
    infra={'finite':True,'mask':True,'coverage':True,'diversity':True,'provenance':True,'gain_logging':True}
    g=j5_gate(result(),result(),infrastructure=infra,tail={'p10':.01,'negative_rate':.02})
    assert classify_j5(g,False,0)=='PASS'
    g=j5_gate(result(),result(),infrastructure=infra,tail={'p10':-.01,'negative_rate':.12})
    assert classify_j5(g,False,0)=='MARGINAL_TAIL'
