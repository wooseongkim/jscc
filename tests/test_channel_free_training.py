from src.evaluation.clean_end_to_end import CheckpointSelector

def test_latent_and_waveform_checkpoint_selection_can_disagree():
    selector=CheckpointSelector()
    selector.update(step=100,latent_loss=.4,delta_si_sdr=-8,path='a.pt')
    selector.update(step=200,latent_loss=.3,delta_si_sdr=-10,path='b.pt')
    assert selector.best_latent['path']=='b.pt';assert selector.best_waveform['path']=='a.pt'
