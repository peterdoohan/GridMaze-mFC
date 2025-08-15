

from GridMaze.analysis.nbeGLM.get_input_data import get_input_data
import nbeGLM
from nbeGLM.models import nbeGLM as Model
from importlib import reload
reload(nbeGLM.models)
from nbeGLM.models import nbeGLM as Model

data = get_input_data(input_groups=["distance_to_goal", "place_direction"])

model1 = Model(Nhid=[100, 50],
                Nlat=10,
                partition=None,
                latent_split = None)

model2 = Model(Nhid=[100, 50],
                Nlat=10,
                partition=(("distance_to_goal",), ("place_direction",)),
                latent_split = None)

model3 = Model(Nhid=[100, 50],
                Nlat=10,
                partition=(("distance_to_goal",), ("place_direction",)),
                latent_split = (5,5))


for model in [model1, model2, model3]:
    print()
    model.set_input_groups(data)
    model.initialise_weights(data)
    print(model)
    for p in model.parameters():
        print(p.shape)
        
        
    model.train(data,test_freq=1,nepochs=3,verbose = True)


