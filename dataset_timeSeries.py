import torch
from torch.utils.data import Dataset
import numpy as np
from create_data import callFunction
from useful import remove_parts_of_graph_encoder, remove_parts_of_graph_encoder_contiformer
from random import choice

#Dataset for encoder Interpolation
class TimeSeriesDataset_Interpolation_roundedInput(Dataset):
    def __init__(self, timeseries_count: int, x_values, args, batch_size) -> None:
        super().__init__()
        self.timeseries_count = timeseries_count
        self.x_values = x_values
        self.args = args
        self.batchSize = batch_size
        self.mask_size = np.random.uniform(0,400)

    def __len__(self):
        return self.timeseries_count
    
    def __getitem__(self, index):
        if(index == 0):
            self.mask_size = int(np.random.uniform(0,500))

        #random start value y_start within boundaries config["y_lim"][0]+1 and config["y_lim"][1]-1
        y_start = np.random.uniform(self.args.y_lim_low + 1,self.args.y_lim_high - 1)
        
        #calculate randomInt to select different time series generating functions
        timeSeries = [0,1,2,3,4,5,6,7]
        randomInt = choice(timeSeries)
        #0: low order
        #1: low order
        #2: low order
        #3: periodic
        #4: high order
        #5: high order (periodic sum)
        #6: discontinuous 
        #7: discontinuous 

        #calculate timeseries
        y_spline, y_noise_spline,min_value, max_value, noise_std = callFunction(x_values=self.x_values, y_start=y_start, random_number_range=[self.args.random_number_range_distribution, self.args.random_number_range_mean, self.args.random_number_range_std], spline_value=[self.args.spline_value_low, self.args.spline_value_high], vocab_size=self.args.vocab_size, randomInt=randomInt, noise_std=[self.args.noise_std_distribution, self.args.noise_std_mean, self.args.noise_std_std])

        #remove arbitrary parts of timeseries
        mask = remove_parts_of_graph_encoder_contiformer(self.x_values, self.mask_size, 10)
        # print((mask == 0).sum())
        mask = torch.tensor(mask, dtype=torch.bool)
        mask = ~mask
        # print((mask == True).sum())

        #mask index 1 -> keep
        #mask index 0 -> remove        
        # mask_indices = np.where(mask == 1)[0]

        #parameter for min-max scaling
        div_term = (max_value-min_value)          

        #perform min max scaling
        timeSeries_noisy = torch.tensor((y_noise_spline-min_value)/div_term,dtype=torch.float32)[:]
        timeSeries_groundTruth = torch.tensor((y_spline-min_value)/div_term,dtype=torch.float32)[:]


        return {
            "div_term": div_term,
            "min_value": min_value,
            "noisy_TimeSeries" : timeSeries_noisy,
            "groundTruth": timeSeries_groundTruth,
            "noise_std": noise_std,
            "mask": mask,
        }








