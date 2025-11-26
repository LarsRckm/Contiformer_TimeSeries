import numpy as np
import torch
import pandas as pd
from random import randint 


def round_numbers_individually(num_intervalls: int, time_series: torch.Tensor):
    smallest_number = 1/num_intervalls
    exp = calc_exp(smallest_number)
    lower_rounding_border = 0.5 * smallest_number
    time_series.apply_(lambda x: handle_single_number(x, smallest_number, lower_rounding_border, exp))

    return time_series

def handle_single_number(number: float, smallest_number: float, lower_rounding_border:float, exp: int):
    smallest_number = smallest_number * (10**exp)
    number = number * (10**exp)
    lower_rounding_border = lower_rounding_border * (10**exp)
    modulo_res = number % smallest_number
    if modulo_res < lower_rounding_border:
        rounded_value = round(number - modulo_res)
        rounded_value = int(rounded_value)/(10**exp)
        return rounded_value
    else:
        rounded_value = round(number - modulo_res + smallest_number)
        rounded_value = int(rounded_value)/(10**exp)
        return rounded_value

def calc_exp(smallest_number):
    step_copy = smallest_number
    exp = 0
    while step_copy%1 != 0:
        exp+=1
        step_copy*=10
    return exp

def index_to_value_dict(vocab_size_numbers: int, vocab_extra_tokens: list):
    total_len = vocab_size_numbers+len(vocab_extra_tokens)
    smallest_number = 1/vocab_size_numbers
    exp = calc_exp(smallest_number)
    number_copy = 0
    dictionary = {}
    i = 0
    while i<=vocab_size_numbers:
        dictionary[f"{i}"] = round_with_exp(number_copy, exp)
        number_copy += smallest_number
        i += 1
    j = i
    while i<=total_len:
        dictionary[f"{i}"] = vocab_extra_tokens[i-j]
        i += 1
    
    return dictionary

def value_to_index_dict(vocab_size_numbers: int, vocab_extra_tokens: list):
    total_len = vocab_size_numbers+len(vocab_extra_tokens)
    smallest_number = 1/vocab_size_numbers
    exp = calc_exp(smallest_number)
    number_copy = 0
    dictionary = {}
    i = 0
    while i<=vocab_size_numbers:
        dictionary[f"{round_with_exp(number_copy, exp)}"] = i
        number_copy += smallest_number
        i += 1    
    j = i
    while i<=total_len:
        dictionary[f"{vocab_extra_tokens[i-j]}"] = i
        i += 1
    return dictionary

def round_with_exp(x:float, exp: int):
    res = round(x*(10**exp))/(10**exp)
    return res

def sliding_window(spline_array, window_size):
    length_spline_array = len(spline_array)
    result = np.array([])
    for index, number in enumerate(spline_array, start=0):
        if index == 0:
            result = np.concatenate((result,[spline_array[index]]), axis = 0)
        elif index == length_spline_array-1:
            result = np.concatenate((result,[spline_array[index]]), axis = 0)
        elif index < window_size:#index < 200
            window_size_copy = index
            spline_array_lower_window = spline_array[:index]
            middle_value = spline_array[index]
            spline_array_upper_window = spline_array[index+1:index+1+window_size_copy]
            arr = np.concatenate((spline_array_lower_window,[middle_value],spline_array_upper_window), axis = 0)
            length = len(arr)
            sum = np.sum(arr)
            new_value = sum/length
            result = np.concatenate((result,[new_value]), axis = 0)
        elif index+window_size > length_spline_array-1:
            window_size_copy = length_spline_array - 1 - index 
            spline_array_lower_window = spline_array[index-window_size_copy:index]
            middle_value = spline_array[index]
            spline_array_upper_window = spline_array[index+1:length_spline_array]
            arr = np.concatenate((spline_array_lower_window,[middle_value],spline_array_upper_window), axis = 0)
            length = len(arr)
            sum = np.sum(arr)
            new_value = sum/length
            result = np.concatenate((result,[new_value]), axis = 0)
        else:
            spline_array_lower_window = spline_array[index-window_size:index]
            middle_value = spline_array[index]
            spline_array_upper_window = spline_array[index+1:index+1+window_size]
            arr = np.concatenate((spline_array_lower_window,[middle_value],spline_array_upper_window), axis = 0)
            length = len(arr)
            sum = np.sum(arr)
            new_value = sum/length
            result = np.concatenate((result,[new_value]), axis = 0)
    
    return result


def remove_parts_of_graph_encoder_contiformer(x_array, number_masking_tokens, offset):
    '''
    Docstring for remove_parts_of_graph_encoder_contiformer
    
    :param x_array: Description
    :param number_masking_tokens: Description
    :param offset: Description
    '''
    '''
    erstelle Maske mit gleich vielen Einträgen wie x_array
    mask = np.zeros_like(x_array)
    iteriere durch Maske, beginnend und endend mit dem Offset

    start_mask = [0+offset]
    end_mask = [1000-offset]
    position = np.random.randint(start_mask[0], end_mask[0])
    -> berechne mask_width = numpy.random.randint(0,number_masking_tokens), der die Breite des Masken arrays darstellt
    start_mask.append(position).sort()
    end_mask.append(position+mask_width).sort()
    mask[position:position+mask_width] = 1


    solange number_masking_tokens nicht 0:
        possible_locations = []
        solange possible_locations leer:
            -> berechne mask_width = numpy.random.randint(0, number_masking_tokens), der die Breite des Masken arrays darstellt
            -> berechne dann Ort, wo die Maske eingesetzt werden soll
            free_mask_width = start_mask[1] - start_mask[0]
            if (free_mask_width-2*offset) > mask_width:
                possible_locations.append([start_mask[0]+offset, start_mask[1]-offset])
            
            free_mask_width = end_mask[-1] - end_mask[-2]
            if (free_mask_width-2*offset) > mask_width:
                possible_locations.append([end_mask[-2]+offset, end_mask[-1]-offset])

            assert len(start_mask) == len(end_mask)
            for i in range(len(start_mask)-2):
                free_mask_width = start_mask[i+2] - end_mask[i]
                if (free_mask_width-2*offset) > mask_width:
                    possible_locations.append([end_mask[i]+offset, start_mask[i+2]-offset])
        
            

        new_start, new_end = choice(possible_locations)
        difference = new_end - new_start
        new_start = np.random.randint(0,difference) + new_start
        mask[new_start:new_end] = 1
        start_mask.append(new_start).sort()
        end_mask.append(new_end).sort()




        -> berechne dann: number_masking_tokens = number_masking_tokens - mask_width
    
    '''


    mask = np.zeros_like(x_array)
    length_x_array = len(x_array)
    start_mask = [0+offset]
    end_mask = [length_x_array-offset]
    # wert = np.random.normal(loc=number_masking_tokens/10, scale=number_masking_tokens/20)
    # wert = np.clip(wert, 0, number_masking_tokens)
    mask_width = np.random.randint(0,150+1)
    position = np.random.randint(start_mask[0], end_mask[0] - mask_width)
    start_mask.append(position)
    start_mask.sort()
    end_mask.append(position+mask_width)
    end_mask.sort()
    mask[position:position+mask_width] = 1
    number_masking_tokens = number_masking_tokens - mask_width

    while number_masking_tokens != 0:
        possible_locations = []
        while len(possible_locations) == 0:
            mask_width = np.random.randint(0, number_masking_tokens+1)
            if mask_width != 0 and mask_width < 150:
                free_mask_width = start_mask[1] - start_mask[0]
                if (free_mask_width-2*offset) > mask_width:
                    possible_locations.append([start_mask[0]+offset, start_mask[1]-offset])
                
                free_mask_width = end_mask[-1] - end_mask[-2]
                if (free_mask_width-2*offset) > mask_width:
                    possible_locations.append([end_mask[-2]+offset, end_mask[-1]-offset])

                assert len(start_mask) == len(end_mask)
                for i in range(len(start_mask)-2):
                    free_mask_width = start_mask[i+2] - end_mask[i]
                    if (free_mask_width-2*offset) > mask_width:
                        possible_locations.append([end_mask[i]+offset, start_mask[i+2]-offset])
        
            
        new_start, new_end = possible_locations[np.random.choice(np.arange(len(possible_locations)))]
        difference = new_end - mask_width
        new_start = np.random.randint(new_start,difference)
        mask[new_start:new_start+mask_width] = 1
        start_mask.append(new_start)
        start_mask.sort()
        end_mask.append(new_start+mask_width)
        end_mask.sort()
        number_masking_tokens = number_masking_tokens - mask_width
    

    return mask




def remove_parts_of_graph_encoder(x_array, y_array, width_array, offset, x_lim):
    '''
    width_array: [min_width, max_width, max_count_width]
    '''
    df = pd.DataFrame({'x_array': x_array, 'y_array': y_array, 'mask':0})
    location_width = pd.DataFrame(columns=['location', 'width', 'min_max_value'])
    count_width = randint(0,width_array[2]) #create random number of interpolation intervals
    min_width = width_array[0]  #minimal width of an interpolation interval
    max_width = width_array[1]  #maximal width of an interpolation interval


    for i in range(count_width):
        width = randint(min_width,max_width)    #create random width within the min and max width
        while width % 2 != 0:                   #make sure width is an even integer
            width = randint(min_width,max_width)
        
        #calculate random position marking the center of interpolation interval
        location = np.random.uniform(x_lim[0] + offset, x_lim[1] - offset)

        #divide width by 2 to be used as left side and right aide width regarding the location
        width *= 0.5

        #x boundaries without offset (0,1000);x boundaries with offset (10,990) 
        #make sure location and right-side as well left-side do not violate the x boundaries
        while (location - width < x_lim[0] + offset) or (location + width > x_lim[1] - offset):
            location = np.random.uniform(x_lim[0] + offset, x_lim[1] - offset)



        match location_width.empty:
            case True:
                location = location
            case False:
                count = 0
                #make sure that new interpolation intervals do not overlap with other existing interpolation intervals
                while any(
                    min(pair[0] - offset, pair[1] + offset) < (location + width) < max(pair[0] - offset, pair[1] + offset)
                    for pair in location_width['min_max_value'].tolist()
                ) or any(
                    min(pair[0] - offset, pair[1] + offset) < (location - width) < max(pair[0] - offset, pair[1] + offset)
                    for pair in location_width['min_max_value'].tolist()
                ):
                    count += 1
                    location = np.random.uniform(x_lim[0] + offset, x_lim[1] - offset)

                    #reduce the interpolation interval size in case of several failed attempts
                    if count % 5 == 0:
                        lower_border = min_width
                        upper_border = 2*min_width
                        assert upper_border < max_width
                        width = randint(lower_border,upper_border)
                        while width % 2 != 0:
                            width = randint(lower_border,upper_border)
                        width *= 0.5

        #add parameters to dataframe
        location_width.loc[len(location_width)] = [location, width * 2, [location - width, location + width]]

        #update mask values for interpolation intervals to 1
        df.loc[(df['x_array'] >= location - width) & (df['x_array'] <= location + width),'mask'] = 1


    return df['mask'].to_numpy()