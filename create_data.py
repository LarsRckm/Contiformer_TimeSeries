import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import math
from useful import calc_exp, sliding_window
from random import normalvariate, randint, uniform

#create an arbitrary time series without respect to y boundaries
def calculate_y_values_without_limitation(x_values, y_start, random_number_range):
    match random_number_range[0]:
        case "uni":
            random_numbers = np.random.uniform(random_number_range[1], random_number_range[2], len(x_values))
        case "norm":
            random_numbers = np.random.normal(random_number_range[1], random_number_range[2], len(x_values))
    output = np.array([y_start + random_numbers[0]])

    for i in range(1, len(x_values)):
        random_number = random_numbers[i]
        output = np.append(output, output[i - 1] + random_number)

    return np.transpose(output)

#create an arbitrary time series without respect to y boundaries (ready for model training --> dataset_timeseries --> callFunction())
def calculate_y_values_without_limitation_discontinuous(x_values, y_start, random_number_range):
    match random_number_range[0]:
        case "uni":
            random_numbers = np.random.uniform(random_number_range[1], random_number_range[2], len(x_values))
        case "norm":
            random_numbers = np.random.normal(random_number_range[1], random_number_range[2], len(x_values))
    output = np.array([y_start + random_numbers[0]])

    for i in range(1, len(x_values)):
        random_number = random_numbers[i]
        if i % 100 == 0:
            num = randint(0,4)
            #in 60% of cases there is no step added
            if num % 2 == 0:
                output = np.append(output, output[i - 1] + random_number)
            #in 40% cases there is a step added to the timeseries
            else:
                max_value = max(output)
                min_value = min(output)
                step = (max_value-min_value)*0.9
                num = randint(0,1)
                match num:
                    case 0:
                        output = np.append(output, output[i - 1] + random_number + step)
                    case 1:
                        output = np.append(output, output[i - 1] + random_number - step)
        else:
            output = np.append(output, output[i - 1] + random_number)

    return np.transpose(output)

#create an arbitrary time series without respect to y boundaries (ready for model training --> dataset_timeseries --> callFunction())
def calculate_y_values_without_limitation_discontinuous2(x_values, y_start, random_number_range, spline_value, step_factor):
    match random_number_range[0]:
        case "uni":
            random_numbers = np.random.uniform(random_number_range[1], random_number_range[2], len(x_values))
        case "norm":
            random_numbers = np.random.normal(random_number_range[1], random_number_range[2], len(x_values))
    output = np.array([y_start + random_numbers[0]])
    for i in range(1, len(x_values)):
        random_number = random_numbers[i]
        output = np.append(output, output[i - 1] + random_number)
        
    spline_value = np.random.uniform(spline_value[0], spline_value[1])
    spline = UnivariateSpline(x_values, output, s=spline_value)
    output = spline(x_values)
    previouschange = 0
    num = randint(0,40)


    for i in range(1, len(x_values)):
        #every 100th time series entry, there is a chance of making the time series discontinuous 
        step = (max(output)-min(output))*step_factor
        if i%10 ==0:
            num = randint(0,40)
        lower_border = 1
        upper_border = 2
        minIntervalWidth = randint(50,200)
        if num < lower_border:
            if i-previouschange > minIntervalWidth:
                output[i:] = output[i:]+step
                previouschange = i
        elif num >= lower_border and num < upper_border:
            if i-previouschange > minIntervalWidth:
                output[i:] = output[i:]-step
                previouschange = i
        else:
            None

    return np.transpose(output)

#create exponential time series (ready for model training --> dataset_timeseries --> callFunction())
def generate_noisy_data_exponential(x_values,vocab_size, noise_std):
    horizontal_scaling = [[10.0,100.0], [-100.0,-10.0]]
    horizontal_scaling = horizontal_scaling[randint(0,1)]
    horizontal_scaling = uniform(horizontal_scaling[0], horizontal_scaling[1])
    vertical_scaling = uniform(-2.0,2.0)
    vertical_setoff = uniform(-5.0,5.0)
    horizontal_setoff = uniform(-5.0,5.0)

    y_trend = vertical_scaling * np.exp((x_values + horizontal_setoff)/horizontal_scaling) + vertical_setoff
    
    match noise_std[0]:
        case "uni":
            noise_std_value = abs(uniform((max(y_trend)-min(y_trend))*noise_std[1],(max(y_trend)-min(y_trend))*noise_std[2]))
        case "norm":
            noise_std_value = abs(normalvariate(noise_std[1],(max(y_trend)-min(y_trend))*noise_std[2]))

    y_noise = np.random.normal(0, noise_std_value, len(x_values))

    y_trend_noise = y_trend + y_noise 


    exp = calc_exp(smallest_number=(1/vocab_size))
    #rounding max to ceil and min to floor to be able to display values properly
    max_value = math.ceil(max(max(y_trend_noise), max(y_trend))*(10**exp))/(10**exp)
    min_value = math.floor(min(min(y_trend_noise), min(y_trend))*(10**exp))/(10**exp)

    return y_trend, y_trend_noise, min_value, max_value, noise_std_value

#create time series based on distances (ready for model training --> dataset_timeseries --> callFunction())
def generate_noisy_data_distance(x_values, y_start,random_number_range, spline_value, vocab_size,noise_std):
    y_trend = calculate_y_values_without_limitation(x_values, y_start, random_number_range)
    spline_value = np.random.uniform(spline_value[0], spline_value[1])
    spline = UnivariateSpline(x_values, y_trend, s=spline_value)
    y_trend = spline(x_values)
    
    match noise_std[0]:
        case "uni":
            noise_std_value = abs(uniform((max(y_trend)-min(y_trend))*noise_std[1],(max(y_trend)-min(y_trend))*noise_std[2]))
        case "norm":
            noise_std_value = abs(normalvariate(noise_std[1],(max(y_trend)-min(y_trend))*noise_std[2]))
            
    noise_total = np.random.normal(0, noise_std_value, len(x_values))
    y_noisy_spline = y_trend + noise_total  #noise + splineGroundTruth
    
    #calculate exponent for rounding
    exp = calc_exp(smallest_number=(1/vocab_size))
    
    max_value_spline = math.ceil(max(max(y_noisy_spline), max(y_trend))*(10**exp))/(10**exp)
    min_value_spline = math.floor(min(min(y_noisy_spline), min(y_trend))*(10**exp))/(10**exp)


    return y_trend, y_noisy_spline,min_value_spline, max_value_spline, noise_std_value

#create discontinuous timeseries (ready for model training --> dataset_timeseries --> callFunction())
def generate_discontinous_timeseries(x_values, y_start,random_number_range, spline_value, vocab_size,noise_std):
    y_trend = calculate_y_values_without_limitation_discontinuous(x_values, y_start, random_number_range)

    spline_value = np.random.uniform(spline_value[0], spline_value[1])
    spline = UnivariateSpline(x_values, y_trend, s=spline_value)
    y_trend = spline(x_values)
    
    match noise_std[0]:
        case "uni":
            noise_std_value = abs(uniform((max(y_trend)-min(y_trend))*noise_std[1],(max(y_trend)-min(y_trend))*noise_std[2]))
        case "norm":
            noise_std_value = abs(normalvariate(noise_std[1],(max(y_trend)-min(y_trend))*noise_std[2]))

    noise_total = np.random.normal(0, noise_std_value, len(x_values))
    y_noisy_spline = y_trend + noise_total  #noise + splineGroundTruth
    
    #calculate exponent for rounding
    exp = calc_exp(smallest_number=(1/vocab_size))
    
    max_value_spline = math.ceil(max(max(y_noisy_spline), max(y_trend))*(10**exp))/(10**exp)
    min_value_spline = math.floor(min(min(y_noisy_spline), min(y_trend))*(10**exp))/(10**exp)


    return y_trend, y_noisy_spline,min_value_spline, max_value_spline, noise_std_value

def generate_discontinous_timeseries2(x_values, y_start,random_number_range, spline_value, vocab_size, noise_std):
    y_trend = calculate_y_values_without_limitation_discontinuous2(x_values, y_start, random_number_range, spline_value, 0.8)
    
    match noise_std[0]:
        case "uni":
            noise_std_value = abs(uniform((max(y_trend)-min(y_trend))*noise_std[1],(max(y_trend)-min(y_trend))*noise_std[2]))
        case "norm":
            noise_std_value = abs(normalvariate(noise_std[1],(max(y_trend)-min(y_trend))*noise_std[2]))

    noise_total = np.random.normal(0, noise_std_value, len(x_values))
    y_noisy_spline = y_trend + noise_total  #noise + splineGroundTruth
    
    #calculate exponent for rounding
    exp = calc_exp(smallest_number=(1/vocab_size))
    
    max_value_spline = math.ceil(max(max(y_noisy_spline), max(y_trend))*(10**exp))/(10**exp)
    min_value_spline = math.floor(min(min(y_noisy_spline), min(y_trend))*(10**exp))/(10**exp)


    return y_trend, y_noisy_spline,min_value_spline, max_value_spline, noise_std_value
    
#create periodic sinus time series
def generate_noisy_data_periodic(x_values, vocab_size, noise_std):
    horizontal_scaling = [[10.0,100.0], [-100.0,-10.0]]
    horizontal_scaling = horizontal_scaling[randint(0,1)]
    horizontal_scaling = uniform(horizontal_scaling[0], horizontal_scaling[1])
    vertical_scaling = uniform(-200.0,200.0)
    vertical_setoff = uniform(-50.0,50.0)
    horizontal_setoff = uniform(-50.0,50.0)

    y_trend = vertical_scaling * np.sin((x_values+horizontal_setoff)/horizontal_scaling) + vertical_setoff


    match noise_std[0]:
        case "uni":
            noise_std_value = abs(uniform((max(y_trend)-min(y_trend))*noise_std[1],(max(y_trend)-min(y_trend))*noise_std[2]))
        case "norm":
            noise_std_value = abs(normalvariate(noise_std[1],(max(y_trend)-min(y_trend))*noise_std[2]))

    y_noise = np.random.normal(0, noise_std_value, len(x_values))

    y_trend_noise = y_trend + y_noise

    exp = calc_exp(smallest_number=(1/vocab_size))
    #rounding max to ceil and min to floor to be able to display values properly
    max_value = math.ceil(max(max(y_trend_noise), max(y_trend))*(10**exp))/(10**exp)
    min_value = math.floor(min(min(y_trend_noise), min(y_trend))*(10**exp))/(10**exp)

    return y_trend, y_trend_noise, min_value, max_value, noise_std_value

#create time series based on slope intervals
def generate_noisy_data_slope(x_values, y_start, lower_border, upper_border, vocab_size, noise_std):
    y_trend = [y_start]
    while len(y_trend)<len(x_values):
        length_difference = len(x_values)-len(y_trend)
        new_interval_len = uniform(0,length_difference)
        if length_difference < 20:
            new_interval_len = length_difference+1
        slope= uniform(lower_border, upper_border)
        last_value = y_trend[-1]
        new_values = np.arange(1, new_interval_len, 1) * slope + last_value
        y_trend = np.append(y_trend, new_values)
    

    for i in range(3):
        y_trend = sliding_window(y_trend, 5)

    match noise_std[0]:
        case "uni":
            noise_std_value = abs(uniform((max(y_trend)-min(y_trend))*noise_std[1],(max(y_trend)-min(y_trend))*noise_std[2]))
        case "norm":
            noise_std_value = abs(normalvariate(noise_std[1],(max(y_trend)-min(y_trend))*noise_std[2]))

    y_noise = np.random.normal(0, noise_std_value, len(x_values))

    y_trend_noise = y_trend + y_noise

    exp = calc_exp(smallest_number=(1/vocab_size))
    #rounding max to ceil and min to floor to be able to display values properly
    max_value = math.ceil(max(max(y_trend_noise), max(y_trend))*(10**exp))/(10**exp)
    min_value = math.floor(min(min(y_trend_noise), min(y_trend))*(10**exp))/(10**exp)

    return y_trend, y_trend_noise, min_value, max_value, noise_std_value
    
    #am ende wende ich noch ein moving average auf den erzeugten trend an, um den graphen kontinuierlicher aussehen zu lassen

#create periodic sinus time series
def generate_noisy_data_periodic_sum(x_values, vocab_size, noise_std):
    horizontal_scaling = [[10.0,100.0], [-100.0,-10.0]]
    horizontal_scaling = horizontal_scaling[randint(0,1)]
    horizontal_scaling = uniform(horizontal_scaling[0], horizontal_scaling[1])
    vertical_scaling = uniform(-200.0,200.0)
    vertical_setoff = uniform(-500.0,500.0)
    horizontal_setoff = uniform(-500.0,500.0)

    y_trend = vertical_scaling * np.sin((x_values+horizontal_setoff)/horizontal_scaling) + vertical_setoff

    for i in range(randint(5,10)):
        horizontal_scaling = [[10.0,100.0], [-100.0,-10.0]]
        horizontal_scaling = horizontal_scaling[randint(0,1)]
        horizontal_scaling = uniform(horizontal_scaling[0], horizontal_scaling[1])
        vertical_scaling = uniform(-200.0,200.0)
        vertical_setoff = uniform(-50.0,50.0)
        horizontal_setoff = uniform(-50.0,50.0)

        y_trend += vertical_scaling * np.sin((x_values+horizontal_setoff)/horizontal_scaling) + vertical_setoff

    match noise_std[0]:
        case "uni":
            noise_std_value = abs(uniform((max(y_trend)-min(y_trend))*noise_std[1],(max(y_trend)-min(y_trend))*noise_std[2]))
        case "norm":
            noise_std_value = abs(normalvariate(noise_std[1],(max(y_trend)-min(y_trend))*noise_std[2]))

    y_noise = np.random.normal(0, noise_std_value, len(x_values))

    y_trend_noise = y_trend + y_noise

    exp = calc_exp(smallest_number=(1/vocab_size))
    #rounding max to ceil and min to floor to be able to display values properly
    max_value = math.ceil(max(max(y_trend_noise), max(y_trend))*(10**exp))/(10**exp)
    min_value = math.floor(min(min(y_trend_noise), min(y_trend))*(10**exp))/(10**exp)

    return y_trend, y_trend_noise, min_value, max_value, noise_std_value

def callFunction(x_values, y_start,random_number_range, spline_value, vocab_size, randomInt, noise_std):
    match randomInt:
        case 0:
            lower_slope = -500.0
            upper_slope = 500.0
            return generate_noisy_data_slope(x_values, y_start, lower_slope, upper_slope, vocab_size, noise_std)
        case 1:
            spline_value = [800000,1100000]
            return generate_noisy_data_distance(x_values, y_start,random_number_range, spline_value, vocab_size,noise_std)
        case 2:
            return generate_noisy_data_exponential(x_values, vocab_size,noise_std)
        case 3:
            return generate_noisy_data_periodic(x_values, vocab_size,noise_std)
        case 4:
            spline_value = [100000,200000]
            return generate_noisy_data_distance(x_values, y_start,random_number_range, spline_value, vocab_size,noise_std)
        case 5:
            return generate_noisy_data_periodic_sum(x_values,vocab_size,noise_std)
        case 6:
            spline_value = [800000,1100000] 
            return generate_discontinous_timeseries(x_values, y_start,random_number_range, spline_value, vocab_size,noise_std)
        case 7:
            spline_value = [800000,1100000] 
            return generate_discontinous_timeseries2(x_values, y_start,random_number_range, spline_value, vocab_size,noise_std)
